/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | RapidCFD: GPU-accelerated OpenFOAM
   \\    /   O peration     |
    \\  /    A nd           | Fused viscous flux (sigmaDotU) computation
     \\/     M anipulation  | (Gemini branch)
\*---------------------------------------------------------------------------*/

#ifndef fusedViscFlux_H
#define fusedViscFlux_H

#include "fvCFD.H"
#include <cuda_runtime.h>

namespace Foam
{

struct tensor9 { scalar xx, xy, xz, yx, yy, yz, zx, zy, zz; };
// vec3 is defined in fusedFlux.H which is included before this in rhoCentralFoam.C

// -------------------------------------------------------------------------
// CUDA kernel: one thread per internal face
// Fuses upwind interpolation of muEff, tauMC and computes sigmaDotU
// -------------------------------------------------------------------------
__global__ void fusedViscFluxKernel
(
    // Cell fields (read)
    const scalar*  __restrict__ muEffCell,
    const tensor9* __restrict__ tauMCCell,
    // Face data (read)
    const label*   __restrict__ own,
    const label*   __restrict__ nei,
    const scalar*  __restrict__ weights,    // mesh.weights()
    const vec3*    __restrict__ Sf,         // mesh.Sf() -> area vector
    const scalar*  __restrict__ magSf,      // mesh.magSf() -> area magnitude
    const vec3*    __restrict__ snGradU,    // fvc::snGrad(U) (precomputed)
    // Saved from inviscid flux computation
    const scalar*  __restrict__ aPos,       // a_pos
    const scalar*  __restrict__ aNeg,       // a_neg
    const vec3*    __restrict__ UPos,       // U_pos
    const vec3*    __restrict__ UNeg,       // U_neg
    // Output (write)
    scalar*        __restrict__ sigmaDotUOut,
    // Config
    label nFaces
)
{
    const label face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= nFaces) return;

    const label o = own[face];
    const label n = nei[face];

    // Read face variables
    const scalar w_f = weights[face];
    const scalar o_w = w_f;
    const scalar n_w = scalar(1.0) - w_f;

    // Interpolate muEff to face
    const scalar muEff_o = muEffCell[o];
    const scalar muEff_n = muEffCell[n];
    const scalar muEff_f = o_w * muEff_o + n_w * muEff_n;

    const tensor9 tau_o = tauMCCell[o];
    const tensor9 tau_n = tauMCCell[n];

    tensor9 tau_f;
    tau_f.xx = o_w * tau_o.xx + n_w * tau_n.xx;
    tau_f.xy = o_w * tau_o.xy + n_w * tau_n.xy;
    tau_f.xz = o_w * tau_o.xz + n_w * tau_n.xz;
    tau_f.yx = o_w * tau_o.yx + n_w * tau_n.yx;
    tau_f.yy = o_w * tau_o.yy + n_w * tau_n.yy;
    tau_f.yz = o_w * tau_o.yz + n_w * tau_n.yz;
    tau_f.zx = o_w * tau_o.zx + n_w * tau_n.zx;
    tau_f.zy = o_w * tau_o.zy + n_w * tau_n.zy;
    tau_f.zz = o_w * tau_o.zz + n_w * tau_n.zz;

    const vec3 sf = Sf[face];
    const scalar mSf = magSf[face];
    const vec3 sn = snGradU[face];

    // Evaluate: muEff_f * magSf * snGradU
    vec3 part1;
    part1.x = muEff_f * mSf * sn.x;
    part1.y = muEff_f * mSf * sn.y;
    part1.z = muEff_f * mSf * sn.z;

    // Evaluate: Sf & tau_f
    // OpenFOAM vector & tensor is: v.x*t.xx + v.y*t.yx + v.z*t.zx ...
    vec3 part2;
    part2.x = sf.x * tau_f.xx + sf.y * tau_f.yx + sf.z * tau_f.zx;
    part2.y = sf.x * tau_f.xy + sf.y * tau_f.yy + sf.z * tau_f.zy;
    part2.z = sf.x * tau_f.xz + sf.y * tau_f.yz + sf.z * tau_f.zz;

    vec3 sigma_f;
    sigma_f.x = part1.x + part2.x;
    sigma_f.y = part1.y + part2.y;
    sigma_f.z = part1.z + part2.z;

    const scalar ap = aPos[face];
    const scalar an = aNeg[face];
    const vec3   up = UPos[face];
    const vec3   un = UNeg[face];

    vec3 U_fused;
    U_fused.x = ap * up.x + an * un.x;
    U_fused.y = ap * up.y + an * un.y;
    U_fused.z = ap * up.z + an * un.z;

    // Final dot product
    sigmaDotUOut[face] = sigma_f.x * U_fused.x + sigma_f.y * U_fused.y + sigma_f.z * U_fused.z;
}

// -------------------------------------------------------------------------
// Host wrapper — called from rhoCentralFoam.C
// -------------------------------------------------------------------------
inline void launchFusedViscFluxKernel
(
    const volScalarField& muEff,
    const volTensorField& tauMC,
    const surfaceVectorField& snGradU,
    const fvMesh& mesh,
    const surfaceScalarField& a_pos_f,
    const surfaceScalarField& a_neg_f,
    const surfaceVectorField& U_pos_f,
    const surfaceVectorField& U_neg_f,
    surfaceScalarField& sigmaDotU
)
{
    const label nIF = mesh.nInternalFaces();
    if (nIF == 0) return;

    // Raw device pointers (assume Single/Double precision handled via Foams scalar)
    const scalar* muEffPtr = muEff.getField().data();
    const tensor9* tauMCPtr = reinterpret_cast<const tensor9*>(tauMC.getField().data());

    const label* ownPtr = mesh.owner().data();
    const label* neiPtr = mesh.neighbour().data();

    const scalar* weightsPtr = mesh.surfaceInterpolation::weights().getField().data();
    const vec3* SfPtr = reinterpret_cast<const vec3*>(mesh.Sf().getField().data());
    const scalar* magSfPtr = mesh.magSf().getField().data();
    
    const vec3* snGradUPtr = reinterpret_cast<const vec3*>(snGradU.getField().data());

    const scalar* aPosPtr = a_pos_f.getField().data();
    const scalar* aNegPtr = a_neg_f.getField().data();
    const vec3* UPosPtr = reinterpret_cast<const vec3*>(U_pos_f.getField().data());
    const vec3* UNegPtr = reinterpret_cast<const vec3*>(U_neg_f.getField().data());

    scalar* sigmaDotUPtr = sigmaDotU.getField().data();

    const int blockSize = 256;
    const int gridSize = (nIF + blockSize - 1) / blockSize;

    fusedViscFluxKernel<<<gridSize, blockSize>>>(
        muEffPtr, tauMCPtr, ownPtr, neiPtr,
        weightsPtr, SfPtr, magSfPtr, snGradUPtr,
        aPosPtr, aNegPtr, UPosPtr, UNegPtr,
        sigmaDotUPtr, nIF
    );
}

} // End namespace Foam

#endif
