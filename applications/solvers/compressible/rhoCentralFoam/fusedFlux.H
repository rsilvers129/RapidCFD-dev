/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | RapidCFD: GPU-accelerated OpenFOAM
   \\    /   O peration     |
    \\  /    A nd           | Fused Kurganov-Tadmor flux computation
     \\/     M anipulation  | (opus-fused-kernels branch)
\*---------------------------------------------------------------------------*/

#ifndef fusedFlux_H
#define fusedFlux_H

#include "fvCFD.H"
#include <cuda_runtime.h>

namespace Foam
{

// Vector component access for AoS layout
// Foam::vector stores [x, y, z] contiguously as 3 scalars
struct vec3 { scalar x, y, z; };

// -------------------------------------------------------------------------
// CUDA kernel: one thread per internal face
// Fuses upwind interpolation + all Kurganov-Tadmor flux arithmetic
// -------------------------------------------------------------------------
__global__ void fusedKTFluxKernel
(
    // Cell fields (read) — scalar fields
    const scalar* __restrict__ rhoCell,
    const vec3*   __restrict__ rhoUCell,    // rhoU as vector
    const scalar* __restrict__ eCell,
    const scalar* __restrict__ psiCell,
    const scalar* __restrict__ cCell,       // speed of sound
    // Mesh connectivity (read)
    const label*  __restrict__ own,
    const label*  __restrict__ nei,
    // Face geometry (read)
    const vec3*   __restrict__ Sf,          // face area vector
    const scalar* __restrict__ magSfArr,
    // Outputs (write) — internal faces only
    scalar* __restrict__ phiOut,            // mass flux
    vec3*   __restrict__ phiUpOut,          // momentum flux
    scalar* __restrict__ phiEpOut,          // energy flux
    scalar* __restrict__ amaxSfOut,         // max wave speed * area
    scalar* __restrict__ aPosFaceOut,       // a_pos per face
    scalar* __restrict__ aNegFaceOut,       // a_neg per face
    vec3*   __restrict__ UPosOut,           // U_pos per face
    vec3*   __restrict__ UNegOut,           // U_neg per face
    // Config
    label nFaces,
    int isTadmor
)
{
    const label face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= nFaces) return;

    const label o = own[face];
    const label n = nei[face];

    // --- Step 1: Read cell data (AoS) ---
    const scalar rho_p   = rhoCell[o];
    const scalar rho_n   = rhoCell[n];
    const vec3   rhoU_p  = rhoUCell[o];
    const vec3   rhoU_n  = rhoUCell[n];
    const scalar e_p     = eCell[o];
    const scalar e_n     = eCell[n];
    const scalar psi_p   = psiCell[o];
    const scalar psi_n   = psiCell[n];
    const scalar c_o     = cCell[o];
    const scalar c_n_val = cCell[n];

    // --- Step 2: Derived quantities (all in registers) ---
    const scalar rPsi_p = scalar(1.0) / psi_p;
    const scalar rPsi_n = scalar(1.0) / psi_n;

    // U = rhoU / rho
    const scalar invRho_p = scalar(1.0) / rho_p;
    const scalar invRho_n = scalar(1.0) / rho_n;
    const scalar Ux_p = rhoU_p.x * invRho_p;
    const scalar Uy_p = rhoU_p.y * invRho_p;
    const scalar Uz_p = rhoU_p.z * invRho_p;
    const scalar Ux_n = rhoU_n.x * invRho_n;
    const scalar Uy_n = rhoU_n.y * invRho_n;
    const scalar Uz_n = rhoU_n.z * invRho_n;

    // p = rho / psi = rho * rPsi
    const scalar p_p = rho_p * rPsi_p;
    const scalar p_n = rho_n * rPsi_n;

    // Face geometry
    const vec3   sf  = Sf[face];
    const scalar mSf = magSfArr[face];

    // phiv = U . Sf
    const scalar phiv_p = Ux_p * sf.x + Uy_p * sf.y + Uz_p * sf.z;
    const scalar phiv_n = Ux_n * sf.x + Uy_n * sf.y + Uz_n * sf.z;

    // cSf = c * |Sf|
    const scalar cSf_p = c_o * mSf;
    const scalar cSf_n = c_n_val * mSf;

    // --- Step 3: Wave speed estimates ---
    scalar ap = fmax(fmax(phiv_p + cSf_p, phiv_n + cSf_n), scalar(0.0));
    scalar am = fmin(fmin(phiv_p - cSf_p, phiv_n - cSf_n), scalar(0.0));

    // --- Step 4: Kurganov-Tadmor weighting ---
    scalar a_pos, a_neg, aSf_val;

    if (isTadmor)
    {
        scalar amaxSfLocal = fmax(ap, -am);
        aSf_val = scalar(-0.5) * amaxSfLocal;
        a_pos = scalar(0.5);
        a_neg = scalar(0.5);
    }
    else
    {
        // Kurganov: a_pos = ap / (ap - am)
        // Guard against ap == am == 0
        const scalar denom = ap - am;
        a_pos = (denom > scalar(1e-30)) ? (ap / denom) : scalar(0.5);
        aSf_val = am * a_pos;
        a_neg = scalar(1.0) - a_pos;
    }

    const scalar aphiv_p = a_pos * phiv_p - aSf_val;
    const scalar aphiv_n = a_neg * phiv_n + aSf_val;

    // amaxSf for Courant number
    const scalar amaxSfVal = fmax(fabs(aphiv_p), fabs(aphiv_n));

    // --- Step 5: Output fluxes ---

    // Mass flux
    phiOut[face] = aphiv_p * rho_p + aphiv_n * rho_n;

    // Momentum flux
    const scalar pFace = a_pos * p_p + a_neg * p_n;
    vec3 phiUpVal;
    phiUpVal.x = aphiv_p * rhoU_p.x + aphiv_n * rhoU_n.x + pFace * sf.x;
    phiUpVal.y = aphiv_p * rhoU_p.y + aphiv_n * rhoU_n.y + pFace * sf.y;
    phiUpVal.z = aphiv_p * rhoU_p.z + aphiv_n * rhoU_n.z + pFace * sf.z;
    phiUpOut[face] = phiUpVal;

    // Energy flux
    const scalar magSqU_p = Ux_p*Ux_p + Uy_p*Uy_p + Uz_p*Uz_p;
    const scalar magSqU_n = Ux_n*Ux_n + Uy_n*Uy_n + Uz_n*Uz_n;
    phiEpOut[face] = aphiv_p * (rho_p * (e_p + scalar(0.5) * magSqU_p) + p_p)
                   + aphiv_n * (rho_n * (e_n + scalar(0.5) * magSqU_n) + p_n)
                   + aSf_val * (p_p - p_n);

    amaxSfOut[face] = amaxSfVal;

    // Store for sigmaDotU
    aPosFaceOut[face] = a_pos;
    aNegFaceOut[face] = a_neg;
    vec3 up = {Ux_p, Uy_p, Uz_p};
    vec3 un = {Ux_n, Uy_n, Uz_n};
    UPosOut[face] = up;
    UNegOut[face] = un;
}

// -------------------------------------------------------------------------
// Host wrapper — called from rhoCentralFoam.C
// -------------------------------------------------------------------------
void launchFusedFluxKernel
(
    const volScalarField& rho,
    const volVectorField& rhoU,
    const volScalarField& e,
    const volScalarField& psi,
    const volScalarField& c,
    const fvMesh& mesh,
    surfaceScalarField& phi,
    surfaceVectorField& phiUp,
    surfaceScalarField& phiEp,
    surfaceScalarField& amaxSf,
    surfaceScalarField& a_pos_f,
    surfaceScalarField& a_neg_f,
    surfaceVectorField& U_pos_f,
    surfaceVectorField& U_neg_f,
    bool isTadmor
)
{
    const label nIF = mesh.nInternalFaces();
    if (nIF == 0) return;

    // Raw device pointers — cast through void* for ABI safety
    const scalar* rhoPtr  = rho.getField().data();
    const vec3*   rhoUPtr = reinterpret_cast<const vec3*>(rhoU.getField().data());
    const scalar* ePtr    = e.getField().data();
    const scalar* psiPtr  = psi.getField().data();
    const scalar* cPtr    = c.getField().data();

    const label* ownPtr = mesh.owner().data();
    const label* neiPtr = mesh.neighbour().data();

    const vec3*   SfPtr   = reinterpret_cast<const vec3*>(mesh.Sf().getField().data());
    const scalar* magSfPtr = mesh.magSf().getField().data();

    scalar* phiPtr    = phi.getField().data();
    vec3*   phiUpPtr  = reinterpret_cast<vec3*>(phiUp.getField().data());
    scalar* phiEpPtr  = phiEp.getField().data();
    scalar* amaxSfPtr = amaxSf.getField().data();
    scalar* aPosPtr   = a_pos_f.getField().data();
    scalar* aNegPtr   = a_neg_f.getField().data();
    vec3*   UPosPtr   = reinterpret_cast<vec3*>(U_pos_f.getField().data());
    vec3*   UNegPtr   = reinterpret_cast<vec3*>(U_neg_f.getField().data());

    const int blockSize = 256;
    const int gridSize = (nIF + blockSize - 1) / blockSize;

    fusedKTFluxKernel<<<gridSize, blockSize>>>
    (
        rhoPtr, rhoUPtr, ePtr, psiPtr, cPtr,
        ownPtr, neiPtr,
        SfPtr, magSfPtr,
        phiPtr, phiUpPtr, phiEpPtr, amaxSfPtr,
        aPosPtr, aNegPtr, UPosPtr, UNegPtr,
        nIF, isTadmor ? 1 : 0
    );
}

} // End namespace Foam

#endif
