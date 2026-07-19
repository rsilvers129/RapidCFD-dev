/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | RapidCFD: GPU-accelerated OpenFOAM
   \\    /   O peration     |
    \\  /    A nd           | Fused post-solve cell-level arithmetic
     \\/     M anipulation  | Eliminates ~8 separate Thrust kernel launches
\*---------------------------------------------------------------------------*/

#ifndef fusedPostSolve_H
#define fusedPostSolve_H

#include "fvCFD.H"
#include <cuda_runtime.h>

namespace Foam
{

// -------------------------------------------------------------------------
// CUDA kernel: fuses e=rhoE/rho-0.5*|U|², p=rho/psi
// One thread per cell. Replaces ~8 individual Thrust kernel launches.
// NOTE: Does NOT compute U — that is handled by the momentum solve
//       and viscous correction steps earlier in the timestep.
// -------------------------------------------------------------------------
__global__ void fusedEandPKernel
(
    // Inputs (read)
    const scalar* __restrict__ rhoIn,
    const vec3*   __restrict__ UIn,
    const scalar* __restrict__ rhoEIn,
    const scalar* __restrict__ psiIn,
    // Outputs (write)
    scalar* __restrict__ eOut,
    scalar* __restrict__ pOut,
    // Config
    label nCells
)
{
    const label cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    const scalar rho_c  = rhoIn[cell];
    const vec3   U_c    = UIn[cell];
    const scalar rhoE_c = rhoEIn[cell];
    const scalar psi_c  = psiIn[cell];

    // e = rhoE/rho - 0.5*|U|^2
    const scalar magSqU = U_c.x*U_c.x + U_c.y*U_c.y + U_c.z*U_c.z;
    eOut[cell] = rhoE_c / rho_c - scalar(0.5) * magSqU;

    // p = rho / psi
    pOut[cell] = rho_c / psi_c;
}

// -------------------------------------------------------------------------
// Fused rhoE update: rhoE = rho*(e + 0.5*|U|^2)
// -------------------------------------------------------------------------
__global__ void fusedRhoEUpdateKernel
(
    const scalar* __restrict__ rhoIn,
    const scalar* __restrict__ eIn,
    const vec3*   __restrict__ UIn,
    scalar* __restrict__ rhoEOut,
    label nCells
)
{
    const label cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nCells) return;

    const scalar rho_c = rhoIn[cell];
    const scalar e_c   = eIn[cell];
    const vec3   U_c   = UIn[cell];

    const scalar magSqU = U_c.x*U_c.x + U_c.y*U_c.y + U_c.z*U_c.z;
    rhoEOut[cell] = rho_c * (e_c + scalar(0.5) * magSqU);
}

// -------------------------------------------------------------------------
// Host wrappers
// -------------------------------------------------------------------------
inline void launchFusedEandP
(
    const volScalarField& rho,
    const volVectorField& U,
    const volScalarField& rhoE,
    const volScalarField& psi,
    volScalarField& e,
    volScalarField& p
)
{
    const label nCells = rho.size();
    if (nCells == 0) return;

    const int blockSize = 256;
    const int gridSize = (nCells + blockSize - 1) / blockSize;

    fusedEandPKernel<<<gridSize, blockSize>>>(
        rho.getField().data(),
        reinterpret_cast<const vec3*>(U.getField().data()),
        rhoE.getField().data(),
        psi.getField().data(),
        e.getField().data(),
        p.getField().data(),
        nCells
    );
}

inline void launchFusedRhoEUpdate
(
    const volScalarField& rho,
    const volScalarField& e,
    const volVectorField& U,
    volScalarField& rhoE
)
{
    const label nCells = rho.size();
    if (nCells == 0) return;

    const int blockSize = 256;
    const int gridSize = (nCells + blockSize - 1) / blockSize;

    fusedRhoEUpdateKernel<<<gridSize, blockSize>>>(
        rho.getField().data(),
        e.getField().data(),
        reinterpret_cast<const vec3*>(U.getField().data()),
        rhoE.getField().data(),
        nCells
    );
}

} // End namespace Foam

#endif
