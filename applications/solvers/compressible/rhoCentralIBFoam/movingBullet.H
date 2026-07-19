/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | RapidCFD: GPU-accelerated OpenFOAM  (wopr-cuda)
   \\    /   O peration     |
    \\  /    A nd           | Prescribed-motion immersed-boundary "bullet"
     \\/     M anipulation  | (Phase 1 — Track R moving projectile)
\*---------------------------------------------------------------------------*/
/*
  A rigid CAPSULE (cylinder of radius R with hemispherical caps) translating
  along the bore axis (z) at prescribed velocity v. Each timestep the cells
  inside the capsule are forced to the solid state (velocity = bullet velocity),
  so the gas sees a moving impermeable body: compression / bow shock ahead, a
  low-pressure wake behind, and the plugging-then-clearing of each baffle bore
  (the uncork). Motion is PRESCRIBED (no 6-DOF) — physically correct here, since
  the gas changes the ~890 m/s bullet speed negligibly over the ~150 us transit.

  This is a STATIC background mesh + moving MASK (no overset, no remesh). The
  fixed geometry (baffles, vent holes, walls) is body-fitted and exact; only the
  bullet is immersed.

  v1 = direct velocity forcing (simple, robust). The bullet base/wake is a strong
  near-vacuum, so this leans on the density/energy floor already in the solver.
  Known refinements to evaluate after first runs (see enforceMovingBullet.H):
    - proper ghost/reflected wall state at the interface (vs. plain forcing),
    - explicit fresh-cell extrapolation when the bullet vacates a cell
      (v1 relies on the imposed solid state — gas at bullet velocity — being a
       reasonable seed for the wake; refine if the wake shows spurious noise),
    - exact ogive nose via the projectile STL SDF (v1 uses an analytic capsule).
*/

#ifndef movingBullet_H
#define movingBullet_H

#include "fvCFD.H"
#include <cuda_runtime.h>

namespace Foam
{

// local POD vector (fusedFlux.H already defines vec3; use a distinct name)
struct bvec3 { scalar x, y, z; };

// -------------------------------------------------------------------------
// One thread per cell. Tests the cell centre against the capsule and, if
// inside, overwrites the cell to the solid (bullet) state. Also writes a
// 0/1 solidMask for diagnostics / rendering (exclude solid cells, draw bullet).
//
// Capsule: axis segment on the z-axis from zNose (leading, most -z) to
// zTail = zNose + length (trailing). A point is inside iff its distance to
// that segment <= R, i.e.  x^2 + y^2 + dz^2 <= R^2  with dz the axial overshoot
// past either cap.
// -------------------------------------------------------------------------
__global__ void enforceBulletKernel
(
    const bvec3* __restrict__ C,        // cell centres
    scalar*      __restrict__ rho,
    bvec3*       __restrict__ U,
    bvec3*       __restrict__ rhoU,
    scalar*      __restrict__ rhoE,
    scalar*      __restrict__ e,
    scalar*      __restrict__ solidMask,
    const scalar zNose,
    const scalar bulletL,               // bullet length (nose s=0 -> tail s=L)
    const scalar* __restrict__ radii,   // R(s) profile, nBins samples over [0,L]
    const int     nBins,
    const scalar vBullet,               // z-velocity (< 0 = downrange)
    const scalar rhoMin,
    const scalar eSolid,
    const label  nCells
)
{
    const label c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nCells) return;

    const bvec3 x = C[c];

    // Real axisymmetric ogive: s = axial distance behind the nose (0..L). Look up
    // the bullet radius R(s) from the STL-derived profile and test r < R(s). This
    // gives the true POINTED nose (attached oblique shock / Mach cone) instead of a
    // blunt capsule (detached bow shock).
    const scalar s = x.z - zNose;
    bool inside = false;
    if (s >= scalar(0) && s <= bulletL)
    {
        scalar f = s / bulletL * scalar(nBins - 1);
        int i0 = (int)f;
        if (i0 < 0) i0 = 0;
        if (i0 > nBins - 2) i0 = nBins - 2;
        const scalar frac = f - scalar(i0);
        const scalar Rs = radii[i0]*(scalar(1) - frac) + radii[i0 + 1]*frac;
        const scalar rr = x.x*x.x + x.y*x.y;
        inside = (rr <= Rs*Rs);
    }

    if (inside)
    {
        // --- inside the bullet: force the solid state ---
        const scalar rc = (rho[c] > rhoMin) ? rho[c] : rhoMin;
        const scalar ec = (e[c]   > eSolid) ? e[c]   : eSolid;
        rho[c]    = rc;
        e[c]      = ec;
        U[c].x    = scalar(0); U[c].y = scalar(0); U[c].z = vBullet;
        rhoU[c].x = scalar(0); rhoU[c].y = scalar(0); rhoU[c].z = rc*vBullet;
        rhoE[c]   = rc*(ec + scalar(0.5)*vBullet*vBullet);
        solidMask[c] = scalar(1);
    }
    else
    {
        solidMask[c] = scalar(0);
    }
}

// -------------------------------------------------------------------------
inline void launchEnforceBullet
(
    const volVectorField& C,            // mesh.C()
    volScalarField&       rho,
    volVectorField&       U,
    volVectorField&       rhoU,
    volScalarField&       rhoE,
    volScalarField&       e,
    volScalarField&       solidMask,
    const scalar zNose,
    const scalar bulletL,
    const scalar* radiiDev,             // device ptr: R(s) profile
    const int    nBins,
    const scalar vBullet,
    const scalar rhoMin,
    const scalar eSolid
)
{
    const label nCells = rho.size();
    if (nCells == 0) return;

    const int blockSize = 256;
    const int gridSize  = (nCells + blockSize - 1) / blockSize;

    enforceBulletKernel<<<gridSize, blockSize>>>
    (
        reinterpret_cast<const bvec3*>(C.getField().data()),
        rho.getField().data(),
        reinterpret_cast<bvec3*>(U.getField().data()),
        reinterpret_cast<bvec3*>(rhoU.getField().data()),
        rhoE.getField().data(),
        e.getField().data(),
        solidMask.getField().data(),
        zNose, bulletL, radiiDev, nBins, vBullet, rhoMin, eSolid, nCells
    );
}

} // End namespace Foam

#endif

// ************************************************************************* //
