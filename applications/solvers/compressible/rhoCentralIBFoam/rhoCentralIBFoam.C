/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2014 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    rhoCentralFoam

Description
    Density-based compressible flow solver based on central-upwind schemes of
    Kurganov and Tadmor.

    GPU-optimized with fused flux kernel (opus-fused-kernels branch).

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "psiThermo.H"
#include "turbulenceModel.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedRhoFvPatchScalarField.H"
#include "fusedFlux.H"
#include "fusedViscFlux.H"
#include "fusedPostSolve.H"
#include "movingBullet.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"

    #include "createTime.H"
    #include "createMesh.H"
    #include "createFields.H"
    #include "readTimeControls.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    #include "readFluxScheme.H"

    dimensionedScalar v_zero("v_zero", dimVolume/dimTime, 0.0);

    bool isTadmor = (fluxScheme == "Tadmor");

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        // --- Compute speed of sound (cell field, needed by fused kernel)
        volScalarField rPsi("rPsi", 1.0/psi);
        volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));

        // --- Allocate output surface fields for fused kernel ---
        // These are filled by the CUDA kernel for internal faces.
        // Boundary faces are handled by OpenFOAM's boundary conditions.

        surfaceScalarField amaxSf
        (
            IOobject("amaxSf", runTime.timeName(), mesh),
            mesh,
            dimensionedScalar("amaxSf", dimVolume/dimTime, 0.0)
        );

        surfaceScalarField a_pos
        (
            IOobject("a_pos", runTime.timeName(), mesh),
            mesh,
            dimensionedScalar("a_pos", dimless, 0.5)
        );

        surfaceScalarField a_neg
        (
            IOobject("a_neg", runTime.timeName(), mesh),
            mesh,
            dimensionedScalar("a_neg", dimless, 0.5)
        );

        surfaceVectorField U_pos
        (
            IOobject("U_pos", runTime.timeName(), mesh),
            mesh,
            dimensionedVector("U_pos", dimVelocity, vector::zero)
        );

        surfaceVectorField U_neg
        (
            IOobject("U_neg", runTime.timeName(), mesh),
            mesh,
            dimensionedVector("U_neg", dimVelocity, vector::zero)
        );

        surfaceVectorField phiUp
        (
            IOobject("phiUp", runTime.timeName(), mesh),
            mesh,
            dimensionedVector
            (
                "phiUp",
                dimDensity*dimVelocity*dimVolume/dimTime,
                vector::zero
            )
        );

        surfaceScalarField phiEp
        (
            IOobject("phiEp", runTime.timeName(), mesh),
            mesh,
            dimensionedScalar("phiEp", dimEnergy/dimTime, 0.0)
        );

        // --- FUSED FLUX KERNEL ---
        // Replaces ~30 separate GPU kernel launches with one:
        //   - 12x fvc::interpolate (rho, rhoU, rPsi, e, c to faces)
        //   - ~18x surface field arithmetic (U, p, phiv, cSf, ap, am, etc.)
        // All computed in a single pass over internal faces.
        //
        // A4: when inviscid AND skipViscStores, use the variant that omits the
        // a_pos/a_neg/U_pos/U_neg per-face stores (dead when there is no viscous
        // flux). Same math otherwise ⇒ bit-identical result, fewer global writes.
        // `skipViscStores false` forces the full path for an apples-to-apples A/B.
        if (inviscid && skipViscStores)
        {
            launchFusedFluxKernelInviscid
            (
                rho, rhoU, e, psi, c, mesh,
                phi, phiUp, phiEp, amaxSf,
                isTadmor
            );
        }
        else
        {
            launchFusedFluxKernel
            (
                rho, rhoU, e, psi, c, mesh,
                phi, phiUp, phiEp, amaxSf,
                a_pos, a_neg, U_pos, U_neg,
                isTadmor
            );
        }

        // --- BOUNDARY FLUX FIX (wopr-cuda) -----------------------------------
        // The fused kernels above fill only INTERNAL faces of phi/phiUp/phiEp;
        // their boundary faces were left at zero, so fvc::div() at boundary
        // cells omitted the physical boundary flux (esp. p*Sf in momentum),
        // producing spurious force at boundary cells (a uniform field would not
        // stay at rest). The KT scheme reduces to the physical face flux at
        // boundaries (pos==neg), so set them to that -- matches CPU rhoCentral.
        forAll(mesh.boundary(), patchi)
        {
            phi.boundaryField()[patchi] =
                rho.boundaryField()[patchi]
               *(U.boundaryField()[patchi] & mesh.Sf().boundaryField()[patchi]);

            phiUp.boundaryField()[patchi] =
                rhoU.boundaryField()[patchi]
               *(U.boundaryField()[patchi] & mesh.Sf().boundaryField()[patchi])
              + p.boundaryField()[patchi]*mesh.Sf().boundaryField()[patchi];

            phiEp.boundaryField()[patchi] =
                (
                    rho.boundaryField()[patchi]
                   *(
                        e.boundaryField()[patchi]
                      + 0.5*(U.boundaryField()[patchi] & U.boundaryField()[patchi])
                    )
                  + p.boundaryField()[patchi]
                )
               *(U.boundaryField()[patchi] & mesh.Sf().boundaryField()[patchi]);
        }
        // ---------------------------------------------------------------------

        #include "compressibleCourantNo.H"
        #include "readTimeControls.H"
        #include "setDeltaT.H"

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        volScalarField muEff(turbulence->muEff());
        volTensorField tauMC("tauMC", muEff*dev2(Foam::T(fvc::grad(U))));

        // --- Solve density
        solve(fvm::ddt(rho) + fvc::div(phi));

        // positivity floor (wopr-cuda): clamp rho>0 BEFORE U=rhoU/rho, else a
        // near-vacuum cell drives U to NaN (observed at t~8.5us on the fine mesh).
        rho = max(rho, rhoMin);

        // --- Solve momentum
        solve(fvm::ddt(rhoU) + fvc::div(phiUp));

        U.dimensionedInternalField() =
            rhoU.dimensionedInternalField()
           /rho.dimensionedInternalField();
        U.correctBoundaryConditions();
        rhoU.boundaryField() = rho.boundaryField()*U.boundaryField();

        volScalarField rhoBydt(rho/runTime.deltaT());

        if (!inviscid)
        {
            solve
            (
                fvm::ddt(rho, U) - fvc::ddt(rho, U)
              - fvm::laplacian(muEff, U)
              - fvc::div(tauMC)
            );
            rhoU = rho*U;
        }

        // --- Solve energy (fused viscous flux)
        surfaceVectorField snGradU("snGradU", fvc::snGrad(U));

        surfaceScalarField sigmaDotU
        (
            IOobject("sigmaDotU", runTime.timeName(), mesh),
            mesh,
            dimensionedScalar("sigmaDotU", dimEnergy/dimTime, 0.0)
        );

        launchFusedViscFluxKernel
        (
            muEff, tauMC, snGradU, mesh,
            a_pos, a_neg, U_pos, U_neg,
            sigmaDotU
        );

        solve
        (
            fvm::ddt(rhoE)
          + fvc::div(phiEp)
          - fvc::div(sigmaDotU)
        );

        // --- Fused e and p: one kernel (replaces ~8 Thrust launches)
        launchFusedEandP(rho, U, rhoE, psi, e, p);
        // positivity floor (wopr-cuda): clamp internal energy > 0 so T(e) and
        // the sound speed c = sqrt(gamma p/rho) stay real next iteration.
        e = max(e, eMin);
        e.correctBoundaryConditions();
        thermo.correct();
        rhoE.boundaryField() =
            rho.boundaryField()*
            (
                e.boundaryField() + 0.5*magSqr(U.boundaryField())
            );

        if (!inviscid)
        {
            solve
            (
                fvm::ddt(rho, e) - fvc::ddt(rho, e)
              - fvm::laplacian(turbulence->alphaEff(), e)
            );
            thermo.correct();
            // --- Fused rhoE update (replaces ~4 Thrust launches)
            launchFusedRhoEUpdate(rho, e, U, rhoE);
        }

        // p internal field already set by fusedPostSolve
        p.correctBoundaryConditions();
        rho.boundaryField() = psi.boundaryField()*p.boundaryField();

        turbulence->correct();

        // --- Immersed-boundary bullet: force the moving-solid state ---------
        #include "enforceMovingBullet.H"

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
