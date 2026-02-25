/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2013 OpenFOAM Foundation
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
    rhoCentralDyMFoam

Description
    Density-based compressible flow solver based on central-upwind schemes of
    Kurganov and Tadmor.

    Hybrid approach: fused GPU kernel for internal faces (fast) +
    standard boundary computation for AMI/coupled patches (correct).

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "psiThermo.H"
#include "turbulenceModel.H"
#include "zeroGradientFvPatchFields.H"
#include "fixedRhoFvPatchScalarField.H"
#include "dynamicFvMesh.H"
#include "fusedFlux.H"
#include "fusedViscFlux.H"
#include "fusedPostSolve.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"

    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createFields.H"
    #include "readTimeControls.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    #include "readFluxScheme.H"

    dimensionedScalar v_zero("v_zero", dimVolume/dimTime, 0.0);

    bool isTadmor = (fluxScheme == "Tadmor");

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        // --- Compute speed of sound
        volScalarField rPsi("rPsi", 1.0/psi);
        volScalarField c("c", sqrt(thermo.Cp()/thermo.Cv()*rPsi));

        // --- Allocate output surface fields ---
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

        // ============================================================
        // STEP 1: Fused kernel for internal faces (FAST — 1 kernel)
        // ============================================================
#include <cuda_runtime.h>
        cudaDeviceSynchronize();
        double t0 = runTime.elapsedClockTime();

        launchFusedFluxKernel
        (
            rho, rhoU, e, psi, c, mesh,
            phi, phiUp, phiEp, amaxSf,
            a_pos, a_neg, U_pos, U_neg,
            isTadmor
        );

        cudaDeviceSynchronize();
        double t1 = runTime.elapsedClockTime();

        // ============================================================
        // STEP 2: Boundary correction for coupled (AMI) patches
        // Standard interpolation gives correct boundary face values.
        // We only copy the boundary portions to the fused output fields.
        // ============================================================
        {
            surfaceScalarField rho_pos
            (
                fvc::interpolate(rho, pos, "reconstruct(rho)")
            );
            surfaceScalarField rho_neg
            (
                fvc::interpolate(rho, neg, "reconstruct(rho)")
            );

            surfaceVectorField rhoU_pos
            (
                fvc::interpolate(rhoU, pos, "reconstruct(U)")
            );
            surfaceVectorField rhoU_neg
            (
                fvc::interpolate(rhoU, neg, "reconstruct(U)")
            );

            surfaceScalarField rPsi_pos
            (
                fvc::interpolate(rPsi, pos, "reconstruct(T)")
            );
            surfaceScalarField rPsi_neg
            (
                fvc::interpolate(rPsi, neg, "reconstruct(T)")
            );

            surfaceScalarField e_pos_f
            (
                fvc::interpolate(e, pos, "reconstruct(T)")
            );
            surfaceScalarField e_neg_f
            (
                fvc::interpolate(e, neg, "reconstruct(T)")
            );

            surfaceVectorField U_pos_s("U_pos_s", rhoU_pos/rho_pos);
            surfaceVectorField U_neg_s("U_neg_s", rhoU_neg/rho_neg);

            surfaceScalarField p_pos("p_pos", rho_pos*rPsi_pos);
            surfaceScalarField p_neg("p_neg", rho_neg*rPsi_neg);

            surfaceScalarField phiv_pos("phiv_pos", U_pos_s & mesh.Sf());
            surfaceScalarField phiv_neg("phiv_neg", U_neg_s & mesh.Sf());

            surfaceScalarField cSf_pos
            (
                "cSf_pos",
                fvc::interpolate(c, pos, "reconstruct(T)")*mesh.magSf()
            );
            surfaceScalarField cSf_neg
            (
                "cSf_neg",
                fvc::interpolate(c, neg, "reconstruct(T)")*mesh.magSf()
            );

            surfaceScalarField ap_s
            (
                max(max(phiv_pos + cSf_pos, phiv_neg + cSf_neg), v_zero)
            );
            surfaceScalarField am_s
            (
                min(min(phiv_pos - cSf_pos, phiv_neg - cSf_neg), v_zero)
            );

            surfaceScalarField a_pos_s("a_pos_s", ap_s/(ap_s - am_s));
            surfaceScalarField amaxSf_s("amaxSf_s", max(mag(am_s), mag(ap_s)));
            surfaceScalarField aSf_s("aSf_s", am_s*a_pos_s);

            if (fluxScheme == "Tadmor")
            {
                aSf_s = -0.5*amaxSf_s;
                a_pos_s = 0.5;
            }

            surfaceScalarField a_neg_s("a_neg_s", 1.0 - a_pos_s);

            phiv_pos *= a_pos_s;
            phiv_neg *= a_neg_s;

            surfaceScalarField aphiv_pos("aphiv_pos", phiv_pos - aSf_s);
            surfaceScalarField aphiv_neg("aphiv_neg", phiv_neg + aSf_s);

            amaxSf_s = max(mag(aphiv_pos), mag(aphiv_neg));

            surfaceScalarField phi_s
            (
                "phi_s",
                aphiv_pos*rho_pos + aphiv_neg*rho_neg
            );
            surfaceVectorField phiUp_s
            (
                "phiUp_s",
                (aphiv_pos*rhoU_pos + aphiv_neg*rhoU_neg)
              + (a_pos_s*p_pos + a_neg_s*p_neg)*mesh.Sf()
            );
            surfaceScalarField phiEp_s
            (
                "phiEp_s",
                aphiv_pos*(rho_pos*(e_pos_f + 0.5*magSqr(U_pos_s)) + p_pos)
              + aphiv_neg*(rho_neg*(e_neg_f + 0.5*magSqr(U_neg_s)) + p_neg)
              + aSf_s*p_pos - aSf_s*p_neg
            );

            // Copy boundary values from standard to fused output
            forAll(mesh.boundary(), patchI)
            {
                phi.boundaryField()[patchI] =
                    phi_s.boundaryField()[patchI];
                phiUp.boundaryField()[patchI] =
                    phiUp_s.boundaryField()[patchI];
                phiEp.boundaryField()[patchI] =
                    phiEp_s.boundaryField()[patchI];
                amaxSf.boundaryField()[patchI] =
                    amaxSf_s.boundaryField()[patchI];
                a_pos.boundaryField()[patchI] =
                    a_pos_s.boundaryField()[patchI];
                a_neg.boundaryField()[patchI] =
                    a_neg_s.boundaryField()[patchI];
                U_pos.boundaryField()[patchI] =
                    U_pos_s.boundaryField()[patchI];
                U_neg.boundaryField()[patchI] =
                    U_neg_s.boundaryField()[patchI];
            }
        }

        cudaDeviceSynchronize();
        double t1b = runTime.elapsedClockTime();

        #include "compressibleCourantNo.H"
        #include "readTimeControls.H"
        #include "setDeltaT.H"

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // ============================================================
        // STEP 3: Dynamic mesh update
        // ============================================================
        mesh.update();

        // Make mass flux relative to mesh motion:
        // phi_relative = phi_absolute - meshPhi * rho_face
        phi -= mesh.phi() * fvc::interpolate(rho);

        // Energy flux mesh motion correction: meshPhi * p_face
        phiEp += mesh.phi() * fvc::interpolate(p);

        cudaDeviceSynchronize();
        double t2 = runTime.elapsedClockTime();

        volScalarField muEff(turbulence->muEff());
        volTensorField tauMC("tauMC", muEff*dev2(Foam::T(fvc::grad(U))));

        // --- Solve density
        solve(fvm::ddt(rho) + fvc::div(phi));

        // --- Solve momentum
        solve(fvm::ddt(rhoU) + fvc::div(phiUp));

        U.dimensionedInternalField() =
            rhoU.dimensionedInternalField()
           /rho.dimensionedInternalField();
        U.correctBoundaryConditions();
        rhoU.boundaryField() = rho.boundaryField()*U.boundaryField();

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

        cudaDeviceSynchronize();
        double t3 = runTime.elapsedClockTime();

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

        cudaDeviceSynchronize();
        double t4 = runTime.elapsedClockTime();

        // --- Fused e and p
        launchFusedEandP(rho, U, rhoE, psi, e, p);
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
            launchFusedRhoEUpdate(rho, e, U, rhoE);
        }

        p.correctBoundaryConditions();
        rho.boundaryField() = psi.boundaryField()*p.boundaryField();

        turbulence->correct();

        cudaDeviceSynchronize();
        double t5 = runTime.elapsedClockTime();

        if (runTime.value() > 0)
        {
            Info<< "TIMING: FusedFlux=" << (t1 - t0)
                << "s, BndFix=" << (t1b - t1)
                << "s, Mesh=" << (t2 - t1b)
                << "s, RhoU=" << (t3 - t2)
                << "s, RhoE=" << (t4 - t3)
                << "s, Post=" << (t5 - t4) << "s" << nl;
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
