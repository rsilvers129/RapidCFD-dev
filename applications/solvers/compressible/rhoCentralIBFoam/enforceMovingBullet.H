/*---------------------------------------------------------------------------*\
  Immersed-boundary bullet — per-step enforcement (wopr-cuda, Phase 1).
  Included near the END of the time loop, after rho/U/rhoU/rhoE/e/p are updated
  and before runTime.write(), so the written fields and the NEXT step's flux
  reconstruction both see the forced bullet state.
\*---------------------------------------------------------------------------*/

if (bulletActive)
{
    // Prescribed motion: nose leads in the travel direction. With vBullet < 0
    // (downrange = -z), the nose position decreases with time; the body extends
    // +z behind the nose to the tail.
    const scalar tNow  = runTime.value();
    const scalar zNose = bulletZ0 + bulletV*tNow;

    launchEnforceBullet
    (
        cellC, rho, U, rhoU, rhoE, e, solidMask,
        zNose, bulletLength, bulletRadiiPtr, bulletNBins,
        bulletV, rhoMin.value(), bulletESolid
    );

    // Re-sync boundary conditions and thermodynamics with the forced interior.
    U.correctBoundaryConditions();
    rhoU.boundaryField() = rho.boundaryField()*U.boundaryField();
    e.correctBoundaryConditions();
    thermo.correct();                       // T, psi, mu from the forced e
    // keep p consistent with the forced (rho, e): p = rho*R*T = rho/psi
    p.dimensionedInternalField() =
        rho.dimensionedInternalField()/psi.dimensionedInternalField();
    p.correctBoundaryConditions();
    rhoE.boundaryField() =
        rho.boundaryField()*(e.boundaryField() + 0.5*magSqr(U.boundaryField()));

    if (runTime.outputTime())
    {
        Info<< "    bullet: nose z=" << zNose << " tail z=" << (zNose + bulletLength)
            << "  solid cells=" << gSum(solidMask.internalField()) << endl;
    }
}

// ************************************************************************* //
