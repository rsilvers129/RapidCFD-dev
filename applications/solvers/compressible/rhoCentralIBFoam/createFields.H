Info<< "Reading thermophysical properties\n" << endl;

autoPtr<psiThermo> pThermo
(
    psiThermo::New(mesh)
);
psiThermo& thermo = pThermo();

volScalarField& p = thermo.p();
volScalarField& e = thermo.he();
const volScalarField& T = thermo.T();
const volScalarField& psi = thermo.psi();
const volScalarField& mu = thermo.mu();

bool inviscid(true);
if (max(mu.internalField()) > 0.0)
{
    inviscid = false;
}

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "rhoBoundaryTypes.H"
volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    thermo.rho(),
    rhoBoundaryTypes
);

volVectorField rhoU
(
    IOobject
    (
        "rhoU",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    rho*U
);

volScalarField rhoE
(
    IOobject
    (
        "rhoE",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    rho*(e + 0.5*magSqr(U))
);

surfaceScalarField pos
(
    IOobject
    (
        "pos",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar("pos", dimless, 1.0)
);

surfaceScalarField neg
(
    IOobject
    (
        "neg",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar("neg", dimless, -1.0)
);


surfaceScalarField phi("phi", mesh.Sf() & fvc::interpolate(rhoU));

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::turbulenceModel> turbulence
(
    compressible::turbulenceModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

// --- Positivity floors (wopr-cuda) -----------------------------------------
// rhoCentralFoam has no native floor. Strong expansions (muzzle blast, the
// exit near-vacuum) can drive rho or e to <= 0, after which U = rhoU/rho and
// T(e) blow up to NaN (rhoCentral does not trap FPE, so it runs on producing
// NaN). These floors clamp density and internal energy each step, mirroring
// Sean's rhoMin lever. Tunable in controlDict (rhoMin [kg/m3], eMin [J/kg]);
// defaults are far below any physical blast state so they only bite artifacts.
dimensionedScalar rhoMin
(
    "rhoMin", dimDensity,
    runTime.controlDict().lookupOrDefault<scalar>("rhoMin", 1.0e-4)
);
dimensionedScalar eMin
(
    "eMin", dimEnergy/dimMass,
    runTime.controlDict().lookupOrDefault<scalar>("eMin", 1.0e4)
);

// A4: skip the viscous-only a_pos/a_neg/U_pos/U_neg per-face stores in the flux
// kernel when the run is inviscid. Default true (the optimization). Set false in
// controlDict to force the full-store path for an apples-to-apples A/B benchmark.
const bool skipViscStores =
    runTime.controlDict().lookupOrDefault<Switch>("skipViscStores", true);

// --- Immersed-boundary bullet (Phase 1, wopr-cuda) --------------------------
// Prescribed capsule translating along z (see movingBullet.H). Active only if
// constant/bulletProperties is present, so this solver reduces exactly to the
// static blast when the dict is absent.
const bool bulletActive =
    isFile(runTime.path()/runTime.constant()/"bulletProperties");

scalar bulletZ0     = 0.02;      // nose position at t=0 [m] (in the bore, +z)
scalar bulletV      = -890.0;    // z-velocity [m/s] (-z = downrange)
scalar bulletLength = 0.02336;   // bullet length [m]
scalar bulletRadius = 0.00278;   // bullet radius [m]  (5.56 mm / 2)
scalar bulletESolid = 2.1e5;     // solid-interior internal-energy floor [J/kg]

if (bulletActive)
{
    IOdictionary bulletDict
    (
        IOobject
        (
            "bulletProperties", runTime.constant(), mesh,
            IOobject::MUST_READ, IOobject::NO_WRITE
        )
    );
    bulletZ0     = bulletDict.lookupOrDefault<scalar>("z0",       bulletZ0);
    bulletV      = bulletDict.lookupOrDefault<scalar>("velocity", bulletV);
    bulletLength = bulletDict.lookupOrDefault<scalar>("length",   bulletLength);
    bulletRadius = bulletDict.lookupOrDefault<scalar>("radius",   bulletRadius);
    bulletESolid = bulletDict.lookupOrDefault<scalar>("eSolid",   bulletESolid);

    Info<< "Immersed-boundary bullet ACTIVE: z0=" << bulletZ0
        << " v=" << bulletV << " L=" << bulletLength
        << " R=" << bulletRadius << nl << endl;
}
else
{
    Info<< "No constant/bulletProperties -> static blast (no bullet)." << nl << endl;
}

// --- real ogive radius profile R(s) from constant/bulletProfile (STL-derived) ---
// s=0 nose .. s=L tail; captures the pointed nose (attached shock). Falls back to a
// cylinder at bulletRadius if the profile file is absent.
label bulletNBins = 2;
List<scalar> bulletRadiiHost(2, bulletRadius);
if (bulletActive && isFile(runTime.path()/runTime.constant()/"bulletProfile"))
{
    IOdictionary prof
    (
        IOobject("bulletProfile", runTime.constant(), mesh,
                 IOobject::MUST_READ, IOobject::NO_WRITE)
    );
    bulletNBins  = readLabel(prof.lookup("nBins"));
    bulletLength = readScalar(prof.lookup("length"));
    bulletRadiiHost = List<scalar>(prof.lookup("radii"));
    Info<< "Bullet ogive profile: " << bulletNBins << " bins, L=" << bulletLength
        << ", nose R=" << bulletRadiiHost[0]
        << ", body R=" << max(bulletRadiiHost) << nl << endl;
}
// upload profile to the GPU (small array; direct CUDA copy, persists for the run)
scalar* bulletRadiiPtr = nullptr;
{
    const label nB = bulletRadiiHost.size();
    cudaMalloc((void**)&bulletRadiiPtr, nB*sizeof(scalar));
    cudaMemcpy
    (
        bulletRadiiPtr, bulletRadiiHost.begin(),
        nB*sizeof(scalar), cudaMemcpyHostToDevice
    );
}

// cell centres (static mesh -> constant) reused by the bullet kernel each step
const volVectorField& cellC = mesh.C();

// 1 = solid (inside bullet) this step, 0 = fluid. Diagnostic + render aid.
volScalarField solidMask
(
    IOobject
    (
        "solidMask", runTime.timeName(), mesh,
        IOobject::NO_READ, IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("solidMask", dimless, 0.0)
);
