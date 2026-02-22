/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Description
    Fast GPU-only mesh update for solidBodyMotion with linearMotion.

    For pure translation (linearMotion), the mesh volumes (V) and face area
    vectors (Sf) are invariant. Only the mesh motion flux (meshPhi) changes.
    This header computes meshPhi analytically on GPU without calling
    fvMesh::movePoints(), avoiding the expensive CPU-side mesh geometry
    recalculation that dominates solver runtime.

    meshPhi[face] = (meshVelocity & Sf[face])

    where meshVelocity is the solid body velocity from dynamicMeshDict.
\*---------------------------------------------------------------------------*/

#ifndef fastMeshUpdate_H
#define fastMeshUpdate_H

#include "dynamicFvMesh.H"
#include "surfaceFields.H"

namespace Foam
{

// Compute mesh motion flux analytically for linear motion
// meshPhi = (velocity . Sf) for each face — purely on GPU
inline void computeAnalyticalMeshPhi
(
    const dynamicFvMesh& mesh,
    surfaceScalarField& meshPhiField,
    const vector& meshVelocity
)
{
    // Internal faces: meshPhi = velocity & Sf
    const surfaceVectorField& Sf = mesh.Sf();

    // Use GPU field operations — Sf is already on GPU
    meshPhiField.internalField() =
        (Sf.internalField() & meshVelocity);

    // Boundary faces
    forAll(mesh.boundary(), patchI)
    {
        meshPhiField.boundaryField()[patchI] =
            (Sf.boundaryField()[patchI] & meshVelocity);
    }
}

} // End namespace Foam

#endif

// ************************************************************************* //
