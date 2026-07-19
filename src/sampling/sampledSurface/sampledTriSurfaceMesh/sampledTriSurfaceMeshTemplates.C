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

\*---------------------------------------------------------------------------*/

#include "sampledTriSurfaceMesh.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class Type>
Foam::tmp<Foam::Field<Type> >
Foam::sampledTriSurfaceMesh::sampleField
(
    const GeometricField<Type, fvPatchField, volMesh>& vField
) const
{
    // One value per face
    tmp<Field<Type> > tvalues(new Field<Type>(sampleElements_.size()));
    Field<Type>& values = tvalues();

    if (sampleSource_ == cells || sampleSource_ == insideCells)
    {
        // Sample cells (RapidCFD GPU-safety: the internal field is in device memory;
        // vField[cellI] would dereference device memory on the host and segfault.
        // Bulk-copy to the host once via gpuField::asField(), then index on the host.)
        const Field<Type> cellVals(vField.getField().asField());

        forAll(sampleElements_, triI)
        {
            values[triI] = cellVals[sampleElements_[triI]];
        }
    }
    else
    {
        // Sample boundary faces

        const polyBoundaryMesh& pbm = mesh().boundaryMesh();
        label nBnd = mesh().nFaces()-mesh().nInternalFaces();

        // Create flat boundary field

        Field<Type> bVals(nBnd, pTraits<Type>::zero);

        forAll(vField.boundaryField(), patchI)
        {
            label bFaceI = pbm[patchI].start() - mesh().nInternalFaces();

            // RapidCFD GPU-safety: fvPatchField is a device gpuField; copy its
            // values to the host (asField()) before assigning into the host bVals.
            const Field<Type> patchVals
            (
                vField.boundaryField()[patchI].asField()
            );

            SubList<Type>
            (
                bVals,
                patchVals.size(),
                bFaceI
            ).assign(patchVals);
        }

        // Sample in flat boundary field

        forAll(sampleElements_, triI)
        {
            label faceI = sampleElements_[triI];
            values[triI] = bVals[faceI-mesh().nInternalFaces()];
        }
    }

    return tvalues;
}


template<class Type>
Foam::tmp<Foam::Field<Type> >
Foam::sampledTriSurfaceMesh::interpolateField
(
    const interpolation<Type>& interpolator
) const
{
    // One value per vertex
    tmp<Field<Type> > tvalues(new Field<Type>(sampleElements_.size()));
    Field<Type>& values = tvalues();

    if (sampleSource_ == cells || sampleSource_ == insideCells)
    {
        // Sample cells.

        forAll(sampleElements_, pointI)
        {
            values[pointI] = interpolator.interpolate
            (
                samplePoints_[pointI],
                sampleElements_[pointI]
            );
        }
    }
    else
    {
        // Sample boundary faces.

        forAll(samplePoints_, pointI)
        {
            label faceI = sampleElements_[pointI];

            values[pointI] = interpolator.interpolate
            (
                samplePoints_[pointI],
                mesh().faceOwner()[faceI],
                faceI
            );
        }
    }

    return tvalues;
}


// ************************************************************************* //
