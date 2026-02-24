/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2013-2014 OpenFOAM Foundation
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

#include "faceAreaWeightAMI.H"

// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

template<class SourcePatch, class TargetPatch>
void Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::calcAddressing
(
    List<DynamicList<label> >& srcAddr,
    List<DynamicList<scalar> >& srcWght,
    List<DynamicList<label> >& tgtAddr,
    List<DynamicList<scalar> >& tgtWght,
    label srcFaceI,
    label tgtFaceI
)
{
    // construct weights and addressing
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    label nFacesRemaining = srcAddr.size();

    // list of tgt face neighbour faces
    DynamicList<label> nbrFaces(10);

    // list of faces currently visited for srcFaceI to avoid multiple hits
    DynamicList<label> visitedFaces(10);

    // list to keep track of tgt faces used to seed src faces
    labelList seedFaces(nFacesRemaining, -1);
    seedFaces[srcFaceI] = tgtFaceI;

    // list to keep track of whether src face can be mapped
    boolList mapFlag(nFacesRemaining, true);

    // reset starting seed
    label startSeedI = 0;

    // Should all faces be matched?
    const bool mustMatch = this->requireMatch_;

    bool continueWalk = true;
    DynamicList<label> nonOverlapFaces;
    do
    {
        nbrFaces.clear();
        visitedFaces.clear();

        // Do advancing front starting from srcFaceI,tgtFaceI
        bool faceProcessed = processSourceFace
        (
            srcFaceI,
            tgtFaceI,

            nbrFaces,
            visitedFaces,

            srcAddr,
            srcWght,
            tgtAddr,
            tgtWght
        );

        mapFlag[srcFaceI] = false;

        if (!faceProcessed)
        {
            nonOverlapFaces.append(srcFaceI);
        }

        // choose new src face from current src face neighbour
        continueWalk = setNextFaces
        (
            startSeedI,
            srcFaceI,
            tgtFaceI,
            mapFlag,
            seedFaces,
            visitedFaces,
            mustMatch
        );
    } while (continueWalk);

    this->srcNonOverlap_.transfer(nonOverlapFaces);
}


template<class SourcePatch, class TargetPatch>
bool Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::processSourceFace
(
    const label srcFaceI,
    const label tgtStartFaceI,

    // list of tgt face neighbour faces
    DynamicList<label>& nbrFaces,
    // list of faces currently visited for srcFaceI to avoid multiple hits
    DynamicList<label>& visitedFaces,

    // temporary storage for addressing and weights
    List<DynamicList<label> >& srcAddr,
    List<DynamicList<scalar> >& srcWght,
    List<DynamicList<label> >& tgtAddr,
    List<DynamicList<scalar> >& tgtWght
)
{
    if (tgtStartFaceI == -1)
    {
        return false;
    }

    nbrFaces.clear();
    visitedFaces.clear();

    // append initial target face and neighbours
    nbrFaces.append(tgtStartFaceI);
    this->appendNbrFaces
    (
        tgtStartFaceI,
        this->tgtPatch_,
        visitedFaces,
        nbrFaces
    );

    bool faceProcessed = false;

    while (nbrFaces.size() > 0)
    {
        // process new target face as LIFO
        label tgtFaceI = nbrFaces.remove();
        visitedFaces.append(tgtFaceI);
        scalar area = interArea(srcFaceI, tgtFaceI);

        // store when intersection fractional area > tolerance
        if (area/this->srcMagSf_[srcFaceI] > faceAreaIntersect::tolerance())
        {
            srcAddr[srcFaceI].append(tgtFaceI);
            srcWght[srcFaceI].append(area);

            tgtAddr[tgtFaceI].append(srcFaceI);
            tgtWght[tgtFaceI].append(area);

            this->appendNbrFaces
            (
                tgtFaceI,
                this->tgtPatch_,
                visitedFaces,
                nbrFaces
            );

            faceProcessed = true;
        }
    }

    return faceProcessed;
}


template<class SourcePatch, class TargetPatch>
bool Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::setNextFaces
(
    label& startSeedI,
    label& srcFaceI,
    label& tgtFaceI,
    const boolList& mapFlag,
    labelList& seedFaces,
    const DynamicList<label>& visitedFaces,
    bool errorOnNotFound
) const
{
    // Count remaining faces
    label nRemaining = 0;
    for (label i = 0; i < mapFlag.size(); i++)
    {
        if (mapFlag[i]) nRemaining++;
    }
    if (nRemaining == 0)
    {
        return false;
    }

    const labelList& srcNbrFaces = this->srcPatch_.faceFaces()[srcFaceI];

    // initialise tgtFaceI
    tgtFaceI = -1;

    // set possible seeds for later use by quick overlap test
    bool valuesSet = false;
    forAll(srcNbrFaces, i)
    {
        label faceS = srcNbrFaces[i];

        if (mapFlag[faceS] && seedFaces[faceS] == -1)
        {
            forAll(visitedFaces, j)
            {
                label faceT = visitedFaces[j];
                const scalar threshold =
                    this->srcMagSf_[faceS]
                  * faceAreaIntersect::tolerance();

                // Fast boolean overlap check instead of full area computation
                if (overlaps(faceS, faceT, threshold))
                {
                    seedFaces[faceS] = faceT;

                    if (!valuesSet)
                    {
                        srcFaceI = faceS;
                        tgtFaceI = faceT;
                        valuesSet = true;
                    }
                }
            }
        }
    }

    if (valuesSet)
    {
        return true;
    }

    // set next src and tgt faces if not set above
    // try to use existing seed
    bool foundNextSeed = false;
    for (label faceI = startSeedI; faceI < mapFlag.size(); faceI++)
    {
        if (mapFlag[faceI])
        {
            if (!foundNextSeed)
            {
                startSeedI = faceI;
                foundNextSeed = true;
            }

            if (seedFaces[faceI] != -1)
            {
                srcFaceI = faceI;
                tgtFaceI = seedFaces[faceI];

                return true;
            }
        }
    }

    // perform new search to find match
    if (debug)
    {
        Pout<< "Advancing front stalled: searching for new "
            << "target face" << endl;
    }

    foundNextSeed = false;
    for (label faceI = startSeedI; faceI < mapFlag.size(); faceI++)
    {
        if (mapFlag[faceI])
        {
            if (!foundNextSeed)
            {
                startSeedI = faceI + 1;
                foundNextSeed = true;
            }

            srcFaceI = faceI;
            tgtFaceI = this->findTargetFace(srcFaceI);

            if (tgtFaceI >= 0)
            {
                return true;
            }
        }
    }

    if (errorOnNotFound)
    {
        FatalErrorIn
        (
            "bool Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::"
            "setNextFaces"
            "("
                "label&, "
                "label&, "
                "label&, "
                "const boolList&, "
                "labelList&, "
                "const DynamicList<label>&, "
                "bool"
            ") const"
        )  << "Unable to set source and target faces" << abort(FatalError);
    }

    return false;
}


template<class SourcePatch, class TargetPatch>
bool Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::isCandidate
(
    const label srcFaceI,
    const label tgtFaceI
) const
{
    // Quick reject if either face has zero area
    if
    (
        (this->srcMagSf_[srcFaceI] < ROOTVSMALL)
     || (this->tgtMagSf_[tgtFaceI] < ROOTVSMALL)
    )
    {
        return false;
    }

    return true;
}


template<class SourcePatch, class TargetPatch>
Foam::scalar Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::interArea
(
    const label srcFaceI,
    const label tgtFaceI
) const
{
    scalar area = 0;

    // Quick reject if not a candidate
    if (!isCandidate(srcFaceI, tgtFaceI))
    {
        return area;
    }

    const pointField& srcPoints = this->srcPatch_.points();
    const pointField& tgtPoints = this->tgtPatch_.points();

    // references to candidate faces
    const face& src = this->srcPatch_[srcFaceI];
    const face& tgt = this->tgtPatch_[tgtFaceI];

    // create intersection object
    faceAreaIntersect inter(srcPoints, tgtPoints, this->reverseTarget_);

    // crude resultant norm
    vector n(-this->srcPatch_.faceNormals()[srcFaceI]);
    if (this->reverseTarget_)
    {
        n -= this->tgtPatch_.faceNormals()[tgtFaceI];
    }
    else
    {
        n += this->tgtPatch_.faceNormals()[tgtFaceI];
    }
    scalar magN = mag(n);

    if (magN > ROOTVSMALL)
    {
        area = inter.calc(src, tgt, n/magN, this->triMode_);
    }
    else
    {
        WarningIn
        (
            "scalar Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::"
            "interArea"
            "("
                "const label, "
                "const label"
            ") const"
        )   << "Invalid normal for source face " << srcFaceI
            << " points " << UIndirectList<point>(srcPoints, src)
            << " target face " << tgtFaceI
            << " points " << UIndirectList<point>(tgtPoints, tgt)
            << endl;
    }


    if ((debug > 1) && (area > 0))
    {
        this->writeIntersectionOBJ(area, src, tgt, srcPoints, tgtPoints);
    }

    return area;
}


template<class SourcePatch, class TargetPatch>
bool Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::overlaps
(
    const label srcFaceI,
    const label tgtFaceI,
    const scalar threshold
) const
{
    // Quick reject if not a candidate
    if (!isCandidate(srcFaceI, tgtFaceI))
    {
        return false;
    }

    const pointField& srcPoints = this->srcPatch_.points();
    const pointField& tgtPoints = this->tgtPatch_.points();

    const face& src = this->srcPatch_[srcFaceI];
    const face& tgt = this->tgtPatch_[tgtFaceI];

    // create intersection object
    faceAreaIntersect inter(srcPoints, tgtPoints, this->reverseTarget_);

    // crude resultant norm
    vector n(-this->srcPatch_.faceNormals()[srcFaceI]);
    if (this->reverseTarget_)
    {
        n -= this->tgtPatch_.faceNormals()[tgtFaceI];
    }
    else
    {
        n += this->tgtPatch_.faceNormals()[tgtFaceI];
    }
    scalar magN = mag(n);

    if (magN > ROOTVSMALL)
    {
        // Use interArea and compare to threshold
        scalar area = inter.calc(src, tgt, n/magN, this->triMode_);
        return area > threshold;
    }

    return false;
}


template<class SourcePatch, class TargetPatch>
void Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::
restartUncoveredSourceFace
(
    List<DynamicList<label> >& srcAddr,
    List<DynamicList<scalar> >& srcWght,
    List<DynamicList<label> >& tgtAddr,
    List<DynamicList<scalar> >& tgtWght
)
{
    // Improved restart: raise threshold from 0.5 to 0.95 (back-ported from
    // OpenFOAM v2412) and use per-point octree search to find missed overlaps

    label nBelowMinWeight = 0;
    const scalar minWeight = 0.95;

    // list of tgt face neighbour faces
    DynamicList<label> nbrFaces(10);

    // list of faces currently visited for srcFaceI to avoid multiple hits
    DynamicList<label> visitedFaces(10);

    forAll(srcWght, srcFaceI)
    {
        const scalar s = sum(srcWght[srcFaceI]);
        const scalar t = s/this->srcMagSf_[srcFaceI];

        if (t < minWeight)
        {
            ++nBelowMinWeight;

            const face& f = this->srcPatch_[srcFaceI];

            // Try octree search from each vertex of the source face
            // (v2412 improvement: catches overlaps missed by centroid-only search)
            forAll(f, fpi)
            {
                const pointField& srcPts = this->srcPatch_.points();
                const point& srcPt = srcPts[f[fpi]];
                const scalar srcFaceArea = this->srcMagSf_[srcFaceI];

                // Search for nearest target face from this vertex
                pointIndexHit sample =
                    this->treePtr_->findNearest
                    (
                        srcPt,
                        10.0*srcFaceArea
                    );

                if (sample.hit())
                {
                    label tgtFaceI = sample.index();

                    // Check it's not already in our addressing
                    bool alreadyFound = false;
                    forAll(srcAddr[srcFaceI], k)
                    {
                        if (srcAddr[srcFaceI][k] == tgtFaceI)
                        {
                            alreadyFound = true;
                            break;
                        }
                    }

                    if (!alreadyFound)
                    {
                        nbrFaces.clear();
                        // Seed visitedFaces with existing addressing
                        // to avoid re-visiting
                        visitedFaces.clear();
                        forAll(srcAddr[srcFaceI], k)
                        {
                            visitedFaces.append(srcAddr[srcFaceI][k]);
                        }

                        processSourceFace
                        (
                            srcFaceI,
                            tgtFaceI,

                            nbrFaces,
                            visitedFaces,

                            srcAddr,
                            srcWght,
                            tgtAddr,
                            tgtWght
                        );
                    }
                }
            }
        }
    }

    if (debug && nBelowMinWeight)
    {
        Pout<< "faceAreaWeightAMI: restarted search on "
            << nBelowMinWeight
            << " faces since sum of weights < " << minWeight
            << endl;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class SourcePatch, class TargetPatch>
Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::faceAreaWeightAMI
(
    const SourcePatch& srcPatch,
    const TargetPatch& tgtPatch,
    const scalarField& srcMagSf,
    const scalarField& tgtMagSf,
    const faceAreaIntersect::triangulationMode& triMode,
    const bool reverseTarget,
    const bool requireMatch,
    const bool restartUncoveredSourceFace
)
:
    AMIMethod<SourcePatch, TargetPatch>
    (
        srcPatch,
        tgtPatch,
        srcMagSf,
        tgtMagSf,
        triMode,
        reverseTarget,
        requireMatch
    ),
    restartUncoveredSourceFace_(restartUncoveredSourceFace)
{}


// * * * * * * * * * * * * * * * * Destructor * * * * * * * * * * * * * * * //

template<class SourcePatch, class TargetPatch>
Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::~faceAreaWeightAMI()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class SourcePatch, class TargetPatch>
void Foam::faceAreaWeightAMI<SourcePatch, TargetPatch>::calculate
(
    labelListList& srcAddress,
    scalarListList& srcWeights,
    labelListList& tgtAddress,
    scalarListList& tgtWeights,
    label srcFaceI,
    label tgtFaceI
)
{
    bool ok =
        this->initialise
        (
            srcAddress,
            srcWeights,
            tgtAddress,
            tgtWeights,
            srcFaceI,
            tgtFaceI
        );

    if (!ok)
    {
        return;
    }

    // temporary storage for addressing and weights
    List<DynamicList<label> > srcAddr(this->srcPatch_.size());
    List<DynamicList<scalar> > srcWght(srcAddr.size());
    List<DynamicList<label> > tgtAddr(this->tgtPatch_.size());
    List<DynamicList<scalar> > tgtWght(tgtAddr.size());

    calcAddressing
    (
        srcAddr,
        srcWght,
        tgtAddr,
        tgtWght,
        srcFaceI,
        tgtFaceI
    );

    if (debug && !this->srcNonOverlap_.empty())
    {
        Pout<< "    AMI: " << this->srcNonOverlap_.size()
            << " non-overlap faces identified"
            << endl;
    }


    // Check for badly covered faces
    if (restartUncoveredSourceFace_)
    {
        restartUncoveredSourceFace
        (
            srcAddr,
            srcWght,
            tgtAddr,
            tgtWght
        );
    }


    // transfer data to persistent storage
    forAll(srcAddr, i)
    {
        srcAddress[i].transfer(srcAddr[i]);
        srcWeights[i].transfer(srcWght[i]);
    }
    forAll(tgtAddr, i)
    {
        tgtAddress[i].transfer(tgtAddr[i]);
        tgtWeights[i].transfer(tgtWght[i]);
    }
}


// ************************************************************************* //
