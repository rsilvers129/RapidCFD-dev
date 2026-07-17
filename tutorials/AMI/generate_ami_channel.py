#!/usr/bin/env python3
"""Generate non-conformal dual-block cyclicAMI channel mesh for RapidCFD."""
import os, sys
from collections import defaultdict

def generate(case_dir):
    mesh = os.path.join(case_dir, "constant", "polyMesh")
    os.makedirs(mesh, exist_ok=True)
    blocks = [
        ("L", 0.0, 0.5, 6, 8, 8),
        ("R", 0.5, 1.0, 6, 5, 5),
    ]
    Ly = Lz = 0.2

    def foam_header(cls, obj):
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "constant/polyMesh";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    all_points = []
    internal = []
    bnd = {k: [] for k in ["inlet", "outlet", "walls", "AMI1", "AMI2"]}
    cell_offset = 0
    point_offset = 0

    for bname, x0, x1, nx, ny, nz in blocks:
        def lpid(i, j, k):
            return i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

        def lcid(i, j, k):
            return i + j * nx + k * nx * ny

        for k in range(nz + 1):
            for j in range(ny + 1):
                for i in range(nx + 1):
                    all_points.append(
                        (x0 + (x1 - x0) * i / nx, Ly * j / ny, Lz * k / nz)
                    )

        face_map = defaultdict(list)

        def add_cell(i, j, k):
            c = cell_offset + lcid(i, j, k)

            def P(i, j, k):
                return point_offset + lpid(i, j, k)

            v = [
                P(i, j, k),
                P(i + 1, j, k),
                P(i + 1, j + 1, k),
                P(i, j + 1, k),
                P(i, j, k + 1),
                P(i + 1, j, k + 1),
                P(i + 1, j + 1, k + 1),
                P(i, j + 1, k + 1),
            ]
            for f in (
                [v[0], v[3], v[2], v[1]],
                [v[4], v[5], v[6], v[7]],
                [v[0], v[1], v[5], v[4]],
                [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]],
                [v[3], v[0], v[4], v[7]],
            ):
                face_map[frozenset(f)].append((f, c))

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    add_cell(i, j, k)

        for entries in face_map.values():
            if len(entries) == 2:
                (f0, c0), (f1, c1) = entries
                owner, neigh = min(c0, c1), max(c0, c1)
                face = f0 if c0 == owner else f1
                for f, c in entries:
                    if c == owner:
                        face = f
                internal.append((face, owner, neigh))
            else:
                face, c = entries[0]
                pts = [all_points[p] for p in face]
                cx = sum(p[0] for p in pts) / 4
                tol = 1e-8
                if bname == "L":
                    if abs(cx - 0.0) < tol:
                        bnd["inlet"].append((face, c))
                    elif abs(cx - 0.5) < tol:
                        bnd["AMI1"].append((face, c))
                    else:
                        bnd["walls"].append((face, c))
                else:
                    if abs(cx - 1.0) < tol:
                        bnd["outlet"].append((face, c))
                    elif abs(cx - 0.5) < tol:
                        bnd["AMI2"].append((face, c))
                    else:
                        bnd["walls"].append((face, c))

        cell_offset += nx * ny * nz
        point_offset += (nx + 1) * (ny + 1) * (nz + 1)

    internal.sort(key=lambda t: (t[1], t[2]))
    faces, owner, neighbour = [], [], []
    for f, o, n in internal:
        faces.append(f)
        owner.append(o)
        neighbour.append(n)
    nInternal = len(faces)
    boundary_meta = []
    for name, typ in (
        ("inlet", "patch"),
        ("outlet", "patch"),
        ("walls", "wall"),
        ("AMI1", "cyclicAMI"),
        ("AMI2", "cyclicAMI"),
    ):
        start = len(faces)
        for f, c in bnd[name]:
            faces.append(f)
            owner.append(c)
        boundary_meta.append((name, typ, start, len(faces) - start))

    with open(os.path.join(mesh, "points"), "w") as fh:
        fh.write(foam_header("vectorField", "points"))
        fh.write(f"{len(all_points)}\n(\n")
        for p in all_points:
            fh.write(f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})\n")
        fh.write(")\n")

    with open(os.path.join(mesh, "faces"), "w") as fh:
        fh.write(foam_header("faceList", "faces"))
        fh.write(f"{len(faces)}\n(\n")
        for f in faces:
            fh.write(f"4({' '.join(map(str, f))})\n")
        fh.write(")\n")

    with open(os.path.join(mesh, "owner"), "w") as fh:
        fh.write(foam_header("labelList", "owner"))
        fh.write(f"{len(owner)}\n(\n" + "\n".join(map(str, owner)) + "\n)\n")

    with open(os.path.join(mesh, "neighbour"), "w") as fh:
        fh.write(foam_header("labelList", "neighbour"))
        fh.write(
            f"{nInternal}\n(\n" + "\n".join(map(str, neighbour)) + "\n)\n"
        )

    with open(os.path.join(mesh, "boundary"), "w") as fh:
        fh.write(foam_header("polyBoundaryMesh", "boundary"))
        fh.write(f"{len(boundary_meta)}\n(\n")
        for name, typ, start, nFaces in boundary_meta:
            if typ == "cyclicAMI":
                nbr = "AMI2" if name == "AMI1" else "AMI1"
                fh.write(
                    f"""    {name}
    {{
        type            cyclicAMI;
        nFaces          {nFaces};
        startFace       {start};
        matchTolerance  0.001;
        transform       noOrdering;
        neighbourPatch  {nbr};
        lowWeightCorrection 0.2;
    }}
"""
                )
            else:
                fh.write(
                    f"""    {name}
    {{
        type            {typ};
        nFaces          {nFaces};
        startFace       {start};
    }}
"""
                )
        fh.write(")\n")

    print(
        f"AMI channel mesh: points={len(all_points)} cells={cell_offset} "
        f"AMI1={len(bnd['AMI1'])} AMI2={len(bnd['AMI2'])} -> {mesh}"
    )


if __name__ == "__main__":
    generate(sys.argv[1] if len(sys.argv) > 1 else ".")
