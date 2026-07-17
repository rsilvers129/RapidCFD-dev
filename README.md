# RapidCFD

Developed by SimFlow CFD  
https://sim-flow.com

### CFD toolbox running on CUDA

OpenFOAM solvers ported to Nvidia CUDA. All the calculations are done on the GPU giving a huge speed-up.
Still in development stage, waiting for your contribution!

### Features:
* most incompressible and compressible solvers on static mesh are available
* all the calculations are done on the GPU
* no overhead for GPU-CPU memory copy
* can run in parallel on multiple GPUs
* **dynamic-mesh branch:** `rhoCentralDyMFoamCUDA` with `solidBodyMotionFvMesh` + fused KT flux kernels

### Branch notes

| Branch | Status |
|--------|--------|
| `master` | Static-mesh CUDA 12.x / sm_120 build proven (`icoFoam` cavity) |
| `dynamic-mesh` | DyM + AMI + LABEL64 workstream (upstream integration branch) |
| `wopr-cuda` | **WOPR workstation track** — NVIDIA **RTX PRO 6000 Max-Q** (Blackwell, `sm_120`), CUDA 12.8, Ubuntu 24.04. Not B200. DyM solidBody proven; AMI tutorial proven; optional `WM_LABEL_SIZE=64`. |

### Build status (this fork, `dynamic-mesh`)

Successfully built and smoke-tested on:

* **Host:** Ubuntu 24.04 LTS
* **GPU:** NVIDIA RTX PRO 6000 Blackwell Max-Q (compute capability 12.0 / `sm_120`)
* **CUDA Toolkit:** 12.8
* **MPI:** system OpenMPI (`SYSTEMOPENMPI`)
* **Platform tag:** `linux64NvccDPOptSM120`

Verified:

* `icoFoam` lid-driven cavity (static)
* `rhoCentralDyMFoamCUDA` with `solidBodyMotionFvMesh` / `linearMotion` on a translating box (mesh transforms each step; run completes)

### Critical compiler flags (Blackwell / CUDA 12.8)

Do **not** enable the Antigravity flag set (`--use_fast_math`, `-std=c++14`, aggressive `-Xptxas`) on this GPU/toolkit combo — it produces `cudaErrorInvalidDeviceFunction` inside Thrust. The working rules are:

```
# wmake/rules/linux64Nvcc/c++
CC = nvcc -Xptxas -dlcm=cg -std=c++11 -m64 -arch=$(WM_GPU_ARCH)

# wmake/rules/linux64Nvcc/c++Opt
c++OPT = -O3
```

### Compilation (Ubuntu 24.04 + modern CUDA)

1. Ensure `nvcc` is on `PATH` (e.g. `/usr/local/cuda-12.8/bin`).

2. Install build deps: `g++`, `make`, `flex`, `bison`, `openmpi` (`libopenmpi-dev`, `openmpi-bin`).

3. Layout:

   ```bash
   export FOAM_INST_DIR=/path/to/parent   # contains RapidCFD-dev/
   ```

4. Set GPU arch in `etc/bashrc` / prefs (`WM_GPU_ARCH`):

   | GPU family | `WM_GPU_ARCH` |
   |------------|---------------|
   | Ada (RTX 40) | `sm_89` |
   | Hopper | `sm_90` |
   | Blackwell (RTX PRO 6000 / RTX 50) | `sm_120` |

5. Build:

   ```bash
   export FOAM_INST_DIR=/path/to/parent
   source $FOAM_INST_DIR/RapidCFD-dev/etc/bashrc
   export WM_NCOMPPROCS=$(nproc)
   mkdir -p "$FOAM_EXT_LIBBIN"
   ./Allwmake 2>&1 | tee build.log
   ```

6. Run DyM solver:

   ```bash
   rhoCentralDyMFoamCUDA -device 0 -case /path/to/case
   ```

   Case needs `constant/dynamicMeshDict` with e.g. `solidBodyMotionFvMesh` + `linearMotion`, plus standard `rhoCentralFoam` thermo/fields.

### Known limitations / unfinished Antigravity work

* Prefer `snGradSchemes { default orthogonal; }` (and matching laplacian). `corrected` snGrad has hit Tensor `invalid device function` paths historically.
* No mesh utilities in this tree (no `blockMesh`) — generate meshes externally.
* **AMI proven** (see `tutorials/AMI/`): non-conformal cyclicAMI 64↔25 faces.
  - `icoFoam` (static), `rhoCentralFoamCUDA` (static), `rhoCentralDyMFoamCUDA` + linearMotion (AMI rebuild each step).
  - Still unproven: sliding/rotating multi-zone AMI (rotor-stator style cell zones).
* `fastMeshUpdate.H` (analytical meshPhi for pure linearMotion) exists but the current solver uses full `mesh.update()`.
* Multi-GPU still benefits from ThirdParty CUDA-aware MPI (optional).

### Original notes (Ubuntu 16.04 / CUDA 8)

* ensure CUDA 7.5 is not installed from Ubuntu repositories
* ensure you are using an nVidia driver compatible with CUDA 8
* ThirdParty-dev is needed for multiple GPUs with the bundled OpenMPI build
