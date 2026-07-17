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

### Build status (this fork)

Successfully built and smoke-tested on:

* **Host:** Ubuntu 24.04 LTS
* **GPU:** NVIDIA RTX PRO 6000 Blackwell Max-Q (compute capability 12.0 / `sm_120`)
* **CUDA Toolkit:** 12.8
* **MPI:** system OpenMPI (`SYSTEMOPENMPI`)
* **Platform tag:** `linux64NvccDPOptSM120`

Verified: `icoFoam` lid-driven cavity (20×20×1) runs to completion on GPU with orthogonal schemes.

### Compilation (Ubuntu 24.04 + modern CUDA)

1. Install CUDA toolkit (nvcc) and a compatible NVIDIA driver. Ensure `nvcc` is on `PATH`:

   ```bash
   export PATH=/usr/local/cuda-12.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
   ```

2. Install build deps: `g++`, `make`, `flex`, `bison`, `openmpi` (`libopenmpi-dev`, `openmpi-bin`).

3. Clone so the install root contains `RapidCFD-dev` (and optionally `ThirdParty-dev`):

   ```bash
   export FOAM_INST_DIR=/path/to/parent   # e.g. $HOME/dev
   # layout: $FOAM_INST_DIR/RapidCFD-dev
   ```

4. Set GPU architecture in `etc/bashrc` / `etc/prefs.sh` (`WM_GPU_ARCH`):

   | GPU family | `WM_GPU_ARCH` |
   |------------|---------------|
   | Ada (RTX 40) | `sm_89` |
   | Hopper | `sm_90` |
   | Blackwell (RTX PRO 6000 / RTX 50) | `sm_120` |

5. Source and build:

   ```bash
   export FOAM_INST_DIR=/path/to/parent
   source $FOAM_INST_DIR/RapidCFD-dev/etc/bashrc
   export WM_NCOMPPROCS=$(nproc)   # or a lower number
   mkdir -p "$FOAM_EXT_LIBBIN"
   ./Allwmake 2>&1 | tee build.log
   ```

6. Run solvers with a GPU device:

   ```bash
   icoFoam -device 0 -case /path/to/case
   ```

Notes:

* This tree only builds **solvers** (no `blockMesh` / mesh utilities). Generate meshes with system OpenFOAM or another tool.
* Prefer `snGradSchemes { default orthogonal; }` and matching laplacian schemes for now; `corrected` snGrad can hit a known CUDA `invalid device function` path on Tensor gradients under CUDA 12.8 / sm_120.
* Multi-GPU still benefits from ThirdParty CUDA-aware MPI (optional).

### Original notes (Ubuntu 16.04 / CUDA 8)

* ensure CUDA 7.5 is not installed from Ubuntu repositories
* ensure you are using an nVidia driver compatible with CUDA 8
* download CUDA 8.0 from NVIDIA's archive
* to compile in parallel, `export WM_NCOMPPROCS=10`
* ThirdParty-dev is needed for multiple GPUs with the bundled OpenMPI build
