# AMI tutorials (RapidCFD)

## amiChannel

Non-conformal **cyclicAMI** channel (two disconnected hex blocks):

- AMI1: 64 faces (8×8)
- AMI2: 25 faces (5×5)
- Proven weights: sum(weights) min/max/average = 1/1/1 on both sides

### Smoke tests proven on RTX PRO 6000 Blackwell (`sm_120`)

| Case | Solver | Result |
|------|--------|--------|
| Static AMI | `icoFoam` | 200 steps, healthy residuals, no NaN |
| Static AMI | `rhoCentralFoamCUDA` | Completes to endTime |
| AMI + solid-body motion | `rhoCentralDyMFoamCUDA` | AMI rebuilt each `mesh.update()`, weights stay 1.0 |

### Run

```bash
export FOAM_INST_DIR=...   # parent of RapidCFD-dev
source $FOAM_INST_DIR/RapidCFD-dev/etc/bashrc
cd tutorials/AMI/amiChannel
./Allrun
```

Regenerate mesh:

```bash
python3 ../generate_ami_channel.py .
```
