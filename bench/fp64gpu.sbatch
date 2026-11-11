#!/bin/bash
#SBATCH --job-name=lagp-fp64
# power9-gpu also has v100s but is ppc64le, where no jax cuda wheels exist
#SBATCH --partition=gpu,gpu-preempt,arm-gpu
#SBATCH --constraint="a100|h100|v100|gh200"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=bench/slurm-%j.out
#SBATCH --error=bench/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

ARCH=$(uname -m)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
TAG="${GPU}"
echo "arch=$ARCH gpu=$GPU"
# the laGP column is a host-CPU number even on a GPU node, so put the CPU on the record
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core"

# the repo .venv is resolved for x86_64; a gh200 lands on aarch64 and needs its own
if [ "$ARCH" != "x86_64" ]; then
  export UV_PROJECT_ENVIRONMENT=".venv-$ARCH"
  uv sync --extra gpu
  RUN="uv run"
  SKIP_R="--skip-r"   # no r/4.4.0 module built for this arch
else
  module load r/4.4.0
  export R_HOME=$(R RHOME)
  RUN="uv run --no-sync"
  SKIP_R=""
fi

$RUN python -c "import jax; print(jax.devices())"

# batch sweep: settles whether the remaining cost is launch latency or arithmetic
for b in 64 256 1024; do
  $RUN -m bench.timing --batch $b --skip-r --out "timing-$TAG-b$b.csv"
done

# full sweep, with laGP on this node's own cores when the arch supports it
$RUN -m bench.timing --batch 256 $SKIP_R --out "timing-$TAG.csv"
