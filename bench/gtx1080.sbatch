#!/bin/bash
#SBATCH --job-name=lagp-1080
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1080_ti:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=bench/slurm-%j.out
#SBATCH --error=bench/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load r/4.4.0
export R_HOME=$(R RHOME)

nvidia-smi -L
uv run --no-sync python -c "import jax; print(jax.devices())"

# batch sweep first: JAX only, cheap, and it settles whether we are launch-latency bound
for b in 64 256 1024; do
  uv run --no-sync -m bench.timing --batch $b --skip-r --out "timing-1080-b$b.csv"
done

# then the full sweep with laGP on this node's own 16 cores
uv run --no-sync -m bench.timing --batch 256 --out timing-1080.csv
uv run --no-sync -m bench.accuracy
