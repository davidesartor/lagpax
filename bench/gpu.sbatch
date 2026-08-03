#!/bin/bash
#SBATCH --job-name=lagp-bench
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
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

# accuracy first: cheap, and it is the run that fails loudly if the GPU changes the math
uv run --no-sync -m bench.accuracy
uv run --no-sync -m bench.timing --batch 256
uv run --no-sync -m bench.plot
