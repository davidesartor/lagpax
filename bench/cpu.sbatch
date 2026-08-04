#!/bin/bash
#SBATCH --job-name=lagp-cpu
#SBATCH --partition=cpu
# amd7763 is dual-socket Zen3, 64 cores/socket = 128 real cores, and the largest
# single-model pool in the partition (20 nodes). Pinned rather than left to the
# scheduler because `cpu` spans ten node types and a laGP baseline is a host-CPU
# number -- an unpinned run produces a ratio that cannot be compared to any other.
#SBATCH --constraint=amd7763
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=bench/slurm-%j.out
#SBATCH --error=bench/slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

hostname
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core"

module load r/4.4.0
export R_HOME=$(R RHOME)

# --r-threads keeps the 1-core row: on a healthy node laGP gains ~6x from 16 threads,
# and a ratio near 1x is how a node that never got its cores announces itself.
uv run --no-sync -m bench.accuracy
uv run --no-sync -m bench.timing --r-threads 1 all
uv run --no-sync -m bench.timing --suite locations --cases 0 1 2 --r-threads 1 all \
    --out timing-cpu-loc.csv
uv run --no-sync -m bench.plot
