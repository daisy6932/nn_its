#!/bin/bash
#SBATCH --job-name=alt_cpu_test3
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/alt_cpu_test3/slurm-%j.out
#SBATCH --error=outputs/alt_cpu_test3/slurm-%j.err

set -euo pipefail
cd $SLURM_SUBMIT_DIR

echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
echo "DATE: $(date)"
echo "PY:   $(which python)"
python -V

python -u -m Archive_nl.run_scaf_server \
  --mode alt \
  --out_base outputs \
  --run_name alt_cpu_test3 \
  --n_train 5000 \
  --n_test 50000 \
  --n_features 20 \
  --n_classes 10 \
  --seed_train 0 \
  --seed_test 1 \
  --train_epochs 600 \
  --alt_cycles 4 \
  --alt_train_epochs 300 \
  --mu_tether 1.0 \
  --admm_epochs 15 \
  --theta_steps 10 \
  --z_steps 50 \
  --num_lambda 60 \
  --max_lambda 80 \
  --round_decimals 3
