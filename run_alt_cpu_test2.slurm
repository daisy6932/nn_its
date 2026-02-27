#!/bin/bash
#SBATCH --job-name=alt_cpu_test2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=outputs/alt_cpu_test2/slurm-%j.out
#SBATCH --error=outputs/alt_cpu_test2/slurm-%j.err

set -euo pipefail
cd $SLURM_SUBMIT_DIR

# (可选) 打印环境信息，方便排查
echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
echo "DATE: $(date)"
which python
python -V

python -u -m Archive_nl.run_scaf_server \
  --mode alt \
  --out_base outputs \
  --run_name alt_cpu_test2 \
  --n_train 1000 \
  --n_test 10000 \
  --n_features 20 \
  --n_classes 10 \
  --seed_train 0 \
  --seed_test 1 \
  --train_epochs 400 \
  --alt_cycles 3 \
  --alt_train_epochs 200 \
  --mu_tether 1.0 \
  --admm_epochs 10 \
  --theta_steps 10 \
  --z_steps 50 \
  --num_lambda 40 \
  --max_lambda 50 \
  --round_decimals 3
