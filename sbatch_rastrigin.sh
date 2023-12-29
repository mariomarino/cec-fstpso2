#!/bin/bash
#SBATCH -p LocalQ
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 tasks out of 128
#SBATCH --mem=2GB          # memory per node out of 246000MB

cd ~
cd "projects"
cd "cec-fstpso2"

eval "$(conda shell.bash hook)"
conda activate cec-fstpso
python function_test.py -F $1 -D $2 -R $3