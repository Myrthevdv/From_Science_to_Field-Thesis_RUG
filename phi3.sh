#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=phi3_llm
#SBATCH --mem=50G

module purge
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/venvs/first_env/bin/activate

python3 -m pip install transformers tensorflow torch accelerate pandas openpyxl

python3 phi3.py  #Code that needs to be executed
