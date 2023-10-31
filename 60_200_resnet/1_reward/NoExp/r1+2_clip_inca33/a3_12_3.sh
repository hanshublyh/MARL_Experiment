#!/bin/bash
#SBATCH -J test111
#SBATCH -p kshcnormal02
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -o out.%j
#SBATCH -e err.%j
export PATH=/public/home/acposw0k49/anaconda3/bin:$PATH
python3 a3_3.py >>a3_12_3.txt