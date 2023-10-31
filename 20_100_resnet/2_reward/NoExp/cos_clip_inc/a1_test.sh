#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 28 
#SBATCH --ntasks-per-node=28
#SBATCH -p normal3,normal
#SBATCH --output=%j.out
#SBATCH --error=%j.err
python3 a1_test.py >>a1_12test.txt