#!/bin/bash
#SBATCH --job-name=kdnysm
#SBATCH --output=kidney_competition.out
#
#SBATCH --partition=hlwill
#SBATCH --time=7-00:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=23
#SBATCH --ntasks=1
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gsmoore@stanford.edu

module load python/3.6.1

python3 simulation.py
