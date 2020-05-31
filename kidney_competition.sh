#!/bin/bash
#SBATCH --job-name=kdnysm
#SBATCH --output=kidney-simulation.log
#
#SBATCH --partition=hlwill
#SBATCH --time=1-00:00:00
#SBATCH --mem=50GB
#SBATCH --ntasks=23
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gsmoore@stanford.edu

module load python

python simulation.py