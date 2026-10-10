#!/bin/bash
#SBATCH --job-name=Q2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --time=03:00:00
#SBATCH --output=./Q2_output.txt

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit --driver-memory 90g --master local[10] Q2_code.py
