#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --mem=5G  
#SBATCH --cpus-per-task=2
#SBATCH --output=./Q1_output.txt 

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit Q1_code.py
