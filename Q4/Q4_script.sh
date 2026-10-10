#!/bin/bash
#SBATCH --job-name=Q4_acp25ra
#SBATCH --account=rse-com6012
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=Q4_output.txt
#SBATCH --mail-type=FAIL

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

module load Java/17.0.4
module load Anaconda3/2024.02-1

source activate myspark

spark-submit --master local[4] --driver-memory 24g --conf spark.local.dir=/mnt/parscratch/users/acp25ra Q4_code.py
