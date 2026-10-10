#!/bin/bash
#SBATCH --job-name=acp25ra_q3
#SBATCH --output=Q3_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:30:00
#SBATCH --mail-type=END,FAIL

module load Java/17.0.4
module load Anaconda3/2022.05

source activate myspark

spark-submit --master local[10] --driver-memory 70g --conf spark.driver.maxResultSize=8g --conf spark.local.dir=/mnt/parscratch/users/acp25ra Q3_code.py
