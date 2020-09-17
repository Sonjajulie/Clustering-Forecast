#!/bin/bash
#SBATCH -J clusters_$pred
#SBATCH -p shared # Partition
#SBATCH -n 1
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=16G  # Memory request (8Gb)
#SBATCH -o output_$pred.out
#SBATCH -e error_$pred.err


var=$1
pred=$2
numbers=$3
boot=$4
outputlabel=$5
echo $pred

module load python
source activate clustering-forecast


python3 ../main_composites.py -ini ../ini/composites_America_$var.ini --outputlabel $outputlabel --predictand $pred --logfile logs/log_${var}_${pred}.log --numbers $numbers --percentage $boot

