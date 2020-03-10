#!/bin/bash
#SBATCH -J clusters_$pred
#SBATCH -p shared # Partition
#SBATCH -n 1
#SBATCH -t 00:10:00 
#SBATCH --mem=4G # Memory request (4Gb)
#SBATCH --ntasks-per-node=4
#SBATCH -o output_$pred.out
#SBATCH -e error_$pred.err


pred=$argv[1]
datamin=$argv[2]
datamax=$argv[3]
outputlabel=$argv[4]
echo $pred

module load python
source activate clustering-forecast

python3 main_cluster_forecast_all_precursors.py -ini ini/clusters_America_${pred}_Forecast.ini --outputlabel $outputlabel --predictand $pred --logfile logs/log_${pred}.log --datarange $datamin $datamax

