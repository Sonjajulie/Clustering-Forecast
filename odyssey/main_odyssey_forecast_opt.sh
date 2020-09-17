#!/bin/bash
#SBATCH -J clusters_$pred
#SBATCH -p shared # Partition
#SBATCH -n 1
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=8G  # Memory request (8Gb)
#SBATCH --ntasks-per-node=8
#SBATCH -o output_$pred.out
#SBATCH -e error_$pred.err


pred=$1
prec=$2
datamin=$3
datamax=$4
outputlabel=$5
echo $pred
echo $prec

module load python
source activate clustering-forecast
# /home/sonja/mnt/harvard_cluster/Clustering-Forecast/ini/clusters_America_TS_Forecast_opt_Z500.ini

python3 ../main_cluster_forecast_all_precursors_opt.py -ini ../ini/clusters_America_${pred}_Forecast_opt_${prec}.ini --outputlabel $outputlabel --predictand $pred --logfile logs/log_${pred}_${prec}.log --datarange $datamin $datamax --forecast_precursors ${prec}

