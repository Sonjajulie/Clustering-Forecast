#!/bin/bash
#SBATCH -J clusters_$pred
#SBATCH -p shared # Partition
#SBATCH -n 1
#SBATCH -t 5-00:00  # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=30G  # Memory request (16Gb)
#SBATCH --ntasks-per-node=16
#SBATCH -o output_$pred.out
#SBATCH -e error_$pred.err


pred=$1
prec=$2
datamin=$3
datamax=$4
outputlabel=$5
echo $pred

module load python
source activate clustering-forecast
echo which python3
# python3 ../main_cluster_forecast_nn_talos.py -ini ../ini/clusters_America_${pred}_Forecast_opt.ini --outputlabel $outputlabel --predictand $pred --logfile logs/log_${pred}.log --datarange $datamin $datamax
python3 ../main_cluster_forecast_nn_talos.py -ini ../ini/clusters_America_${pred}_Forecast_opt_${prec}.ini --outputlabel $outputlabel --predictand $pred --logfile logs/log_${pred}_${prec}.log --datarange $datamin $datamax --forecast_precursors ${prec}




