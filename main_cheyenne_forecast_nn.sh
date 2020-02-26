#!/bin/tcsh
#SBATCH -J clusters_$pred
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -A UHAR0013
#SBATCH -p dav
#SBATCH --ntasks-per-node=4
#SBATCH -o output_$pred.out
#SBATCH -e error_$pred.err


set pred = $argv[1]
set datamin = $argv[2]
set datamax = $argv[3]
set outputlabel = $argv[4]
echo $pred
source /etc/profile.d/modules.csh
module load python/3.6.8
ncar_pylib -c 20190723 /glade/work/totz/my_npl_clone_casper
ncar_pylib my_npl_clone_casper

python3 main_cluster_forecast_all_precursors.py -ini ini/clusters_America_${pred}_Forecast.ini --outputlabel $outputlabel --predictand $pred --logfile logs/log_${pred}.log --datarange $datamin $datamax

