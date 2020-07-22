#!/bin/tcsh
#SBATCH -J cl_f_${var}_${pred}
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -A UHAR0013
#SBATCH -p dav
#SBATCH --ntasks-per-node=4
#SBATCH -o output_cl_${var}_${pred}.out
#SBATCH -e error_cl_${var}_${pred}.err


set var = $argv[1]
set pred = $argv[2]
set outputlabel = $argv[3]
echo $var
source /etc/profile.d/modules.csh
module load python/3.6.8
ncar_pylib -c 20190723 /glade/work/totz/my_npl_clone_casper
ncar_pylib my_npl_clone_casper

python3 ../main_cluster_forecast_all_precursors_opt.py -ini ../ini/clusters_America_${var}_Forecast.ini .ini --outputlabel $outputlabel --predictand $pred --logfile logs/log_${var}.log - --datarange 0 1980 --forecast_precursors $pred
