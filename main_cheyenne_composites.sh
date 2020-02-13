#!/bin/tcsh
#SBATCH -J clusters_$var
#SBATCH -n 4
#SBATCH -t 24:00:00
#SBATCH -A UHAR0013
#SBATCH -p dav
#SBATCH --ntasks-per-node=4
#SBATCH -o output_$var.out
#SBATCH -e error_$var.err


set var = $argv[1]
set pred = $argv[2]
set numbers = $argv[3]
set boot = $argv[4]
echo $var
source /etc/profile.d/modules.csh
module load python/3.6.8
ncar_pylib -c 20190723 /glade/work/totz/my_npl_clone_casper
ncar_pylib my_npl_clone_casper

python3 main_composites.py -ini ini/composites_America_$var.ini --outputlabel not-normalized --predictand $pred --logfile logs/log_$var_.log--numbers $numbers --percentage $boot

