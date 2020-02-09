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
set output_label = $argv[2]
echo $var
source /etc/profile.d/modules.csh
module load python/3.6.8
ncar_pylib my_npl_clone_casper

python3 main_cluster.py $var ${var}_log $output_label

