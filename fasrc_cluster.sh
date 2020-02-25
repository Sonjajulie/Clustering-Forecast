#!/bin/bash
#SBATCH -J NMME_data # A single job name for the array
#SBATCH -p shared # Partition
#SBATCH -n 1
#SBATCH -t 00:10:00 
#SBATCH --mem=4G # Memory request (4Gb)
#SBATCH --ntasks-per-node=4
#SBATCH -o %A-%j.o # Standard output
#SBATCH -e %A-%j.e # Standard error


## Load required modules 
## https://www.rc.fas.harvard.edu/resources/documentation/software-on-the-cluster/python/
module load Anaconda3/5.0.1-fasrc02

#you must enter the directory where your data is, with an absolute path
#cd /n/home04/stotz/DownloadClimateModelData

#source local environment
source activate download-climate-data

## execute with sbatch submit.sh
# for i in {0..3}
# do
#    python3 main_cluster.py $i
#done
python3 test_forecast_nn.py

source deactivate 
