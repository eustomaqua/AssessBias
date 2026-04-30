#!/bin/bash

#SBATCH --job-name=MyJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

#SBATCH -p gpu --gres=gpu:0
# From here on, we can start our program


module load singularity
cd /home/qgl539/GitH/AB_29Apr  # <path>

for EXPT in KF_exp1b KF_exp1c  # mCV_exp1b mCV_exp1c
do
  for PREP in min_max none normalize standard min_abs
  do
    for DATA in ricci german ppr ppvr # adult
    do
      # python fair_int_exec.py -exp $EXPT -dat $DATA -pre $PREP
      singularity exec /home/qgl539/Singularity/enfair.sif bash -c "
            source /opt/conda/etc/profile.d/conda.sh
            conda activate py38
            python fair_int_exec.py -exp $EXPT -dat $DATA -pre $PREP
            "
    done
  done
done


# The partition is the queue you want to run on. standard is gpu and can be ommitted.
# number of independent tasks we are going to start in this script
# number of cpus we want to allocate for each program
# We expect that our program should not run longer than 2 days
# Note that a program will be killed once it exceeds this time!
# Skipping many options! see man sbatch


# chmod +x fair_int_main.sh
# cd ~/Singdocker
# module load singularity
# singularity run enfair.sif
# source activate ensem
# cd ~/GitH/PyFairness
#
# conda deactivate
# exit
