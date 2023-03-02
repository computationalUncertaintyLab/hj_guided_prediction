#!/bin/bash

#SBATCH --partition=health,lts,hawkcpu,infolab,engi,eng

#--Request 1 hour of computing time
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=12

#--Give a name to your job to aid in monitoring
#SBATCH --job-name flumodel

#--Write Standard Output and Error
#SBATCH --output="flu.%j.%N.out"

cd ${SLURM_SUBMIT_DIR} # cd to directory where you submitted the job

#--launch job
module load anaconda3
conda activate $HOME/condaenv/flu

#--export environmental variables
export N=${N}
export TIME=${TIME}
export NOISE=${NOISE}

python build_data_for_plot__cut__plushj__pieces.py --N ${N} --NOISE ${NOISE} --TIME ${TIME}

exit
