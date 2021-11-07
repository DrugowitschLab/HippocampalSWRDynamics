#!/bin/bash
# Memory requirements: ~8Gb
#SBATCH -n 1                               # Request single core
#SBATCH -N 1                               # Request one node
#SBATCH -t 0-11:59                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=10G                          # Total memory MB
# SBATCH --array=0                          # jobs within array
#SBATCH -o ek195_%j_out                    # File to which STDOUT will be written, including job ID
#SBATCH -e ek195_%j_err                    # File to which STDERR will be written, including job ID
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL


JOB_IDX=$SLURM_ARRAY_TASK_ID

echo "session: ${SLURM_ARRAY_TASK_ID}"

python diffusion_gridsearch.py $JOB_IDX $1 $2 $3 $4