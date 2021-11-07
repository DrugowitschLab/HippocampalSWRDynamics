#!/bin/bash
# Memory requirements: ~8Gb
#SBATCH -n 1                               # Request single core
#SBATCH -N 1                               # Request one node
#SBATCH -t 0-11:59                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=1G                           # Total memory MB
#SBATCH -o ek195_%j_out                    # File to which STDOUT will be written, including job ID
#SBATCH -e ek195_%j_err                    # File to which STDERR will be written, including job ID
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=emkrause14@gmail.com   # Email to which notifications will be sent

RIPPLE=$SLURM_ARRAY_TASK_ID

echo "rat: $1, day: $2 ripple: ${RIPPLE}"

python momentum_gridsearch_ratdayripple.py $1 $2 $3 $4 $5 $RIPPLE