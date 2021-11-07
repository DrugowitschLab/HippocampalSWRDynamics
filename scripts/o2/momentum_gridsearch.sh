#!/bin/bash
# Memory requirements: ~8Gb
#SBATCH -n 1                               # Request single core
#SBATCH -N 1                               # Request one node
#SBATCH -t 0-11:59                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=1G                           # Total memory MB
# SBATCH --array=0                        # jobs within array
#SBATCH -o ek195_%j_out                    # File to which STDOUT will be written, including job ID
#SBATCH -e ek195_%j_err                    # File to which STDERR will be written, including job ID
#SBATCH --mail-type=ALL                    # Type of email notification - BEGIN,END,FAIL,ALL


if [ "$1" == "real" ]; then
	if [ "$2" == "ripples" ]; then
		declare -a session_n_ripples=(322 527 222 257 406 296 594 356)
	elif [ "$2" == "placefieldID_shuffle" ]; then 
		declare -a session_n_ripples=(322 527 222 257 406 296 594 356)
	elif [ "$2" == "placefield_rotation" ]; then 
		declare -a session_n_ripples=(322 527 222 257 406 296 594 356)
	elif [ "$2" == "run_snippets" ]; then 
		declare -a session_n_ripples=(322 527 222 257 406 296 594 356)
	elif [ "$2" == "high_synchrony_events" ]; then 
		declare -a session_n_ripples=(395 729 430 360 794 537 721 503)
	else
		echo "Invalid rat-day"
	fi 
elif [ "$1" == "simulated" ]; then
	declare -a session_n_ripples=(100 100 100 100 100)
else
	echo "Invalid data-type"
fi

JOB_IDX=$SLURM_ARRAY_TASK_ID
N_RIPPLES=${session_n_ripples[(JOB_IDX)]}

echo "session: ${JOB_IDX}, n_ripples: ${N_RIPPLES}"

sbatch --array=0-$N_RIPPLES momentum_gridsearch_ratdayripple.sh $JOB_IDX $1 $2 $3 $4


