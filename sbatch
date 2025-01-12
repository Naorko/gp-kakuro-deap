#!/bin/bash

##################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like ##SBATCH
##################

#SBATCH --partition main		### specify partition name where to run a job. debug: 2 hours limit; short: 7 days limit; gtx1080: 7 days
#SBATCH --time 2-12:00:00			### limit the time of job running, partition limit can override this. Format: D-H:MM:SS
#SBATCH --job-name GP-RHP	### name of the job
#SBATCH --output job_outputs/%A-%a.out			### output log for running job - %J for job number
#SBATCH --mail-user=naorko@post.bgu.ac.il	### user email for sending job status
#SBATCH --mail-type=END   ###ARRAY_TASKS,FAIL			### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

##SBATCH --gpus=1				### number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU
##SBATCH --mem=32G				### amount of RAM memory
#SBATCH --cpus-per-task=12			### number of CPU cores

#SBATCH --array=1-18

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_ARRAYTASKID:\t" $SLURM_ARRAY_TASK_ID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start you code below ####
module load anaconda				### load anaconda module (must present when working with conda environments)
source activate deap-env				### activating environment, environment must be configured before running the job
python /home/naorko/EA/gp-kakuro-deap/gp_main.py $SLURM_ARRAY_TASK_ID		### replace with your own command