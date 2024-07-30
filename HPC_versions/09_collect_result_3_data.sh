#!/bin/bash
#SBATCH -N 1               	                                # number of nodes (no MPI, so we only use a single node)
#SBATCH -n 10            	                                # number of cores
#SBATCH --time=01:00:00    	                                # walltime allocation, which has the format (D-HH:MM:SS), here set to 1 hour
#SBATCH --mem=5GB         	                                # memory required per node (here set to 4 GB)
#SBATCH --output=slurm_outputs/slurm-%A_%a.out


# Notification configuration
#SBATCH --array=1-1
#SBATCH --mail-type=END					    	# Send a notification email when the job is done (=END)
#SBATCH --mail-user=rya200@csiro.au  	# Email to which notifications will be sent

#loading modules
module load python/3.12.3 

source $(which virtualenvwrapper_lazy.sh)

workon bad-emerge

# Execute the program

python3 code/07_results_3_collect_data.py
