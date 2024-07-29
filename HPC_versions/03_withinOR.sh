#!/bin/bash
#SBATCH -N 1               	                                # number of nodes (no MPI, so we only use a single node)
#SBATCH -n 20            	                                # number of cores
#SBATCH --time=02:00:00    	                                # walltime allocation, which has the format (D-HH:MM:SS), here set to 1 hour
#SBATCH --mem=5GB         	                                # memory required per node (here set to 4 GB)
#SBATCH --output=slurm_outputs/withinOR/slurm-%A_%a.out
#SBATCH --account=OD-223323

# Notification configuration
#SBATCH --array=1-3
#SBATCH --mail-type=END					    	# Send a notification email when the job is done (=END)
#SBATCH --mail-user=rya200@csiro.au  	# Email to which notifications will be sent

#loading modules
module load python/3.12.3 

source $(which virtualenvwrapper_lazy.sh)

workon bad-emerge

# Execute the program

python3 code/12_within_OR_simulations.py
