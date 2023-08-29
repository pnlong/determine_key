#!/bin/bash
#SBATCH --job-name=test_key_nn          ## job name
#SBATCH -A tdlong_lab                   ## account to charge
#SBATCH -p standard                     ## run on the standard cpu partition
#SBATCH --nodes=1                       ## run on a single node
#SBATCH --ntasks=1                      ## request 1 task
#SBATCH --cpus-per-task=1               ## number of cores the job needs

# README
# Phillip Long
# August 20, 2023
# script to test the neural network; request CPU partition
# assumes I have already run key_dataset.py and key_neural_network.py

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"
output_prefix="${data}/key_nn"

# module load conda (hpc3 help says not to load python + conda together)
module load miniconda3/4.12.0

# activate conda env
eval "$(/opt/apps/miniconda3/4.12.0/bin/conda 'shell.bash' 'hook')"
conda activate artificial_dj

# run python script
python "${artificial_dj}/determine_key/key_inferences.py" "${data}/key_data.cluster.tsv" "${output_prefix}.pth"
