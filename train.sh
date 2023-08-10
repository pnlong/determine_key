#!/bin/bash
#SBATCH --job-name=train_tempo_nn       ## job name
#SBATCH -A tdlong_lab_gpu               ## account to charge
#SBATCH -p gpu                          ## run on the gpu partition
#SBATCH --nodes=1                       ## run on a single node
#SBATCH --ntasks=1                      ## request 1 task
#SBATCH --cpus-per-task=1               ## number of cores the job needs
#SBATCH --gres=gpu:V100:1               ## request 1 gpu of type V100
#SBATCH --error=/dfs7/adl/pnlong/artificial_dj/determine_tempo/train.err            ## error log file
#SBATCH --output=/dfs7/adl/pnlong/artificial_dj/determine_tempo/train.out           ## output log file

# README
# Phillip Long
# August 4, 2023
# script to train the neural network on the cluster; request GPU partition
# assumes I have already run tempo_dataset.py

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"

# command to replace filepaths in data file
# sed "s+/Volumes/Seagate/artificial_dj_data+${data}+g" "${data}/tempo_data.tsv" > "${data}/tempo_data.cluster.tsv"

# module load conda (hpc3 help says not to load python + conda together)
module load miniconda3/4.12.0

# activate conda env
eval "$(/opt/apps/miniconda3/4.12.0/bin/conda 'shell.bash' 'hook')"
conda activate artificial_dj

# run python script
python "${artificial_dj}/determine_tempo/tempo_neural_network.py" "${data}/tempo_data.cluster.tsv" "${data}/tempo_nn.pth"
