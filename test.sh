#!/bin/bash
#SBATCH --job-name=test_key_nn          ## job name
#SBATCH -A tdlong_lab_gpu               ## account to charge
#SBATCH -p gpu                          ## run on the gpu partition
#SBATCH --nodes=1                       ## run on a single node
#SBATCH --ntasks=1                      ## request 1 task
#SBATCH --cpus-per-task=1               ## number of cores the job needs
#SBATCH --gres=gpu:V100:1               ## request 1 gpu of type V100

module load cuda/11.7.1
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# README
# Phillip Long
# August 4, 2023
# script to test the neural network; request GPU partition
# assumes I have already run key_dataset.py and key_neural_network.py

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"

# module load conda (hpc3 help says not to load python + conda together)
module load miniconda3/4.12.0

# activate conda env
eval "$(/opt/apps/miniconda3/4.12.0/bin/conda 'shell.bash' 'hook')"
conda activate artificial_dj

# run python script
python "${artificial_dj}/determine_key/key_inferences.py" "${data}/key_data.cluster.tsv" "${data}/key_nn.pretrained.pth"
