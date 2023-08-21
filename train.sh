#!/bin/bash
#SBATCH --job-name=train_key_nn       ## job name
#SBATCH -A tdlong_lab_gpu               ## account to charge
#SBATCH -p gpu                          ## run on the gpu partition
#SBATCH --nodes=1                       ## run on a single node
#SBATCH --ntasks=1                      ## request 1 task
#SBATCH --cpus-per-task=1               ## number of cores the job needs
#SBATCH --gres=gpu:V100:1               ## request 1 gpu of type V100

echo "JOB ID: ${SLURM_JOBID}"

# README
# Phillip Long
# August 11, 2023
# script to train the neural network on the cluster; request GPU partition
# assumes I have already run key_dataset.py

artificial_dj="/dfs7/adl/pnlong/artificial_dj"
data="${artificial_dj}/data"
output_prefix="${data}/key_nn.pretrained"

# command to replace filepaths in data file
# sed "s+/Volumes/Seagate/artificial_dj_data+${data}+g" "${data}/key_data.tsv" > "${data}/key_data.cluster.tsv"

# set number of epochs
epochs="default"
while getopts ":e:" opt
do
    case $opt in
        e) epochs=$OPTARG;;
       \?) echo "ERROR: Invalid option: ${0} [-e <e>]"
           exit 1;;
    esac
done
echo "EPOCHS: ${epochs}"

# module load conda (hpc3 help says not to load python + conda together)
module load miniconda3/4.12.0

# activate conda env
eval "$(/opt/apps/miniconda3/4.12.0/bin/conda 'shell.bash' 'hook')"
conda activate artificial_dj

# run python training script
python "${artificial_dj}/determine_key/key_neural_network.py" "${data}/key_data.cluster.tsv" "${output_prefix}.pth" "${epochs}"

# create plots
python "${artificial_dj}/determine_key/training_plots.py" "${output_prefix}.history.tsv" "${output_prefix}.percentiles_history.tsv" "${output_prefix}.png"
