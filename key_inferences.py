# README
# Phillip Long
# August 13, 2023

# Uses a neural network to make predictions of songs' keys.

# python ./tempo_inferences.py labels_filepath nn_filepath n_predictions
# python /dfs7/adl/pnlong/artificial_dj/determine_tempo/tempo_inferences.py "/dfs7/adl/pnlong/artificial_dj/data/tempo_data.cluster.tsv" "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.pth" "100"


# IMPORTS
##################################################
import sys
from os.path import join, dirname
import torch
from numpy import mean, percentile
import matplotlib.pyplot as plt
from tempo_dataset import tempo_dataset # import dataset class
from tempo_neural_network import tempo_nn # import neural network class
# sys.argv = ("./tempo_inferences.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth", "20")
##################################################


# CONSTANTS
##################################################
LABELS_FILEPATH = sys.argv[1]
NN_FILEPATH = sys.argv[2]
##################################################


# RELOAD MODEL AND MAKE PREDICTIONS
##################################################

# I want to do all the predictions on CPU, GPU seems a bit much
device = "cpu"

# load back the model
tempo_nn = tempo_nn()
state_dict = torch.load(NN_FILEPATH, map_location = device)
tempo_nn.load_state_dict(state_dict["state_dict"])
print("Imported neural network parameters.")

# instantiate our dataset object
tempo_data = tempo_dataset(labels_filepath = LABELS_FILEPATH, set_type = "validate", device = device)

# get a sample from the urban sound dataset for inference
N_PREDICTIONS = int(sys.argv[3]) if len(sys.argv) == 4 else len(tempo_data)
N_PREDICTIONS = len(tempo_data) if N_PREDICTIONS > len(tempo_data)  else N_PREDICTIONS
inputs, targets = tempo_data.sample(n_predictions = N_PREDICTIONS)

# make an inference
print("Making predictions...")
print("----------------------------------------------------------------")
tempo_nn.eval()
with torch.no_grad():
    predictions = tempo_nn(inputs).view(N_PREDICTIONS, 1)

# print results
error = torch.abs(input = predictions - targets).numpy()
for i in range(N_PREDICTIONS):
    print(f"Case {i + 1}: Predicted = {predictions[i].item():.2f}, Expected = {targets[i].item():.2f}, Difference = {error[i].item():.2f}")
print("----------------------------------------------------------------")
print(f"Average Error: {mean(error):.2f}")

# calculate percentiles
percentiles = range(0, 101)
percentile_values = percentile(error, q = percentiles)
print("\n".join((f"{i}% percentile: {percentile_values[i]:.2f}" for i in (5, 10, 25, 50, 75, 90, 95))))

# output percentile plot
plt.plot(percentiles, percentile_values, "-b")
plt.xlabel("Percentile")
plt.ylabel("Difference")
plt.title("Validation Data Percentiles")
plt.savefig(join(dirname(NN_FILEPATH), "percentiles.validation.png")) # save image
print("Outputting percentile plot...")

##################################################
