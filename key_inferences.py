# README
# Phillip Long
# August 29, 2023

# Uses multiple trained neural networks to make predictions of songs' keys.

# python ./key_inferences.py labels_filepath key_class_nn_filepath key_quality_nn
# python /dfs7/adl/pnlong/artificial_dj/determine_key/key_inferences.py "/dfs7/adl/pnlong/artificial_dj/data/key_data.cluster.tsv" "/dfs7/adl/pnlong/artificial_dj/data/key_class_nn.pth" "/dfs7/adl/pnlong/artificial_dj/data/key_quality_nn.pth"


# IMPORTS
##################################################
import sys
from os.path import dirname, join
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from numpy import percentile
import matplotlib.pyplot as plt
from key_dataset import key_dataset, key_class_dataset, key_quality_dataset, KEY_CLASS_MAPPINGS, KEY_QUALITY_MAPPINGS # import dataset class
from key_class_neural_network import key_class_nn # import key class neural network class
from key_quality_neural_network import key_quality_nn, CONFIDENCE_THRESHOLD # import key quality neural network class
# sys.argv = ("./key_inferences.py", "/dfs7/adl/pnlong/artificial_dj/data/key_data.cluster.tsv", "/dfs7/adl/pnlong/artificial_dj/data/key_class_nn.pth", "/dfs7/adl/pnlong/artificial_dj/data/key_quality_nn.pth")
##################################################


# CONSTANTS
##################################################
BATCH_SIZE = 32
LABELS_FILEPATH = sys.argv[1]
NN_FILEPATH = {
    "class": sys.argv[2],
    "quality": sys.argv[3]
}
OUTPUT_PREFIX = ".".join(NN_FILEPATH["class"].split(".")[:-1]).split(".")
OUTPUT_PREFIX[0] = join(dirname(OUTPUT_PREFIX[0]), "key_nn")
OUTPUT_PREFIX = ".".join(OUTPUT_PREFIX)
OUTPUT_FILEPATH = OUTPUT_PREFIX + ".test.png"
INCLUDE_CONFUSION_MATRIX = False # whether or not to calculate and display a confusion matrix
##################################################


# RELOAD MODEL
##################################################

# I want to do all the predictions on CPU, GPU seems a bit much
print("----------------------------------------------------------------")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

# load back the model
key_class_nn = key_class_nn().to(device)
key_class_nn.load_state_dict(torch.load(NN_FILEPATH["class"], map_location = device)["state_dict"])
key_quality_nn = key_quality_nn().to(device)
key_quality_nn.load_state_dict(torch.load(NN_FILEPATH["quality"], map_location = device)["state_dict"])
print("Imported neural network parameters.")

# instantiate our dataset object and data loader
data = {
    "class": key_class_dataset(labels_filepath = LABELS_FILEPATH, set_type = "test", device = device),
    "quality": key_quality_dataset(labels_filepath = LABELS_FILEPATH, set_type = "test", device = device),
    "key": key_dataset(labels_filepath = LABELS_FILEPATH, set_type = "test", device = device)
}
data_loader = {
    "class": DataLoader(dataset = data["class"], batch_size = BATCH_SIZE, shuffle = True),
    "quality": DataLoader(dataset = data["quality"], batch_size = BATCH_SIZE, shuffle = True),
    "key": DataLoader(dataset = data["key"], batch_size = BATCH_SIZE, shuffle = True)
}

# instantiate statistics
accuracy = dict(zip(data.keys(), (0.0,) * len(data.keys())))
##################################################


# MAKE PREDICTIONS
##################################################
with torch.no_grad():
            
    # set to evaluation mode
    key_class_nn.eval()
    key_quality_nn.eval()

    # validation loop
    if INCLUDE_CONFUSION_MATRIX:
        confusion_matrix = { # rows = actual, columns = prediction
            "class": torch.zeros(len(KEY_CLASS_MAPPINGS), len(KEY_CLASS_MAPPINGS), dtype = torch.float32).to(device),
            "quality": torch.zeros(len(KEY_QUALITY_MAPPINGS), len(KEY_QUALITY_MAPPINGS), dtype = torch.float32).to(device)
        }
    error = torch.tensor(data = [], dtype = torch.float32).to(device)
    for (inputs_class, labels_class), (inputs_quality, labels_quality), (_, labels) in tqdm(zip(data_loader["class"], data_loader["quality"], data_loader["key"]), desc = "Making predictions", total = (len(data["key"]) // BATCH_SIZE) + 1): # https://stackoverflow.com/questions/41171191/tqdm-progressbar-and-zip-built-in-do-not-work-together#:~:text=tqdm%20can%20be%20used%20with,provided%20in%20the%20tqdm%20call.&text=The%20issue%20is%20that%20tqdm,single%20length%20of%20its%20arguments.

        # register inputs and labels with device
        inputs_class, labels_class, inputs_quality, labels_quality, labels = inputs_class.to(device), labels_class.view(-1).to(device), inputs_quality.to(device), labels_quality.view(-1).to(device), labels.view(-1).to(device)

        # KEY CLASS
        ##################################################

        # forward pass: compute predictions on input data using the model
        predictions_class = key_class_nn(inputs_class)

        # calculate accuracy
        predictions_class = torch.argmax(input = predictions_class, dim = 1, keepdim = True).view(-1) # convert to class indicies
        accuracy["class"] += torch.sum(input = (predictions_class == labels_class)).item() # add value to running accuracy count
        error_batch = torch.abs(input = predictions_class - labels_class) # compute absolute error, flattening dataset
        error_batch = torch.tensor(data = list(map(lambda difference: min(difference, len(KEY_CLASS_MAPPINGS) - difference), error_batch)), dtype = error.dtype).view(-1).to(device) # previously lambda difference: len(KEY_CLASS_MAPPINGS) - difference if difference > len(KEY_CLASS_MAPPINGS) // 2 else difference
        error = torch.cat(tensors = (error, error_batch), dim = 0) # append to error

        ##################################################


        # KEY QUALITY
        ##################################################

        # forward pass: compute predictions on input data using the model
        predictions_quality = key_quality_nn(inputs_quality)

        # calculate accuracy
        predictions_quality = (predictions_quality >= CONFIDENCE_THRESHOLD).view(-1) # convert to class indicies
        accuracy["quality"] += torch.sum(input = (predictions_quality == labels_quality)).item() # add value to running accuracy count

        ##################################################

        
        # ADD TO CONFUSION MATRIX
        ##################################################
        if INCLUDE_CONFUSION_MATRIX:
            confusion_matrix["class"] += torch.tensor(data = [[sum(predicted_key_class_index == predictions_class[labels_class == actual_key_class_index]) for predicted_key_class_index in range(len(KEY_CLASS_MAPPINGS))] for actual_key_class_index in range(len(KEY_CLASS_MAPPINGS))], dtype = confusion_matrix["class"].dtype).to(device)
            confusion_matrix["quality"] += torch.tensor(data = [[sum(predicted_key_quality_index == predictions_quality[labels_quality == actual_key_quality_index]) for predicted_key_quality_index in range(len(KEY_QUALITY_MAPPINGS))] for actual_key_quality_index in range(len(KEY_QUALITY_MAPPINGS))], dtype = confusion_matrix["quality"].dtype).to(device)
        ##################################################


        # GENERAL KEY
        ##################################################

        predictions = (2 * predictions_class) + predictions_quality # get predictions
        accuracy["key"] += torch.sum(input = (predictions == labels)).item() # add value to running accuracy count

        ##################################################


print("----------------------------------------------------------------")

# compute accuracy percentages
accuracy["class"] /= (len(data["class"]) / 100)
accuracy["quality"] /= (len(data["quality"]) / 100)
accuracy["key"] /= (len(data["key"]) / 100)

if INCLUDE_CONFUSION_MATRIX:
    # normalize confusion matrix
    normalized_confusion_matrix = {
        "precision" + "class": confusion_matrix["class"] / torch.sum(input = confusion_matrix["class"], axis = 0).view(1, -1),
        "recall" + "class": confusion_matrix["class"] / torch.sum(input = confusion_matrix["class"], axis = 1).view(-1, 1),
        "precision" + "quality": confusion_matrix["quality"] / torch.sum(input = confusion_matrix["quality"], axis = 0).view(1, -1),
        "recall" + "quality": confusion_matrix["quality"] / torch.sum(input = confusion_matrix["quality"], axis = 1).view(-1, 1),
    }

##################################################


# PRINT RESULTS
##################################################

# print accuracies
print(f"Determining Key Accuracy: {accuracy['key']:.2f}%")
print(f"Key Quality Accuracy: {accuracy['quality']:.2f}%")
print(f"Key Class Accuracy: {accuracy['class']:.2f}%")

# print results
print(f"Average Error: {torch.mean(input = error).item():.2f}")

# calculate percentiles
percentiles = range(0, 101)
percentile_values = percentile(error.numpy(force = True), q = percentiles)
print(f"Minimum Error: {percentile_values[0]:.2f}")
print(*(f"{i}% Percentile: {percentile_values[i]:.2f}" for i in (5, 10, 25, 50, 75, 90, 95)), sep = "\n")
print(f"Maximum Error: {percentile_values[100]:.2f}")
print("----------------------------------------------------------------")

##################################################


# MAKE PLOTS
##################################################

mosaic = [["confusion" + "class", "confusion" + "class", "confusion" + "quality", "normalized" + "quality"],
          ["normalized" + "class", "normalized" + "class", "percentiles", "percentiles"]] if INCLUDE_CONFUSION_MATRIX else [["percentiles"]]
fig, axes = plt.subplot_mosaic(mosaic = mosaic, constrained_layout = True, figsize = (12, 8))
fig.suptitle("Testing the Key Neural Networks")

##################################################


# CONFUSION MATRICES
##################################################

# define helper function
def plot_confusion_matrices(nn_type, mappings, normalized_confusion_matrix_type = "precision"):

    # amount to rotate x axis labels
    rotation_amount = 30 # in degrees

    # plot confusion matrix
    confusion_plot_temp = axes["confusion" + nn_type].imshow(confusion_matrix[nn_type], aspect = "auto", origin = "upper", cmap = "Blues")
    fig.colorbar(confusion_plot_temp, ax = axes["confusion" + nn_type], label = "n", location = "right")
    axes["confusion" + nn_type].set_xlabel("Predicted")
    axes["confusion" + nn_type].set_xticks(ticks = range(len(mappings)), labels = mappings, rotation = rotation_amount, rotation_mode = "anchor", horizontalalignment = "right")
    axes["confusion" + nn_type].set_ylabel("Actual")
    axes["confusion" + nn_type].set_yticks(ticks = range(len(mappings)), labels = mappings)
    axes["confusion" + nn_type].set_title(f"Confusion Matrix")
    # create annotations
    # for row in range(len(mappings)):
    #     for col in range(len(mappings)):
    #         axes["confusion" + nn_type].text(col, row, confusion_matrix[nn_type][row, col].item(), horizontalalignment = "center", verticalalignment = "center", color = "k")

    # plot normalized confusion matrix (either precision or recall)
    normalized_confusion_plot_temp = axes["normalized" + nn_type].imshow(normalized_confusion_matrix[normalized_confusion_matrix_type + nn_type], aspect = "auto", origin = "upper", cmap = "Reds")
    fig.colorbar(normalized_confusion_plot_temp, ax = axes["normalized" + nn_type], label = "", location = "right")
    axes["normalized" + nn_type].set_xlabel("Predicted")
    axes["normalized" + nn_type].set_xticks(ticks = range(len(mappings)), labels = mappings, rotation = rotation_amount, rotation_mode = "anchor", horizontalalignment = "right")
    axes["normalized" + nn_type].set_ylabel("Actual")
    axes["normalized" + nn_type].set_yticks(ticks = range(len(mappings)), labels = mappings)
    axes["normalized" + nn_type].set_title(f"{normalized_confusion_matrix_type.title()}")

    # make specific changes to certain plots
    if nn_type == "class": #  left big plots
        axes["confusion" + nn_type].sharex(other = axes["normalized" + nn_type]) # share the x-axis labels
    elif nn_type == "quality": # top small plots
        axes["normalized" + nn_type].sharey(other = axes["confusion" + nn_type]) # share the y-axis labels

# plot confusion matrix
if INCLUDE_CONFUSION_MATRIX:
    plot_confusion_matrices(nn_type = "class", mappings = KEY_CLASS_MAPPINGS)
    plot_confusion_matrices(nn_type = "quality", mappings = tuple(key_quality + "or" for key_quality in KEY_QUALITY_MAPPINGS)) # ("Maj", "min") -> ("Major", "minor")

##################################################


# MAKE PERCENTILES PLOT
##################################################

axes["percentiles"].plot(percentiles, percentile_values, color = "tab:blue", linestyle = "-")
axes["percentiles"].set_xlabel("Percentile")
axes["percentiles"].set_ylabel("Error")
axes["percentiles"].set_title("Test Key Class Data Percentiles")
axes["percentiles"].grid()

##################################################


# OUTPUT
##################################################

print("Outputting plot...")
fig.savefig(OUTPUT_FILEPATH, dpi = 240) # save image
print(f"Plot saved to {OUTPUT_FILEPATH}.")

##################################################
