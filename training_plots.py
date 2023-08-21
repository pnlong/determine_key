# README
# Phillip Long
# August 21, 2023

# Makes plots describing the training of the neural network.

# python ./training_plots.py history_filepath percentiles_history_filepath output_filepath


# IMPORTS
##################################################
import sys
import pandas as pd
import matplotlib.pyplot as plt
# sys.argv = ("./training_plots.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pretrained.history.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pretrained.percentiles_history.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pretrained.png")
##################################################


# IMPORT FILEPATHS, LOAD FILES
##################################################

# make arguments
OUTPUT_FILEPATH_HISTORY = sys.argv[1]
OUTPUT_FILEPATH_PERCENTILES_HISTORY = sys.argv[2]
OUTPUT_FILEPATH = sys.argv[3]

# load in tsv files that have been generated
history = pd.read_csv(OUTPUT_FILEPATH_HISTORY, sep = "\t", header = 0, index_col = False)
percentiles_history = pd.read_csv(OUTPUT_FILEPATH_PERCENTILES_HISTORY, sep = "\t", header = 0, index_col = False)

##################################################


# CREATE PLOT
##################################################

# plot loss and percentiles per epoch
fig, (loss_plot, percentiles_history_plot) = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 7))
fig.suptitle("Tempo Neural Network")
colors = ["b", "r", "g", "c", "m", "y", "k"]

##################################################


# PLOT LOSS AND ACCURACY
##################################################

loss_plot.set_xlabel("Epoch")

# left side is loss per epoch, in blue
color_loss = colors[0]
loss_plot.set_ylabel("Loss", color = color_loss)
for set_type, ls in zip(("train_loss", "validate_loss"), ("solid", "dashed")):
    loss_plot.plot(history["epoch"], history[set_type], color = color_loss, linestyle = ls, label = set_type.split("_")[0].title())
loss_plot.tick_params(axis = "y", labelcolor = color_loss)
loss_plot.legend(title = "Loss", loc = "upper left")

# right side is accuracy per epoch, in red
loss_plot_accuracy = loss_plot.twinx()
color_accuracy = colors[1]
loss_plot_accuracy.set_ylabel("Average Error", color = color_accuracy)
for set_type, ls in zip(("train_accuracy", "validate_accuracy"), ("solid", "dashed")):
    loss_plot_accuracy.plot(history["epoch"], history[set_type], color = color_accuracy, linestyle = ls, label = set_type.split("_")[0].title())
loss_plot_accuracy.tick_params(axis = "y", labelcolor = color_accuracy)
loss_plot_accuracy.legend(title = "Error", loc = "upper right")
loss_plot.set_title("Learning Curve & Average Error")

##################################################


# PLOT PERCENTILES PER EPOCH
##################################################

# plot percentiles per epoch (final 5 epochs)
epochs = sorted(pd.unique(percentiles_history["epoch"]))
n_epochs = min(5, len(epochs), len(colors))
colors = colors[:n_epochs]
percentiles_history = percentiles_history[percentiles_history["epoch"] > (max(percentiles_history["epoch"] - n_epochs))]
for i, epoch in enumerate(epochs):
    percentile_at_epoch = percentiles_history[percentiles_history["epoch"] == epoch]
    percentiles_history_plot.plot(percentile_at_epoch["percentile"], percentile_at_epoch["value"], color = colors[i], linestyle = "solid", label = epoch)
percentiles_history_plot.set_xlabel("Percentile")
percentiles_history_plot.set_ylabel("Error")
percentiles_history_plot.legend(title = "Epoch", loc = "upper left")
percentiles_history_plot.grid()
percentiles_history_plot.set_title("Validation Data Percentiles")

##################################################


# SAVE
##################################################

# save figure
fig.tight_layout()
fig.savefig(OUTPUT_FILEPATH, dpi = 180) # save image

##################################################
