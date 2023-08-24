# README
# Phillip Long
# August 21, 2023

# Makes plots describing the training of the neural network.

# python ./training_plots.py history_filepath percentiles_history_filepath output_filepath
# python /dfs7/adl/pnlong/artificial_dj/determine_key/training_plots.py "/dfs7/adl/pnlong/artificial_dj/data/key_nn.pretrained.history.tsv" "/dfs7/adl/pnlong/artificial_dj/data/key_nn.pretrained.percentiles_history.tsv" "/dfs7/adl/pnlong/artificial_dj/data/key_nn.pretrained.png"


# IMPORTS
##################################################
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# sys.argv = ("./training_plots.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pretrained.history.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pretrained.percentiles_history.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pretrained.png")
# sys.argv = ("./training_plots.py", "/dfs7/adl/pnlong/artificial_dj/data/key_nn.pretrained.history.tsv", "/dfs7/adl/pnlong/artificial_dj/data/key_nn.pretrained.percentiles_history.tsv", "/dfs7/adl/pnlong/artificial_dj/data/key_nn.pretrained.png")
##################################################


# IMPORT FILEPATHS, LOAD FILES
##################################################

# make arguments
OUTPUT_FILEPATH_HISTORY = sys.argv[1]
OUTPUT_FILEPATH_PERCENTILES_HISTORY = sys.argv[2]
OUTPUT_FILEPATH = sys.argv[3]

# load in tsv files that have been generated
history = pd.read_csv(OUTPUT_FILEPATH_HISTORY, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
history["epoch"] = range(history.at[0, "epoch"], history.at[0, "epoch"] + len(history)) # make sure epochs are constantly ascending
percentiles_history = pd.read_csv(OUTPUT_FILEPATH_PERCENTILES_HISTORY, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
percentiles_history["epoch"] = [epoch for epoch in range(percentiles_history.at[0, "epoch"], percentiles_history.at[0, "epoch"] + (len(percentiles_history) // 101)) for _ in range(101)] # make sure percentiles epochs are constantly ascending

# get list of epochs where training switches from pretrained layers to final regression layers
freeze_pretrained_epochs = [(history.at[0, "epoch"], history.at[0, "freeze_pretrained"])] + [(history.at[i, "epoch"], history.at[i, "freeze_pretrained"]) for i in range(1, len(history)) if history.at[i, "freeze_pretrained"] != history.at[i - 1, "freeze_pretrained"]] + [(history.at[len(history) - 1, "epoch"], history.at[len(history) - 1, "freeze_pretrained"])]
if freeze_pretrained_epochs[-1] == freeze_pretrained_epochs[-2]: # remove final element if duplicate
    del freeze_pretrained_epochs[-1]
freeze_pretrained_alpha = lambda freeze_pretrained: str(1 - (0.2 * int(not freeze_pretrained))) # get the color (grayscale) for the backgrounds representing freeze_pretrained
freeze_pretrained_epochs_colors = [freeze_pretrained_alpha(freeze_pretrained = epoch[1]) for epoch in freeze_pretrained_epochs[:-1]] # get color indicies from freeze_pretrained values
freeze_pretrained_epochs = [epoch[0] for epoch in freeze_pretrained_epochs] # get epoch values
freeze_pretrained_epochs[-1] += 1 # add one to the last epoch, since it is a repeat of the second to last

##################################################


# CREATE PLOT
##################################################

# plot loss and percentiles per epoch
fig, axes = plt.subplot_mosaic(mosaic = [["loss", "percentiles_history"], ["accuracy", "percentiles_history"]], constrained_layout = True, figsize = (12, 8))
fig.suptitle("Key Neural Network")
colors = ["tab:blue", "tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

# add freeze_pretrained legend
freeze_pretrained_edgecolor = str(0.6 * min(float(freeze_pretrained_alpha(freeze_pretrained = True)), float(freeze_pretrained_alpha(freeze_pretrained = False))))
fig.legend(handles = (mpatches.Patch(facecolor = freeze_pretrained_alpha(freeze_pretrained = True), label = "Pretrained Layers Frozen", edgecolor = freeze_pretrained_edgecolor), mpatches.Patch(facecolor = freeze_pretrained_alpha(freeze_pretrained = False), label = "Pretrained Layers Unfrozen", edgecolor = freeze_pretrained_edgecolor)), loc = "upper left")

##################################################


# PLOT LOSS AND ACCURACY
##################################################

# plot background training type
for i in range(len(freeze_pretrained_epochs) - 1):
    axes["loss"].axvspan(freeze_pretrained_epochs[i], freeze_pretrained_epochs[i + 1], facecolor = freeze_pretrained_epochs_colors[i])
    axes["accuracy"].axvspan(freeze_pretrained_epochs[i], freeze_pretrained_epochs[i + 1], facecolor = freeze_pretrained_epochs_colors[i])

# plot line for every training session
for i in range(len(freeze_pretrained_epochs) - 1):
    # create subset of history
    history_subset = history[(freeze_pretrained_epochs[i] <= history["epoch"]) & (history["epoch"] < freeze_pretrained_epochs[i + 1])].drop(columns = ("freeze_pretrained")).reset_index(drop = True)
    # add point to make the graph more aesthetically pleasing
    if len(history_subset) == 1: # if there is one point, extend straight line
        history_subset.loc[len(history_subset)] = history_subset.loc[len(history_subset) - 1].to_list()
        history_subset.at[len(history_subset) - 1, "epoch"] += 1
    elif len(history_subset) >= 2: # if more than one point, extend line with same slope
        slopes = history_subset.loc[len(history_subset) - 1].subtract(other = history_subset.loc[len(history_subset) - 2], fill_value = 0)
        history_subset.loc[len(history_subset)] = history_subset.loc[len(history_subset) - 1].add(other = slopes, fill_value = 0).to_list()

    # plot values
    for set_type, color in zip(("train", "validate"), colors[:2]):
        axes["loss"].plot(history_subset["epoch"], history_subset[set_type + "_loss"], color = color, linestyle = "solid", label = set_type.title())
        axes["accuracy"].plot(history_subset["epoch"], history_subset[set_type + "_accuracy"], color = color, linestyle = "dashed", label = set_type.title())

# set x-axis labels
axes["accuracy"].set_xlabel("Epoch")
axes["loss"].sharex(other = axes["accuracy"]) # share the x-axis labels

# set y-axis labels
axes["loss"].set_ylabel("Loss")
axes["accuracy"].set_ylabel("Accuracy")

# set legends
handles_by_label_loss = dict(zip(*(axes["loss"].get_legend_handles_labels()[::-1]))) # for removing duplicate legend values
axes["loss"].legend(handles = handles_by_label_loss.values(), labels = handles_by_label_loss.keys(), loc = "upper right")
handles_by_label_accuracy = dict(zip(*(axes["accuracy"].get_legend_handles_labels()[::-1]))) # for removing duplicate legend values
axes["accuracy"].legend(handles = handles_by_label_accuracy.values(), labels = handles_by_label_accuracy.keys(), loc = "upper right")

# set titles
axes["loss"].set_title("Learning Curve")
axes["accuracy"].set_title("Accuracy")

##################################################


# PLOT PERCENTILES PER EPOCH
##################################################

# plot percentiles per epoch (final 5 epochs)
n_epochs = min(5, len(pd.unique(percentiles_history["epoch"])), len(colors)) # get the minimum value so there is no error
colors = colors[:n_epochs] # subet colors to the number of epochs
percentiles_history = percentiles_history[percentiles_history["epoch"] > (max(percentiles_history["epoch"]) - n_epochs)] # get 5 or less most recent epochs
epochs = sorted(pd.unique(percentiles_history["epoch"])) # get the epoch numbers that will be displayed in the plot
for i, epoch in enumerate(epochs):
    percentile_at_epoch = percentiles_history[percentiles_history["epoch"] == epoch]
    axes["percentiles_history"].plot(percentile_at_epoch["percentile"], percentile_at_epoch["value"], color = colors[i], linestyle = "solid", label = epoch)

# set x-axis label
axes["percentiles_history"].set_xlabel("Percentile")

# set y-axis label
axes["percentiles_history"].set_ylabel("Error")

# set legend
axes["percentiles_history"].legend(title = "Epoch", loc = "upper left")

# add gridlines
axes["percentiles_history"].grid()

# set title
axes["percentiles_history"].set_title("Validation Data Percentiles")

##################################################


# SAVE
##################################################

# save figure
fig.savefig(OUTPUT_FILEPATH, dpi = 240) # save image
print(f"Training plot saved to {OUTPUT_FILEPATH}.")

##################################################
