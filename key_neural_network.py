# README
# Phillip Long
# August 13, 2023

# Creates and trains a linear regression neural network in PyTorch.
# Given an audio file as input, it classifies the sample as one of 24 classes (see KEY_MAPPINGS in key_dataset.py for more)

# python ./tempo_neural_network.py labels_filepath nn_filepath epochs


# IMPORTS
##################################################
import sys
from time import time
from os.path import exists
import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from numpy import percentile, mean
from tempo_dataset import tempo_dataset # import dataset class
# sys.argv = ("./tempo_neural_network.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth")
# sys.argv = ("./tempo_neural_network.py", "/dfs7/adl/pnlong/artificial_dj/data/tempo_data.cluster.tsv", "/dfs7/adl/pnlong/artificial_dj/data/tempo_nn.pth", "")
##################################################


# CONSTANTS
##################################################
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
try:
    EPOCHS = max(0, int(sys.argv[3])) # in case of a negative number
except (IndexError, ValueError): # in case there is no epochs argument or there is a non-int string
    EPOCHS = 10
##################################################


# NEURAL NETWORK CLASS
##################################################
class tempo_nn(nn.Module):

    def __init__(self):
        super().__init__()
        # convolutional block 1 -> convolutional block 2 -> convolutional block 3 -> convolutional block 4 -> flatten -> linear 1 -> linear 2 -> output
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 2), nn.ReLU(), nn.MaxPool2d(kernel_size = 2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 2), nn.ReLU(), nn.MaxPool2d(kernel_size = 2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2), nn.ReLU(), nn.MaxPool2d(kernel_size = 2))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 2), nn.ReLU(), nn.MaxPool2d(kernel_size = 2))
        self.flatten = nn.Flatten(start_dim = 1)
        self.linear1 = nn.Linear(in_features = 17920, out_features = 100)
        self.linear2 = nn.Linear(in_features = 100, out_features = 10)
        self.output = nn.Linear(in_features = 10, out_features = 1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        output = self.output(x)
        return output

##################################################


# MODEL TRAINING FUNCTION
##################################################
# train the whole model
def train(model, dataset, optimizer, device, start_epoch):

    # instantiate loss function and data loader
    loss_function = nn.MSELoss() # make sure loss function agrees with the problem (see https://neptune.ai/blog/pytorch-loss-functions for more), assumes loss function is some sort of mean
    data_loader = DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True) # shuffles the batches each epoch to reduce overfitting

    # tracking statistics
    epochs_to_train = range(start_epoch, start_epoch + EPOCHS)
    losses = []
    percentiles_per_epoch = pd.DataFrame(columns = ("epoch", "percentile", "value"))

    # epoch for loop
    for epoch in epochs_to_train:

        loss_per_epoch = 0
        start_time_epoch = time()

        # train an epoch
        i = 0
        for inputs, labels in data_loader:
            # register inputs and labels with device
            inputs, labels = inputs.to(device), labels.to(device)

            # calculate loss
            predictions = model(inputs)
            loss = loss_function(predictions, labels)

            # backpropagate loss and update weights
            optimizer.zero_grad() # zero the gradients
            loss.backward() # conduct backpropagation
            optimizer.step() # update parameters
            loss_per_epoch += (BATCH_SIZE * loss.item()) if (i < (len(dataset) // BATCH_SIZE)) else ((len(dataset) % BATCH_SIZE) * loss.item())
            i += 1
        # print(f"i = {i}, {((i - 1) * BATCH_SIZE) + (len(dataset) % BATCH_SIZE)} == {len(dataset)}") # for debugging
        del i
        
        # calculate statistics
        end_time_epoch = time()

        # calculate "accuracy" statistic
        n_predictions = min(100, len(dataset)) # number of samples to use in statistic
        inputs, targets = dataset.sample(n_predictions = n_predictions) # get a sample of n_predictions rows from the dataset
        model.eval() # turn on eval mode
        with torch.no_grad():
            predictions = model(inputs).view(n_predictions, 1) # make prdictions, reshape to match the targets tensor from dataset.sample
        model.train() # turn off eval mode, back to train mode
        error = torch.abs(input = predictions - targets).numpy(force = True) # force error calculations into numpy array on CPU
        percentiles = range(0, 101)
        percentile_values = percentile(error, q = percentiles)
        percentiles_per_epoch = pd.concat([percentiles_per_epoch, pd.DataFrame(data = {"epoch": [epoch + 1,] * len(percentiles), "percentile": percentiles, "value": percentile_values})])
        del inputs, predictions, targets, percentiles

        # calculate loss per epoch, update losses list
        loss_per_epoch = loss_per_epoch / len(dataset)
        losses.append(loss_per_epoch)

        # calculate time elapsed
        total_time_epoch = end_time_epoch - start_time_epoch
        del end_time_epoch, start_time_epoch

        # save current model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, NN_FILEPATH)

        # print out updates
        print(f"EPOCH {epoch + 1}")
        print(f"Loss: {loss_per_epoch:.3f}")
        print(f"Average Error: {mean(error):.3f}")
        print(f"Five Number Summary: {' '.join((f'{i:.2f}' for i in (percentile_values[j] for j in (0, 25, 50, 75, 100))))}")
        print(f"Time: {(total_time_epoch / 60):.1f} minutes")
        del loss_per_epoch, error, percentile_values, total_time_epoch
        print("----------------------------------------------------------------")

    # plot loss and percentiles per epoch
    fig, (loss_plot, percentiles_per_epoch_plot) = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 7))
    fig.suptitle("Tempo Neural Network")

    # plot loss as a function of epoch
    loss_plot.plot([epoch + 1 for epoch in epochs_to_train], losses, "-b")
    loss_plot.set_xlabel("Epoch")
    loss_plot.set_ylabel("Loss")
    loss_plot.set_title("Learning Curve")

    # plot percentiles plot over each epoch (final 3 epochs)
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    n_epochs = min(EPOCHS, 5, len(colors))
    colors = colors[:n_epochs]
    percentiles_per_epoch = percentiles_per_epoch[percentiles_per_epoch["epoch"] > (max(percentiles_per_epoch["epoch"] - n_epochs))]
    for i, epoch in enumerate(sorted(pd.unique(percentiles_per_epoch["epoch"]))):
        percentile_at_epoch = percentiles_per_epoch[percentiles_per_epoch["epoch"] == epoch]
        percentiles_per_epoch_plot.plot(percentile_at_epoch["percentile"], percentile_at_epoch["value"], "-" + colors[i], label = epoch)
        del percentile_at_epoch
    percentiles_per_epoch_plot.set_xlabel("Percentile")
    percentiles_per_epoch_plot.set_ylabel("Difference")
    percentiles_per_epoch_plot.legend(title = "Epoch", loc = "upper left")
    percentiles_per_epoch_plot.set_title("Train Data Percentiles")

    # save figure
    fig.savefig(NN_FILEPATH.split(".")[0] + ".png", dpi = 180) # save image

    del losses, percentiles_per_epoch, loss_plot, percentiles_per_epoch_plot, n_epochs, colors
    
##################################################


if __name__ == "__main__":

    # CONSTANTS
    ##################################################
    LABELS_FILEPATH = sys.argv[1]
    NN_FILEPATH = sys.argv[2]
    ##################################################

    # TRAIN NEURAL NETWORK
    ##################################################

    # determine device
    print("----------------------------------------------------------------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    # instantiate our dataset object
    tempo_data = tempo_dataset(labels_filepath = LABELS_FILEPATH, set_type = "train", device = device)

    # construct model and assign it to device, also summarize 
    tempo_nn = tempo_nn().to(device)
    if device == "cuda": # some memory usage statistics
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Memory Usage:")
        print(f"  - Allocated: {(torch.cuda.memory_allocated(0)/ (1024 ** 3)):.1f} GB")
        print(f"  - Cached: {(torch.cuda.memory_reserved(0) / (1024 ** 3)):.1f} GB")
    print("================================================================")
    print("Summary of Neural Network:")
    summary(model = tempo_nn, input_size = tempo_data[0][0].shape) # input_size = (# of channels, # of mels [frequency axis], time axis)
    print("================================================================")

    # instantiate optimizer
    optimizer = torch.optim.Adam(tempo_nn.parameters(), lr = LEARNING_RATE)

    # load previously trained info if applicable
    start_epoch = 0
    if exists(NN_FILEPATH):
        checkpoint = torch.load(NN_FILEPATH, map_location = device)
        tempo_nn.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    # train
    start_time = time()
    train(model = tempo_nn, dataset = tempo_data, optimizer = optimizer, device = device, start_epoch = start_epoch)
    end_time = time()
    print("================================================================")
    print("Training is done.")
    total_time = end_time - start_time
    del end_time, start_time
    print(f"Time Elapsed: {total_time // (60 * 60):.0f} hours and {(total_time % (60 * 60)) / 60:.1f} minutes")
    
    ##################################################
