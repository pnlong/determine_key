# README
# Phillip Long
# August 13, 2023

# Creates and trains a linear regression neural network in PyTorch.
# Given an audio file as input, it classifies the sample as one of 24 classes (see KEY_MAPPINGS in key_dataset.py for more)

# python ./tempo_neural_network.py labels_filepath nn_filepath epochs
# python /Users/philliplong/Desktop/Coding/artificial_dj/determine_tempo/tempo_neural_network.py "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv" "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth" "1"


# IMPORTS
##################################################
import sys
from time import time
from os.path import exists
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
import pandas as pd
import matplotlib.pyplot as plt
from numpy import percentile
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
class tempo_nn(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # convolutional block 1 -> convolutional block 2 -> convolutional block 3 -> convolutional block 4 -> flatten -> linear 1 -> linear 2 -> output
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        self.flatten = torch.nn.Flatten(start_dim = 1)
        self.linear1 = torch.nn.Linear(in_features = 17920, out_features = 100)
        self.linear2 = torch.nn.Linear(in_features = 100, out_features = 10)
        self.output = torch.nn.Linear(in_features = 10, out_features = 1)

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


if __name__ == "__main__":

    # CONSTANTS
    ##################################################
    LABELS_FILEPATH = sys.argv[1]
    NN_FILEPATH = sys.argv[2]
    ##################################################

    # PREPARE TO TRAIN NEURAL NETWORK
    ##################################################

    # determine device
    print("----------------------------------------------------------------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    # instantiate our dataset objects and data loader
    data = {
        "train": tempo_dataset(labels_filepath = LABELS_FILEPATH, set_type = "train", device = device),
        "validate": tempo_dataset(labels_filepath = LABELS_FILEPATH, set_type = "validate", device = device)
    }
    data_loader = { # shuffles the batches each epoch to reduce overfitting
        "train": DataLoader(dataset = data["train"], batch_size = BATCH_SIZE, shuffle = True),
        "validate": DataLoader(dataset = data["validate"], batch_size = BATCH_SIZE, shuffle = True)
    }

    # construct model and assign it to device, also summarize 
    tempo_nn = tempo_nn().to(device)
    if device == "cuda": # some memory usage statistics
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Memory Usage:")
        print(f"  - Allocated: {(torch.cuda.memory_allocated(0)/ (1024 ** 3)):.1f} GB")
        print(f"  - Cached: {(torch.cuda.memory_reserved(0) / (1024 ** 3)):.1f} GB")
    print("================================================================")
    print("Summary of Neural Network:")
    summary(model = tempo_nn, input_size = data["train"][0][0].shape) # input_size = (# of channels, # of mels [frequency axis], time axis)
    print("================================================================")

    # instantiate loss function and optimizer
    loss_criterion = torch.nn.MSELoss() # make sure loss function agrees with the problem (see https://neptune.ai/blog/pytorch-loss-functions for more), assumes loss function is some sort of mean
    optimizer = torch.optim.Adam(tempo_nn.parameters()) # if I am not using a pretrained model, I need to specify lr = LEARNING_RATE

    # load previously trained info if applicable
    start_epoch = 0
    if exists(NN_FILEPATH):
        checkpoint = torch.load(NN_FILEPATH, map_location = device)
        tempo_nn.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1    

    # tracking statistics
    epochs_to_train = range(start_epoch, start_epoch + EPOCHS)
    history = {
        "train_loss": [0.0,] * len(epochs_to_train),
        "train_accuracy": [0.0,] * len(epochs_to_train),
        "validate_loss": [0.0,] * len(epochs_to_train),
        "validate_accuracy": [0.0,] * len(epochs_to_train)
        }
    best_accuracy = 1e+24 # make sure to adjust for different accuracy metrics
    percentiles_history = pd.DataFrame(columns = ("epoch", "percentile", "value"))

    # mark when I started training
    start_time = time()

    ##################################################


    # EPOCH FOR LOOP
    for i, epoch in enumerate(epochs_to_train):

        print(f"EPOCH {epoch + 1} / {epochs_to_train.stop}")

        # TRAIN AN EPOCH
        ##################################################

        # set to training mode
        tempo_nn.train()

        # instantiate some stats values
        loss = {"train": 0.0, "validate": 0.0}
        accuracy = {"train": 0.0, "validate": 0.0} # in the case of linear regression, accuracy is actually the average absolute error
        start_time_epoch = time()

        # training loop
        for inputs, labels in data_loader["train"]:

            # register inputs and labels with device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # clear existing gradients
            optimizer.zero_grad()

            # forward pass: compute predictions on input data using the model
            predictions = tempo_nn(inputs)

            # compute loss
            loss_batch = loss_criterion(predictions, labels)

            # backpropagate the gradients
            loss_batch.backward()

            # update the parameters
            optimizer.step()

            # compute the total loss for the batch and add it to loss["train"]
            loss["train"] += loss_batch.item() * inputs.size(0) # inputs.size(0) is the number of inputs in the current batch
            
            # compute the accuracy
            accuracy_batch = torch.abs(input = predictions.view(-1) - labels.view(-1))

            # compute the total accuracy for the batch and add it to accuracy["train"]
            accuracy["train"] += torch.sum(input = accuracy_batch).item()
        
        # for calculating time statistics
        end_time_epoch = time()
        total_time_epoch = end_time_epoch - start_time_epoch
        del end_time_epoch, start_time_epoch

        ##################################################


        # VALIDATE MODEL
        ##################################################

        # no gradient tracking needed
        with torch.no_grad():
            
            # set to evaluation mode
            tempo_nn.eval()

            # validation loop
            error_validate = torch.Tensor()
            for inputs, labels in data_loader["validate"]:

                # register inputs and labels with device
                inputs, labels = inputs.to(device), labels.to(device)

                # forward pass: compute predictions on input data using the model
                predictions = tempo_nn(inputs)

                # compute loss
                loss_batch = loss_criterion(predictions, labels)

                # compute the total loss for the batch and add it to loss["validate"]
                loss["validate"] += loss_batch.item() * inputs.size(0) # inputs.size(0) is the number of inputs in the current batch
            
                # compute the accuracy
                accuracy_batch = torch.abs(input = predictions.view(-1) - labels.view(-1))

                # compute the total accuracy for the batch and add it to accuracy["validate"]
                accuracy["validate"] += torch.sum(input = accuracy_batch).item()

                # add accuracy to running count of all the errors in the validation dataset
                error_validate = torch.cat(tensors = (error_validate, accuracy_batch), dim = 0)

        ##################################################


        # OUTPUT SUMMARY STATISTICS
        ##################################################

        # compute average losses and accuracies
        loss["train"] /= len(data["train"])
        accuracy["train"] /= len(data["train"])
        loss["validate"] /= len(data["validate"])
        accuracy["validate"] /= len(data["validate"])
        history["train_loss"][i], history["train_accuracy"][i], history["validate_loss"][i], history["validate_accuracy"][i] = loss["train"], accuracy["train"], loss["validate"], accuracy["validate"] # store average losses and accuracies in history

        # calculate percentiles
        percentiles = range(0, 101)
        percentile_values = percentile(error_validate, q = percentiles)
        percentiles_history = pd.concat([percentiles_history, pd.DataFrame(data = {"epoch": [epoch + 1,] * len(percentiles), "percentile": percentiles, "value": percentile_values})])

        # save current model if its validation accuracy is the best so far
        if accuracy["validate"] <= best_accuracy:
            best_accuracy = accuracy["validate"] # update best_accuracy
            checkpoint = {
                "epoch": epoch,
                "state_dict": tempo_nn.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, NN_FILEPATH)

        # print out updates
        print(f"Training Time: {(total_time_epoch / 60):.1f} minutes")
        print(f"Training Loss: {loss['train']:.3f}, Validation Loss: {loss['validate']:.3f}")
        print(f"Mean Training Error: {accuracy['train']:.3f}, Mean Validation Error: {accuracy['validate']:.3f}")
        print(f"Five Number Summary of Validation Errors: {' '.join((f'{value:.2f}' for value in (percentile_values[percentile] for percentile in (0, 25, 50, 75, 100))))}")
        print("----------------------------------------------------------------")

        ##################################################


    # MAKE TRAINING PLOTS TO SHOWCASE TRAINING OF MODEL
    ##################################################

    # plot loss and percentiles per epoch
    fig, (loss_plot, percentiles_history_plot) = plt.subplots(nrows = 1, ncols = 2, figsize = (2, 1))
    fig.suptitle("Tempo Neural Network")
    colors = ["b", "r", "g", "c", "m", "y", "k"]

    # plot loss as a function of epoch
    epochs_to_train = [epoch + 1 for epoch in epochs_to_train]
    loss_plot.set_xlabel("Epoch")
    # left side is loss per epoch, in blue
    color_loss = colors[0]
    loss_plot.set_ylabel("Loss", color = color_loss)
    for set_type, ls in zip(("train_loss", "validate_loss"), ("solid", "dashed")):
        loss_plot.plot(epochs_to_train, history[set_type], color = color_loss, linestyle = ls, label = set_type.split("_")[0].title())
    loss_plot.tick_params(axis = "y", labelcolor = color_loss)
    loss_plot.legend(title = "Loss", loc = "upper left")
    # right side is accuracy per epoch, in red
    loss_plot_accuracy = loss_plot.twinx()
    color_accuracy = colors[1]
    loss_plot_accuracy.set_ylabel("Average Error", color = color_accuracy)
    for set_type, ls in zip(("train_accuracy", "validate_accuracy"), ("solid", "dashed")):
        loss_plot_accuracy.plot(epochs_to_train, history[set_type], color = color_accuracy, linestyle = ls, label = set_type.split("_")[0].title())
    loss_plot_accuracy.tick_params(axis = "y", labelcolor = color_accuracy)
    loss_plot_accuracy.legend(title = "Error", loc = "upper right")
    loss_plot.set_title("Learning Curve & Average Error")

    # plot percentiles per epoch (final 5 epochs)
    n_epochs = min(EPOCHS, 5, len(colors))
    colors = colors[:n_epochs]
    percentiles_history = percentiles_history[percentiles_history["epoch"] > (max(percentiles_history["epoch"] - n_epochs))]
    for i, epoch in enumerate(sorted(pd.unique(percentiles_history["epoch"]))):
        percentile_at_epoch = percentiles_history[percentiles_history["epoch"] == epoch]
        percentiles_history_plot.plot(percentile_at_epoch["percentile"], percentile_at_epoch["value"], color = colors[i], linestyle = "-", label = epoch)
    percentiles_history_plot.set_xlabel("Percentile")
    percentiles_history_plot.set_ylabel("Error")
    percentiles_history_plot.legend(title = "Epoch", loc = "upper left")
    percentiles_history_plot.grid()
    percentiles_history_plot.set_title("Validation Data Percentiles")

    # save figure
    fig.savefig(NN_FILEPATH.split(".")[0] + ".png", dpi = 180) # save image

    ##################################################
    
    # PRINT TRAINING STATISTICS
    ##################################################

    # mark when training ended, calculate total time
    end_time = time()
    total_time = end_time - start_time
    del end_time, start_time

    # print training statistics
    print("================================================================")
    print("Training is done.")
    print(f"Time Elapsed: {total_time // (60 * 60):.0f} hours and {(total_time % (60 * 60)) / 60:.1f} minutes")
    
    ##################################################
