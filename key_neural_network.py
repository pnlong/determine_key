# README
# Phillip Long
# August 1, 2023

# Creates and trains a linear regression neural network in PyTorch.
# Given an audio file as input, it outputs a single number representing the song's tempo in Beats per Minute (BPM).

# python ./tempo_neural_network.py labels_filepath nn_filepath


# IMPORTS
##################################################
import sys
from tqdm import tqdm
from time import time
from os.path import exists, join, dirname
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchaudio
from tempo_dataset import tempo_dataset, SAMPLE_RATE, SAMPLE_DURATION # dataset class + some constants
# sys.argv = ("./tempo_neural_network.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_nn.pth")
##################################################


# CONSTANTS
##################################################
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
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
def train(model, data_loader, loss_function, optimizer, device, start_epoch):

    epochs_to_train = range(start_epoch, start_epoch + EPOCHS)
    losses = []
    for epoch in epochs_to_train: # epoch for loop

        loss_per_epoch = 0

        # train an epoch
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
            loss_per_epoch += loss.item()

        # print out updates
        print(f"EPOCH {epoch + 1}")
        print(f"Loss: {loss_per_epoch:.5f}")
        print("----------------------------------------------------------------")

        losses.append(loss_per_epoch)

        # save current model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, NN_FILEPATH)


    # plot loss as a function of n iterations
    plt.plot([epoch + 1 for epoch in epochs_to_train], losses, "-b")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.savefig(join(dirname(NN_FILEPATH), "loss.png"))  # save image
    

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
    tempo_data = tempo_dataset(labels_filepath = LABELS_FILEPATH,
                               set_type = "train",
                               target_sample_rate = SAMPLE_RATE,
                               sample_duration = SAMPLE_DURATION,
                               device = device,
                               transformation = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64)
                               )

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

    # instantiate data loader, loss function, and optimizer
    data_loader = DataLoader(tempo_data, batch_size = BATCH_SIZE)
    loss_function = nn.MSELoss() # make sure loss function agrees with the problem (see https://neptune.ai/blog/pytorch-loss-functions for more)
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
    train(model = tempo_nn, data_loader = data_loader, loss_function = loss_function, optimizer = optimizer, device = device, start_epoch = start_epoch)
    end_time = time()
    print("================================================================")
    print("Training is done.")
    total_time = end_time - start_time
    del end_time, start_time
    print(f"Time Elapsed: {total_time // (60 * 60):.0f} hours and {(total_time % (60 * 60)) / 60:.1f} minutes")
    
    ##################################################