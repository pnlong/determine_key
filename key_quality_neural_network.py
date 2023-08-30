# README
# Phillip Long
# August 23, 2023

# Creates and trains a logistic regression neural network in PyTorch.
# Given an audio file as input, it classifies the sample as Major or minor (see KEY_QUALITY_MAPPINGS in key_dataset.py for more)

# python ./key_quality_neural_network.py labels_filepath nn_filepath freeze_pretrained epochs
# python /Users/philliplong/Desktop/Coding/artificial_dj/determine_key/key_quality_neural_network.py "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_data.tsv" "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_quality_nn.pth"


# IMPORTS
##################################################
import sys
from time import time
from os.path import exists
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
import pandas as pd
from key_dataset import key_quality_dataset # import dataset class
# sys.argv = ("./key_quality_neural_network.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_quality_nn.pth")
# sys.argv = ("./key_quality_neural_network.py", "/dfs7/adl/pnlong/artificial_dj/data/key_data.cluster.tsv", "/dfs7/adl/pnlong/artificial_dj/data/key_quality_nn.pth")
##################################################


# CONSTANTS
##################################################
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
CONFIDENCE_THRESHOLD = 0.5 # threshold to declare 1 or 0
# freeze pretrained parameters (true = freeze pretrained, false = unfreeze pretrained)
try:
    if sys.argv[3].lower().startswith("f"):
        FREEZE_PRETRAINED = False
    else:
        FREEZE_PRETRAINED = True
except (IndexError):
    FREEZE_PRETRAINED = True
# number of epochs to train
try:
    EPOCHS = max(0, int(sys.argv[4])) # in case of a negative number
except (IndexError, ValueError): # in case there is no epochs argument or there is a non-int string
    EPOCHS = 10
##################################################


# NEURAL NETWORK CLASS
##################################################
USE_PRETRAINED = True
class key_quality_nn(torch.nn.Module):

    def __init__(self, nn_filepath, device, freeze_pretrained = None):
        super().__init__()

        if USE_PRETRAINED:
            # initialize pretrained model from pytorch, setting pretrained to True
            self.model = resnet50(weights = ResNet50_Weights.DEFAULT)

            # change the final layer of the model to match my problem, change depending on the transfer learning model being used
            self.model.fc = torch.nn.Sequential(torch.nn.Linear(in_features = 2048, out_features = 1000), torch.nn.ReLU(),
                                                torch.nn.Linear(in_features = 1000, out_features = 500), torch.nn.ReLU(),
                                                torch.nn.Linear(in_features = 500, out_features = 100), torch.nn.ReLU(),
                                                torch.nn.Linear(in_features = 100, out_features = 1), torch.nn.Sigmoid()) # one logistic regression feature

            # try to load previously saved parameters
            if exists(nn_filepath):
                checkpoint = torch.load(nn_filepath, map_location = device)
                self.model.load_state_dict(checkpoint["state_dict"], strict = False)

            # freeze layers according to freeze_pretrained argument, by default all layers require gradient
            for parameter in self.model.parameters(): # unfreeze all layers
                parameter.requires_grad = True
            if freeze_pretrained:
                for parameter in self.model.parameters(): # freeze all layers
                    parameter.requires_grad = False
                for parameter in self.model.fc.parameters(): # unfreeze my layers
                    parameter.requires_grad = True

        else:    
            # convolutional block 1 -> convolutional block 2 -> convolutional block 3 -> convolutional block 4 -> flatten -> linear block 1 -> linear block 2 -> output
            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
            self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
            self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
            self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
            self.flatten = torch.nn.Flatten(start_dim = 1)
            self.linear1 = torch.nn.Sequential(torch.nn.Linear(in_features = 17920, out_features = 1000), torch.nn.ReLU())
            self.linear2 = torch.nn.Sequential(torch.nn.Linear(in_features = 1000, out_features = 100), torch.nn.ReLU())
            self.output = torch.nn.Sequential(torch.nn.Linear(in_features = 100, out_features = 1), torch.nn.Sigmoid())

    def forward(self, input_data):
        if USE_PRETRAINED:
            output = self.model(input_data)
        else:
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
    OUTPUT_PREFIX = ".".join(NN_FILEPATH.split(".")[:-1])
    ##################################################

    # PREPARE TO TRAIN NEURAL NETWORK
    ##################################################

    # determine device
    print("----------------------------------------------------------------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    # instantiate our dataset objects and data loader
    data = {
        "train": key_quality_dataset(labels_filepath = LABELS_FILEPATH, set_type = "train", device = device),
        "validate": key_quality_dataset(labels_filepath = LABELS_FILEPATH, set_type = "validate", device = device)
    }
    data_loader = { # shuffles the batches each epoch to reduce overfitting
        "train": DataLoader(dataset = data["train"], batch_size = BATCH_SIZE, shuffle = True),
        "validate": DataLoader(dataset = data["validate"], batch_size = BATCH_SIZE, shuffle = True)
    }

    # construct model and assign it to device, also summarize 
    key_quality_nn = key_quality_nn(nn_filepath = NN_FILEPATH, device = device, freeze_pretrained = FREEZE_PRETRAINED).to(device)
    if device == "cuda": # some memory usage statistics
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Memory Usage:")
        print(f"  - Allocated: {(torch.cuda.memory_allocated(0)/ (1024 ** 3)):.1f} GB")
        print(f"  - Cached: {(torch.cuda.memory_reserved(0) / (1024 ** 3)):.1f} GB")
    print("================================================================")
    print("Summary of Neural Network:")
    summary(model = key_quality_nn, input_size = data["train"][0][0].shape) # input_size = (# of channels, # of mels [frequency axis], time axis)
    print("================================================================")

    # instantiate loss function and optimizer
    loss_criterion = torch.nn.BCELoss() # make sure loss function agrees with the problem (see https://neptune.ai/blog/pytorch-loss-functions for more), assumes loss function is some sort of mean
    optimizer = torch.optim.Adam(key_quality_nn.parameters()) if USE_PRETRAINED else torch.optim.Adam(key_quality_nn.parameters(), lr = LEARNING_RATE) # if I am not using a pretrained model, I need to specify lr = LEARNING_RATE

    # load previously trained info if applicable
    start_epoch = 0
    if exists(NN_FILEPATH):
        checkpoint = torch.load(NN_FILEPATH, map_location = device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1    

    # STARTING BEST ACCURACY, ADJUST IF NEEDED
    best_accuracy = 0.0 # make sure to adjust for different accuracy metrics

    # history of losses and accuracy
    history_columns = ("epoch", "train_loss", "train_accuracy", "validate_loss", "validate_accuracy", "freeze_pretrained")
    OUTPUT_FILEPATH_HISTORY = OUTPUT_PREFIX + ".history.tsv"
    if not exists(OUTPUT_FILEPATH_HISTORY): # write column names if they are not there yet
        pd.DataFrame(columns = history_columns).to_csv(OUTPUT_FILEPATH_HISTORY, sep = "\t", header = True, index = False, na_rep = "NA", mode = "w") # write column names

    # mark when I started training
    start_time = time()

    ##################################################


    # FUNCTION FOR TRAINING AN EPOCH
    ##################################################
    def train_an_epoch(epoch):

        # TRAIN AN EPOCH
        ##################################################

        # set to training mode
        key_quality_nn.train()

        # instantiate some stats values
        history_epoch = dict(zip(history_columns, (0.0,) * len(history_columns))) # in the case of linear regression, accuracy is actually the average absolute error
        history_epoch["epoch"] = epoch + 1
        history_epoch["freeze_pretrained"] = FREEZE_PRETRAINED
        start_time_epoch = time()

        # training loop
        for inputs, labels in tqdm(data_loader["train"], desc = "Training"):

            # register inputs and labels with device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # clear existing gradients
            optimizer.zero_grad()

            # forward pass: compute predictions on input data using the model
            predictions = key_quality_nn(inputs)

            # compute loss
            loss_batch = loss_criterion(predictions, labels)

            # backpropagate the gradients
            loss_batch.backward()

            # update the parameters
            optimizer.step()

            # compute the total loss for the batch and add it to history_epoch["train_loss"]
            history_epoch["train_loss"] += loss_batch.item() * inputs.size(0) # inputs.size(0) is the number of inputs in the current batch, assumes loss is an average over the batch
            
            # compute the accuracy
            predictions = (predictions >= CONFIDENCE_THRESHOLD) # convert into booleans

            # compute the total accuracy for the batch and add it to history_epoch["train_accuracy"]
            history_epoch["train_accuracy"] += torch.sum(input = (predictions.view(-1) == labels.view(-1))).item()
        
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
            key_quality_nn.eval()

            # validation loop
            for inputs, labels in tqdm(data_loader["validate"], desc = "Validating"):

                # register inputs and labels with device
                inputs, labels = inputs.to(device), labels.to(device)

                # forward pass: compute predictions on input data using the model
                predictions = key_quality_nn(inputs)

                # compute loss
                loss_batch = loss_criterion(predictions, labels)

                # compute the total loss for the batch and add it to history_epoch["validate_loss"]
                history_epoch["validate_loss"] += loss_batch.item() * inputs.size(0) # inputs.size(0) is the number of inputs in the current batch, assumes loss is an average over the batch
            
                # compute the accuracy
                predictions = (predictions >= CONFIDENCE_THRESHOLD) # convert into booleans

                # compute the total accuracy for the batch and add it to history_epoch["validate_accuracy"]
                history_epoch["validate_accuracy"] += torch.sum(input = (predictions.view(-1) == labels.view(-1))).item()

        ##################################################


        # OUTPUT SUMMARY STATISTICS
        ##################################################

        # compute average losses and accuracies
        history_epoch["train_loss"] /= len(data["train"])
        history_epoch["train_accuracy"] /= len(data["train"])
        history_epoch["validate_loss"] /= len(data["validate"])
        history_epoch["validate_accuracy"] /= len(data["validate"])
        # store average losses and accuracies in history
        pd.DataFrame(data = [history_epoch], columns = history_columns).to_csv(OUTPUT_FILEPATH_HISTORY, sep = "\t", header = False, index = False, na_rep = "NA", mode = "a") # write to file

        # save current model if its validation accuracy is the best so far
        global best_accuracy
        if history_epoch["validate_accuracy"] >= best_accuracy:
            best_accuracy = history_epoch["validate_accuracy"] # update best_accuracy
            checkpoint = {
                "epoch": epoch,
                "state_dict": key_quality_nn.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, NN_FILEPATH)

        # print out updates
        print(f"Training Time: {(total_time_epoch / 60):.1f} minutes")
        print(f"Training Loss: {history_epoch['train_loss']:.3f}, Validation Loss: {history_epoch['validate_loss']:.3f}")
        print(f"Training Accuracy: {100 * history_epoch['train_accuracy']:.3f}%, Validation Accuracy: {100 * history_epoch['validate_accuracy']:.3f}%")

        ##################################################


    ##################################################


    # TRAIN EPOCHS
    ##################################################

    # helper function for training 
    def train_epochs(start, n): # start = epoch to start training on; n = number of epochs to train from there
        
        # print what section is being trained
        if FREEZE_PRETRAINED:
            print("Training final regression layer...")
        elif not FREEZE_PRETRAINED:
            print("Fine-tuning pretrained layers...")
        else:
            print("Training all layers...")

        # epochs loop
        epochs_to_train = range(start, start + n)
        for epoch in epochs_to_train:
            print("----------------------------------------------------------------")
            print(f"EPOCH {epoch + 1} / {epochs_to_train.stop}")
            train_an_epoch(epoch = epoch)
        print("================================================================")
    
    # train epochs
    train_epochs(start = start_epoch, n = EPOCHS)

    ##################################################


    # PRINT TRAINING STATISTICS
    ##################################################

    # mark when training ended, calculate total time
    end_time = time()
    total_time = end_time - start_time
    del end_time, start_time

    # print training statistics
    print("Training is done.")
    print(f"Time Elapsed: {total_time // (60 * 60):.0f} hours and {(total_time % (60 * 60)) / 60:.1f} minutes")
    
    ##################################################
