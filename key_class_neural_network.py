# README
# Phillip Long
# August 23, 2023

# Creates and trains a neural network in PyTorch.
# Given an audio file as input, it classifies the sample as one of 12 relative-key (see KEY_CLASS_MAPPINGS in key_dataset.py for more)

# python ./key_neural_network.py labels_filepath nn_filepath freeze_pretrained epochs
# python /Users/philliplong/Desktop/Coding/artificial_dj/determine_key/key_neural_network.py "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_data.tsv" "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pth"


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
from numpy import percentile
from key_dataset import key_dataset, KEY_MAPPINGS # import dataset class
# sys.argv = ("./key_neural_network.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_nn.pth")
# sys.argv = ("./key_neural_network.py", "/dfs7/adl/pnlong/artificial_dj/data/key_data.cluster.tsv", "/dfs7/adl/pnlong/artificial_dj/data/key_nn.pth")
##################################################


# CONSTANTS
##################################################
BATCH_SIZE = 32
# LEARNING_RATE = 1e-3
# freeze pretrained parameters (true = freeze pretrained, false = unfreeze pretrained, freeze my parameters)
try:
    if sys.argv[3].lower().startswith("t"):
        FREEZE_PRETRAINED = True
    elif sys.argv[3].lower().startswith("f"):
        FREEZE_PRETRAINED = False
    else:
        FREEZE_PRETRAINED = None
except (IndexError):
    FREEZE_PRETRAINED = None
# number of epochs to train
try:
    EPOCHS = max(0, int(sys.argv[4])) # in case of a negative number
except (IndexError, ValueError): # in case there is no epochs argument or there is a non-int string
    EPOCHS = 10
##################################################


# NEURAL NETWORK CLASS
##################################################
class key_nn(torch.nn.Module):

    def __init__(self, nn_filepath, device, freeze_pretrained = None):
        super().__init__()

        # initialize pretrained model from pytorch, setting pretrained to True
        self.model = resnet50(weights = ResNet50_Weights.DEFAULT)

        # change the final layer of the model to match my problem, change depending on the transfer learning model being used
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(in_features = 2048, out_features = 1000),
                                            torch.nn.Linear(in_features = 1000, out_features = len(KEY_MAPPINGS)) # one feature per key
                                            )

        # try to load previously saved parameters
        if exists(nn_filepath):
            checkpoint = torch.load(nn_filepath, map_location = device)
            self.model.load_state_dict(checkpoint["state_dict"], strict = False)

        # freeze layers according to freeze_pretrained argument, by default all layers require gradient
        for parameter in self.model.parameters(): # unfreeze all layers
                parameter.requires_grad = True
        if freeze_pretrained is not None:
            # freeze_pretrained == False
            for parameter in self.model.fc.parameters(): # create the distinction between my layers and the pretrained layers by freezing my layers (freeze_pretrained == False)
                parameter.requires_grad = False
            # freeze_pretrained == True
            if freeze_pretrained: # if (freeze_pretrained == True), switch all values such that the pretrained layers do not requires_grad and the output regression layer does
                for parameter in self.model.parameters():
                    parameter.requires_grad = not (parameter.requires_grad)
        
        # convolutional block 1 -> convolutional block 2 -> convolutional block 3 -> convolutional block 4 -> flatten -> linear 1 -> linear 2 -> output
        # self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        # self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        # self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        # self.conv4 = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 2), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size = 2))
        # self.flatten = torch.nn.Flatten(start_dim = 1)
        # self.linear1 = torch.nn.Linear(in_features = 17920, out_features = 100)
        # self.linear2 = torch.nn.Linear(in_features = 100, out_features = 10)
        # self.output = torch.nn.Linear(in_features = 10, out_features = 1)

    def forward(self, input_data):

        output = self.model(input_data)
        return output

        # x = self.conv1(input_data)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        # output = self.output(x)
        # return output

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
        "train": key_dataset(labels_filepath = LABELS_FILEPATH, set_type = "train", device = device),
        "validate": key_dataset(labels_filepath = LABELS_FILEPATH, set_type = "validate", device = device)
    }
    data_loader = { # shuffles the batches each epoch to reduce overfitting
        "train": DataLoader(dataset = data["train"], batch_size = BATCH_SIZE, shuffle = True),
        "validate": DataLoader(dataset = data["validate"], batch_size = BATCH_SIZE, shuffle = True)
    }

    # construct model and assign it to device, also summarize 
    key_nn = key_nn(nn_filepath = NN_FILEPATH, device = device, freeze_pretrained = FREEZE_PRETRAINED).to(device)
    if device == "cuda": # some memory usage statistics
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print("Memory Usage:")
        print(f"  - Allocated: {(torch.cuda.memory_allocated(0)/ (1024 ** 3)):.1f} GB")
        print(f"  - Cached: {(torch.cuda.memory_reserved(0) / (1024 ** 3)):.1f} GB")
    print("================================================================")
    print("Summary of Neural Network:")
    summary(model = key_nn, input_size = data["train"][0][0].shape) # input_size = (# of channels, # of mels [frequency axis], time axis)
    print("================================================================")

    # instantiate loss function and optimizer
    loss_criterion = torch.nn.MSELoss() # make sure loss function agrees with the problem (see https://neptune.ai/blog/pytorch-loss-functions for more), assumes loss function is some sort of mean
    optimizer = torch.optim.Adam(key_nn.parameters()) # if I am not using a pretrained model, I need to specify lr = LEARNING_RATE

    # load previously trained info if applicable
    start_epoch = 0
    if exists(NN_FILEPATH):
        checkpoint = torch.load(NN_FILEPATH, map_location = device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1    

    # STARTING BEST ACCURACY, ADJUST IF NEEDED
    best_accuracy = 1e+24 # make sure to adjust for different accuracy metrics

    # history of losses and accuracy
    history_columns = ("epoch", "train_loss", "train_accuracy", "validate_loss", "validate_accuracy", "freeze_pretrained")
    OUTPUT_FILEPATH_HISTORY = OUTPUT_PREFIX + ".history.tsv"
    if not exists(OUTPUT_FILEPATH_HISTORY): # write column names if they are not there yet
        pd.DataFrame(columns = history_columns).to_csv(OUTPUT_FILEPATH_HISTORY, sep = "\t", header = True, index = False, na_rep = "NA", mode = "w") # write column names

    # history of percentiles in validation data
    percentiles_history_columns = ("epoch", "percentile", "value")
    OUTPUT_FILEPATH_PERCENTILES_HISTORY = OUTPUT_PREFIX + ".percentiles_history.tsv"
    if not exists(OUTPUT_FILEPATH_PERCENTILES_HISTORY): # write column names if they are not there yet
        pd.DataFrame(columns = percentiles_history_columns).to_csv(OUTPUT_FILEPATH_PERCENTILES_HISTORY, sep = "\t", header = True, index = False, na_rep = "NA", mode = "w") # write column names
    
    # percentiles for percentiles plots; define here since it doesn't need to be redefined every epoch
    percentiles = range(0, 101)

    # mark when I started training
    start_time = time()

    ##################################################


    # FUNCTION FOR TRAINING AN EPOCH
    ##################################################
    def train_an_epoch(epoch):

        # TRAIN AN EPOCH
        ##################################################

        # set to training mode
        key_nn.train()

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
            predictions = key_nn(inputs)

            # compute loss
            loss_batch = loss_criterion(predictions, labels)

            # backpropagate the gradients
            loss_batch.backward()

            # update the parameters
            optimizer.step()

            # compute the total loss for the batch and add it to history_epoch["train_loss"]
            history_epoch["train_loss"] += loss_batch.item() * inputs.size(0) # inputs.size(0) is the number of inputs in the current batch, assumes loss is an average over the batch
            
            # compute the accuracy
            accuracy_batch = torch.abs(input = predictions.view(-1) - labels.view(-1))

            # compute the total accuracy for the batch and add it to history_epoch["train_accuracy"]
            history_epoch["train_accuracy"] += torch.sum(input = accuracy_batch).item()
        
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
            key_nn.eval()

            # validation loop
            error_validate = torch.tensor(data = [], dtype = torch.float32).to(device)
            for inputs, labels in tqdm(data_loader["validate"], desc = "Validating"):

                # register inputs and labels with device
                inputs, labels = inputs.to(device), labels.to(device)

                # forward pass: compute predictions on input data using the model
                predictions = key_nn(inputs)

                # compute loss
                loss_batch = loss_criterion(predictions, labels)

                # compute the total loss for the batch and add it to history_epoch["validate_loss"]
                history_epoch["validate_loss"] += loss_batch.item() * inputs.size(0) # inputs.size(0) is the number of inputs in the current batch, assumes loss is an average over the batch
            
                # compute the accuracy
                accuracy_batch = torch.abs(input = predictions.view(-1) - labels.view(-1))

                # compute the total accuracy for the batch and add it to history_epoch["validate_accuracy"]
                history_epoch["validate_accuracy"] += torch.sum(input = accuracy_batch).item()

                # add accuracy to running count of all the errors in the validation dataset
                error_validate = torch.cat(tensors = (error_validate, accuracy_batch), dim = 0)

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

        # calculate percentiles
        percentile_values = percentile(error_validate.numpy(force = True), q = percentiles)
        pd.DataFrame(data = {"epoch": [epoch + 1,] * len(percentiles), "percentile": percentiles, "value": percentile_values}, columns = percentiles_history_columns).to_csv(OUTPUT_FILEPATH_PERCENTILES_HISTORY, sep = "\t", header = False, index = False, na_rep = "NA", mode = "a") # write to file

        # save current model if its validation accuracy is the best so far
        global best_accuracy
        if history_epoch["validate_accuracy"] <= best_accuracy:
            best_accuracy = history_epoch["validate_accuracy"] # update best_accuracy
            checkpoint = {
                "epoch": epoch,
                "state_dict": key_nn.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, NN_FILEPATH)

        # print out updates
        print(f"Training Time: {(total_time_epoch / 60):.1f} minutes")
        print(f"Training Loss: {history_epoch['train_loss']:.3f}, Validation Loss: {history_epoch['validate_loss']:.3f}")
        print(f"Mean Training Error: {history_epoch['train_accuracy']:.3f}, Mean Validation Error: {history_epoch['validate_accuracy']:.3f}")
        print(f"Five Number Summary of Validation Errors: {' '.join((f'{value:.2f}' for value in (percentile_values[percentile] for percentile in (0, 25, 50, 75, 100))))}")

        ##################################################


    ##################################################


    # TRAIN EPOCHS
    ##################################################

    # helper function for training 
    def train_epochs(start, n): # start = epoch to start training on; n = number of epochs to train from there
        epochs_to_train = range(start, start + n)
        for epoch in epochs_to_train:
            print("----------------------------------------------------------------")
            print(f"EPOCH {epoch + 1} / {epochs_to_train.stop}")
            train_an_epoch(epoch = epoch)
        print("================================================================")

    # train
    # start by training my section of the neural network for some epochs
    if FREEZE_PRETRAINED:
        print("Training final regression layer...")
        train_epochs(start = start_epoch, n = EPOCHS)
    elif not FREEZE_PRETRAINED:
        print("Fine-tuning pretrained layers...")
        train_epochs(start = start_epoch, n = EPOCHS)
    else:
        sys.exit("All parameters are frozen, so the model will not be trained. Exiting program...")

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
