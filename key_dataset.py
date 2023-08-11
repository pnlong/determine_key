# README
# Phillip Long
# August 10, 2023

# Create a custom audio dataset for PyTorch with torchaudio.
# Uses songs from my music library.

# python ./key_dataset.py labels_filepath output_filepath audio_dir


# IMPORTS
##################################################
import sys
from os.path import exists, join, dirname
from os import makedirs, remove
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset # base dataset class to create datasets
import torchaudio
import pandas as pd
# sys.argv = ("./key_dataset.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/tempo_key_data.tsv", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_data.tsv", "/Volumes/Seagate/artificial_dj_data/key_data")
##################################################


# CONSTANTS
##################################################
SAMPLE_RATE = 44100 // 2
SAMPLE_DURATION = 20.0 # in seconds
STEP_SIZE = SAMPLE_DURATION / 4 # in seconds, the amount of time between the start of each .wav file
SET_TYPES = {"train": 0.7, "validation": 0.2, "test": 0.1, "": 1.0} # train-validation-test fractions
KEY_MAPPINGS = ("C Maj", "A min",
                "G Maj", "E min",
                "D Maj", "B min",
                "A Maj", "F# min",
                "E Maj", "C# min",
                "B Maj", "G# min",
                "F# Maj", "D# min",
                "C# Maj", "A# min",
                "G# Maj", "F min",
                "D# Maj", "C min",
                "A# Maj", "G min",
                "F Maj", "D min",
                ) # in circle of fifths order
##################################################


# KEY DATASET OBJECT CLASS
##################################################

class key_dataset(Dataset):

    def __init__(self, labels_filepath, set_type, device, target_sample_rate = SAMPLE_RATE, sample_duration = SAMPLE_DURATION, use_pseudo_replicates = True):
        # set_type can take on one of three values: ("train", "validation", "test")

        # import labelled data file, preprocess
        # it is assumed that the data are mono wav files
        self.data = pd.read_csv(labels_filepath, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
        self.data = self.data[self.data["path"].apply(lambda path: exists(path))] # remove files that do not exist
        self.data = self.data[~pd.isna(self.data["key"])] # remove na values
        if not use_pseudo_replicates: # if no pseudo-replicates, transform self.data once more
            self.data = self.data.groupby(["title", "artist", "tempo", "path_origin"]).sample(n = 1).reset_index(drop = True) # randomly pick a sample from each song

        # partition into the train, validation, or test dataset
        self.data = self.data.sample(frac = SET_TYPES["" if set_type not in SET_TYPES.keys() else set_type], replace = False, ignore_index = True)
        self.data.reset_index(drop = True) # reset indicies

        # import constants
        # self.target_sample_rate = target_sample_rate # not being used right now
        # self.sample_duration = sample_duration # not being used right now
        self.device = device

        # import torch audio transformation(s), mel spectrogram transformation in this case
        self.transformation = torchaudio.transforms.MelSpectrogram(sample_rate = SAMPLE_RATE, n_fft = 1024, hop_length = 1024 // 2, n_mels = 64).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get waveform data by loading in audio
        signal, sample_rate = torchaudio.load(self.data.at[index, "path"], format = "wav") # returns the waveform data and sample rate
        # register signal onto device (gpu [cuda] or cpu)
        signal = signal.to(self.device)
        # resample; sample_rate was already set in preprocessing
        # signal, sample_rate = _resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = self.target_sample_rate, device = self.device) # resample for consistent sample rate
        # convert from stereo to mono; already done in preprocessing
        # signal = _mix_down_if_necessary(signal = signal)
        # pad/crop for fixed signal duration; duration was already set in preprocessing
        # signal = _edit_duration_if_necessary(signal = signal, sample_rate = sample_rate, target_duration = self.sample_duration) # crop/pad if signal is too long/short
        # apply transformations
        signal = self.transformation(signal) # convert waveform to melspectrogram

        return signal, torch.tensor([self.data.at[index, "key"]], dtype = torch.int32) # returns the transformed signal and the actual BPM
    
    # get info (title, artist, original filepath) of a file given its index; return as dictionary
    def get_info(self, index):
        return self.data.loc[index, ["title", "artist", "tempo", "path_origin", "path"]].to_dict()
    
    # sample n_predictions random rows from data, return a tensor of the audios and a tensor of the labels
    def sample(self, n_predictions):
        inputs_targets = [self.__getitem__(index = i) for i in self.data.sample(n = n_predictions, replace = False, ignore_index = False).index]
        inputs = torch.cat([torch.unsqueeze(input = input_target[0], dim = 0) for input_target in inputs_targets], dim = 0) # key_nn expects (batch_size, num_channels, frequency, time) [4-dimensions], so we add the batch size dimension here with unsqueeze()
        targets = torch.cat([input_target[1] for input_target in inputs_targets], dim = 0).view(n_predictions, 1)
        del inputs_targets
        return inputs, targets

##################################################


# HELPER FUNCTIONS
##################################################

# resampler
def _resample_if_necessary(signal, sample_rate, new_sample_rate, device):
        if sample_rate != new_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = new_sample_rate).to(device)
            signal = resampler(signal)
        return signal, new_sample_rate

# convert from stereo to mono
def _mix_down_if_necessary(signal):
    if signal.shape[0] > 1: # signal.shape[0] = # of channels; if # of channels is more than one, it is stereo, so convert to mono
        signal = torch.mean(signal, dim = 0, keepdim = True)
    return signal

# crop/pad if waveform is too long/short
def _edit_duration_if_necessary(signal, sample_rate, target_duration):
    n = int(target_duration * sample_rate) # n = desired signal length in # of samples; convert from seconds to # of samples
    if signal.shape[1] > n: # crop if too long
        signal = signal[:, :n]
    elif signal.shape[1] < n: # zero pad if too short
        last_dim_padding = (0, n - signal.shape[1])
        signal = torch.nn.functional.pad(signal, pad = last_dim_padding, value = 0)
    return signal

# given a mono signal, return the sample #s for which the song begins and ends (trimming silence)
def _trim_silence(signal, sample_rate, window_size = 0.1): # window_size = size of rolling window (in SECONDS)
    # preprocess signal
    signal = torch.flatten(input = signal) # flatten signal into 1D tensor
    signal = torch.abs(input = signal) # make all values positive
    # parse signal with a sliding window
    window_size = int(sample_rate * window_size) # convert window size from seconds to # of samples
    starting_frames = tuple(range(0, len(signal), window_size)) # determine starting frames
    is_silence = [torch.mean(input = signal[i:(i + window_size)]).item() for i in starting_frames] # slide window over audio and get average level for each window
    # determine a threshold to cutoff audio
    threshold = max(is_silence) * 1e-4 # determine a threshold, ADJUST THIS VALUE TO ADJUST THE CUTOFF THRESHOLD
    is_silence = [x < threshold for x in is_silence] # determine which windows are below the threshold
    start_frame = starting_frames[is_silence.index(False)] if sum(is_silence) != len(is_silence) else 0 # get starting frame of audible audio
    end_frame = starting_frames[len(is_silence) - is_silence[::-1].index(False) - 1] if sum(is_silence) != len(is_silence) else 0 # get ending from of audible audio
    return start_frame, end_frame

# get the key class index from key name
def get_key_index(key):
    return KEY_MAPPINGS.index(key)

##################################################


# if __name__ == "__main__" only runs the code inside the if statement when the program is run directly by the Python interpreter.
# The code inside the if statement is not executed when the file's code is imported as a module.
if __name__ == "__main__":

    # CONSTANTS
    ##################################################
    LABELS_FILEPATH = sys.argv[1]
    OUTPUT_FILEPATH = sys.argv[2]
    AUDIO_DIR = sys.argv[3]
    ##################################################


    # CREATE AND PREPROCESS WAV FILE CHOPS FROM FULL SONGS
    ##################################################

    # create audio output directory and output_filepath
    if not exists(AUDIO_DIR): 
        makedirs(AUDIO_DIR)
    if not exists(dirname(OUTPUT_FILEPATH)):
        makedirs(dirname(OUTPUT_FILEPATH))   

    # clear AUDIO_DIR
    for filepath in glob(join(AUDIO_DIR, "*")):
        remove(filepath)
    
    # determine what device to run things on
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")
    
    # load in labels
    data = pd.read_csv(LABELS_FILEPATH, sep = "\t", header = 0, index_col = False, keep_default_na = False, na_values = "NA")
    data = data[data["path"].apply(lambda path: exists(path))] # remove files that do not exist
    data = data[~pd.isna(data["key"])] # remove NA and unclassified keys
    data = data.reset_index(drop = True) # reset indicies

    # convert key column from key names (str) to class indicies (int)
    data["key"] = data["key"].apply(get_key_index)
    
    # loop through songs and create .wav files
    origin_filepaths, output_filepaths, keys = [], [], []
    for i in tqdm(data.index, desc = "Chopping up songs into WAV files"): # start from start index

        # preprocess audio
        try: # try to import the file
            signal, sample_rate = torchaudio.load(data.at[i, "path"], format = "mp3") # load in the audio file
        except RuntimeError:
            continue
        signal = signal.to(device) # register signal onto device (gpu [cuda] or cpu)
        signal, sample_rate = _resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = SAMPLE_RATE, device = device) # resample for consistent sample rate
        signal = _mix_down_if_necessary(signal = signal) # if there are multiple channels, convert from stereo to mono

        # chop audio into many wav files
        start_frame, end_frame = _trim_silence(signal = signal, sample_rate = sample_rate, window_size = 0.1) # return frames for which audible audio begins and ends
        window_size = int(SAMPLE_DURATION * sample_rate) # convert window size from seconds to frames
        starting_frames = tuple(range(start_frame, end_frame - window_size, int(STEP_SIZE * sample_rate))) # get frame numbers for which each chop starts
        for j, starting_frame in enumerate(starting_frames):
            path = join(AUDIO_DIR, f"{i}_{j}.wav") # create filepath
            torchaudio.save(path, signal[:, starting_frame:(starting_frame + window_size)], sample_rate = sample_rate, format = "wav") # save chop as .wav file
            origin_filepaths.append(data.at[i, "path"]) # add original filepath to origin_filepaths
            output_filepaths.append(path) # add filepath to output_filepaths
            keys.append(data.at[i, "key"]) # add key to keys
        
    # write to OUTPUT_FILEPATH
    data = data.rename(columns = {"path": "path_origin"}).drop(columns = ["key"]) # rename path column in the original dataframe
    key_data = pd.DataFrame(data = {"path_origin": origin_filepaths, "path": output_filepaths, "key": keys}) # create key_data dataframe
    key_data = pd.merge(key_data, data, on = "path_origin", how = "left").reset_index(drop = True) # left-join key_data and data
    key_data = key_data[["title", "artist", "tempo", "path_origin", "path", "key"]] # select columns
    # most of the information in key_data is merely to help me locate a file if it causes problem; in an ideal world, I should be able to ignore it
    print(f"\nWriting output to {OUTPUT_FILEPATH}.")
    key_data.to_csv(OUTPUT_FILEPATH, sep = "\t", header = True, index = False, na_rep = "NA") # write output

    ##################################################


    # TEST DATASET OBJECT
    ##################################################

    # instantiate key dataset
    key_data = key_dataset(labels_filepath = OUTPUT_FILEPATH, set_type = "", device = device)

    # test len() functionality
    print(f"There are {len(key_data)} samples in the dataset.")

    # test __getitem__ functionality
    signal, label = key_data[0]

    # test get_info() functionality
    print(f"The artist of the 0th sample is {key_data.get_info(0)['artist']}.")

    ##################################################