# README
# Phillip Long
# September 5, 2023

# Given a song, output its predicted key.

# python ./determine_key.py key_class_nn_filepath key_quality_nn_filepath song_filepath


# IMPORTS
##################################################
import sys
from os.path import dirname, join, exists
import torch
import torchaudio
import torchvision
import key_dataset as key_data
from key_class_neural_network import key_class_nn
from key_quality_neural_network import key_quality_nn, CONFIDENCE_THRESHOLD
# sys.argv = ("./determine_key.py", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_class_nn.pth", "/Users/philliplong/Desktop/Coding/artificial_dj/data/key_quality_nn.pth", "")
##################################################


# MAIN FUNCTION
##################################################

class key_determiner():

    # determine device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initializer function
    def __init__(self, nn_filepath = {"class": join(dirname(__file__), "key_class_nn.pth"), "quality": join(dirname(__file__), "key_quality_nn.pth")}):

        # IMPORT NEURAL NETWORKS
        ##################################################

        # preliminary checks
        if type(nn_filepath) is not dict:
            raise TypeError("Invalid nn_filepath argument: nn_filepath is not of type 'dict'.")
        if nn_filepath.keys() != {"class", "quality"}:
            raise ValueError("Invalid nn_filepath argument: incorrect keys for nn_filepath dict. nn_filepath.keys() = {'class', 'quality'}")
        if not exists(nn_filepath["class"]):
            raise ValueError(f"Invalid nn_filepath argument: {nn_filepath['class']} does not exist.")
        if not exists(nn_filepath["quality"]):
            raise ValueError(f"Invalid nn_filepath argument: {nn_filepath['quality']} does not exist.")
        
        # import neural networks, load parameters
        # key class
        self.key_class_nn = key_class_nn().to(self.device)
        self.key_class_nn.load_state_dict(torch.load(nn_filepath["class"], map_location = self.device)["state_dict"])
        # key quality
        self.key_quality_nn = key_quality_nn().to(self.device)
        self.key_quality_nn.load_state_dict(torch.load(nn_filepath["quality"], map_location = self.device)["state_dict"])

        ##################################################


    # determine the key of a given song
    def determine_key(self, song_filepath):

        # LOAD AUDIO, RESAMPLE, CONVERT TO MONO
        ##################################################

        # get waveform data by loading in audio
        signal, sample_rate = torchaudio.load(song_filepath, format = song_filepath.split(".")[-1]) # returns the waveform data and sample rate
        
        # register signal onto device
        signal = signal.to(self.device)

        # resample
        signal, sample_rate = key_data._resample_if_necessary(signal = signal, sample_rate = sample_rate, new_sample_rate = key_data.SAMPLE_RATE, device = self.device) # resample for consistent sample rate

        # convert from stereo to mono
        signal = key_data._mix_down_if_necessary(signal = signal)
        
        ##################################################


        # SPLIT SONG INTO CLIPS
        ##################################################

        # determine where to chop audio up
        start_frame, end_frame = key_data._trim_silence(signal = signal, sample_rate = sample_rate, window_size = 0.1) # return frames for which audible audio begins and ends
        window_size = int(key_data.SAMPLE_DURATION * sample_rate) # convert window size from seconds to frames
        starting_frames = tuple(range(start_frame, end_frame - window_size, int(key_data.STEP_SIZE * sample_rate))) # get frame numbers for which each chop starts

        # create clips, apply transforms
        inputs = torch.tensor(data = [], dtype = torch.float32).to(self.device)
        for starting_frame in starting_frames:
            clip = torch.unsqueeze(input = self._transform(signal = signal[:, starting_frame:(starting_frame + window_size)], sample_rate = sample_rate), dim = 0).to(self.device)
            inputs = torch.cat(tensors = (inputs, clip), dim = 0)
        
        ##################################################


        # APPLY NEURAL NETWORK, MAKE PREDICTION
        ##################################################

        # key class
        predictions_class = self.key_class_nn(inputs)
        predictions_class = torch.argmax(input = predictions_class, dim = 1, keepdim = True).view(-1) # convert to class indicies
        predicted_class = torch.mode(input = predictions_class, dim = 0)[0].item() # get the mode

        # key quality
        predictions_quality = self.key_quality_nn(inputs)
        predictions_quality = (predictions_quality >= CONFIDENCE_THRESHOLD).view(-1) # convert to class indicies
        predicted_quality = (torch.mean(input = predictions_quality, dim = 0, dtype = torch.float32).item() >= CONFIDENCE_THRESHOLD) # make quality prediction
        
        # make prediction
        predicted_key = key_data.get_key(key_class_index = predicted_class, key_quality_index = predicted_quality)

        ##################################################

        return predicted_key
    

    # transform functions
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate = key_data.SAMPLE_RATE, n_fft = key_data.N_FFT, n_mels = key_data.N_MELS).to(device) # make sure to adjust MelSpectrogram parameters such that # of mels > 224 and ceil((2 * SAMPLE_DURATION * SAMPLE_RATE) / n_fft) > 224
    normalize = torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).to(device) # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)


    # transform a waveform into the input of the neural network
    def _transform(self, signal, sample_rate):

        # pad/crop for fixed signal duration; duration was already set in preprocessing
        signal = key_data._edit_duration_if_necessary(signal = signal, sample_rate = sample_rate, target_duration = key_data.SAMPLE_DURATION) # crop/pad if signal is too long/short

        # convert waveform to melspectrogram
        signal = self.mel_spectrogram(signal) # (single channel, # of mels, # of time samples) = (1, 128, ceil((SAMPLE_DURATION * SAMPLE_RATE) / n_fft) = 431)

        # perform local min-max normalization such that the pixel values span from 0 to 255 (inclusive)
        signal = (signal - torch.min(signal).item()) * (255 / (torch.max(signal).item() - torch.min(signal).item()))

        # make image height satisfy PyTorch image processing requirements, (1, 128, 431) -> (1, 256, 431)
        signal = torch.repeat_interleave(input = signal, repeats = key_data.IMAGE_HEIGHT // key_data.N_MELS, dim = 1)

        # convert from 1 channel to 3 channels (mono -> RGB); I will treat this as an image classification problem
        signal = torch.repeat_interleave(input = signal, repeats = 3, dim = 0) # (3 channels, # of mels, # of time samples) = (3, 256, 431)

        # normalize the image according to PyTorch docs (https://pytorch.org/vision/0.8/models.html)
        signal = self.normalize(signal)

        # return the signal as a transformed tensor registered to the correct device
        return signal

##################################################


# PROOF OF FUNCTION
##################################################

if __name__ == "__main__":

    song_filepath = sys.argv[3]
    nn_filepath = {"class": sys.argv[1], "quality": sys.argv[2]}
    key_determiner = key_determiner(nn_filepath = nn_filepath)
    predicted_key = key_determiner.determine_key(song_filepath = song_filepath)
    print(f"Predicted key of {song_filepath}: {predicted_key}or")

##################################################