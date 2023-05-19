# README
# Phillip Long
# February 18, 2023

# I just bought a Hercules Inpulse 200 DJ Controller with my friend Carson.
# However, one of the most important traits that a DJ must have is music curation -- after all, DJs are "vibe curators".
# To mix songs well, they need to have somewhat similar BPMs and keys.
# I want a program that can determine the BPM and key of all the songs in my library.
# Essentially, this program must edit the metadata of my mp3 files.


# Or not. This is just a test of the tensorflow package -- much thanks to this video: (https://www.youtube.com/watch?v=ZLIPkmmDJAc).
# The hope is that I can build an AI algorithm to classify songs by key using tensorflow,
# First, however, I must figure out how to use AI and tensorflow by classifying bird calls

# IMPORTS AND DEFINE FILEPATHS
##################################################

import os # for path names
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # disable tensorflow optimization message: https://stackoverflow.com/questions/65298241/what-does-this-tensorflow-message-mean-any-side-effect-was-the-installation-su
import tensorflow as tf # for creating neural networks
import tensorflow_io as tfio # for processing audio data in tensorflow
from time import sleep # for waits

# whether or not to display plots
visualize_summary_statistics = True
display_plots = False
if display_plots:
    from matplotlib import pyplot as plt # for visualization

# paths to sample audio clips
CAPUCHIN_FILE = "/Volumes/Seagate/data/capuchin/Parsed_Capuchinbird_Clips/XC3776-3.wav"
NOT_CAPUCHIN_FILE = "/Volumes/Seagate/data/capuchin/Parsed_Not_Capuchinbird_Clips/afternoon-birds-song-in-forest-0.wav"
# define paths to positive and negative data
POS = os.path.dirname(CAPUCHIN_FILE) # Positive examples
NEG = os.path.dirname(NOT_CAPUCHIN_FILE) # Negative examples

# make sure paths exists
if not (os.path.exists(POS) and os.path.exists(NEG)):
    print("Invalid filepaths. Try plugging in external hard drive!")
    quit()

##################################################

# BUILD DATA LOADING FUNCTION
##################################################

# define function
def load_wav_16k_mono(filename):
    # load encoded wav file
    file_contents = tf.io.read_file(filename)
    # decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
    # remove trailing axis
    wav = tf.squeeze(wav, axis = -1)
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)
    # goes from 44100 Hz to 16000 Hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = 16000)
    
    return wav

if display_plots:
    # create waveforms
    wave = load_wav_16k_mono(CAPUCHIN_FILE)
    nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)
    
    # plot waveforms
    waveform = plt.figure()
    waveform.canvas.manager.set_window_title("Waveform")
    plt.plot(wave, label = "Capuchin")
    plt.plot(nwave, label = "Not Capuchin")
    plt.legend(frameon = True)
    plt.show()
    
##################################################

# CREATE TENSORFLOW DATASET
##################################################

# create tensorflow datasets
pos = tf.data.Dataset.list_files(POS + "/*.wav")
neg = tf.data.Dataset.list_files(NEG + "/*.wav")

# add labels and combine positive and negative samples
positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

##################################################

# DETERMINE AVERAGE LENGTH OF CAPUCHIN CALL
##################################################

# calculate wave cycle length
lengths = []
for file in os.listdir(POS):
    tensor_wave = load_wav_16k_mono(os.path.join(POS, file))
    lengths.append(len(tensor_wave))

# calculate mean, min, and max
mean_size = tf.math.reduce_mean(lengths)
min_size = tf.math.reduce_min(lengths)
max_size = tf.math.reduce_max(lengths)

##################################################

# BUILD PREPROCESSING FUNCTION TO CONVERT TO SPECTROGRAM
##################################################

# build preprocessing function
a = min_size + tf.cast(x = tf.cast(x = 1/4, dtype = tf.float32) * tf.cast(x = max_size - min_size, dtype = tf.float32), dtype = tf.int32)
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:a]
    zero_padding = tf.zeros([a] - tf.shape(wav), dtype = tf.float32) # pad audio file if sample is not long enough
    wav = tf.concat([zero_padding, wav], 0) # concatenate pad to the wav file
    spectrogram = tf.signal.stft(wav, frame_length = 320, frame_step = 32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis = 2)
    
    return spectrogram, label

# test out the function and visualize the spectrogram
filepath, label = positives.shuffle(buffer_size = 10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)

if display_plots:
    # plot spectrogram
    spectrogram_plot = plt.figure()
    plt.imshow(tf.transpose(spectrogram)[0])
    plt.show()

##################################################

# CREATE TRAINING AND TESTING PARTITIONS
##################################################

# create a tensorflow data pipeline
data = data.map(preprocess) # apply spectrogram method to each sample
data = data.cache()
data = data.shuffle(buffer_size = 1000) # mix up all the training samples, makes sure we don't overfit or introduce any unnecessary bias or variance
data = data.batch(16) # train on 16 samples of a time
data = data.prefetch(8) # prefetch 8 examples to eliminate any CPU bottlenecking

# split into training and testing partitions
b = round(len(data) * 0.7) # train on 70% of data
train = data.take(b)
test = data.skip(b).take(len(data) - b) # test of remaining 30% of data

# test one batch
samples, labels = train.as_numpy_iterator().next()

##################################################

# BUILD DEEP LEARNING CNN MODEL
##################################################

print("building model...")

# load tensorflow dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# build sequential model, compile, and view summary
model = Sequential() # instantiate instance of this class
model.add(Conv2D(16, (3, 3), activation = "relu", input_shape = spectrogram.shape)) # we want 16 different kernels of shape 3x3
model.add(Conv2D(16, (3, 3), activation = "relu"))
model.add(Flatten()) # flatten convolutional outputs (in 3 dimensions) to one dimension
model.add(Dense(128, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.compile("Adam", loss = "BinaryCrossentropy", metrics = [tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]) # compile model

model.summary()

# fit model, view loss and KPI plots
print("training model...")
hist = model.fit(train, epochs = 4, validation_data = test)

# plot some summary statistics of neural network
if display_plots and visualize_summary_statistics:
    figure, axes = plt.subplot_mosaic([["topleft", "topright"],["bottomleft", "bottomright"]],
                                      constrained_layout = True, figsize = (15, 10))
    
    # plot Loss
    axes["topleft"].plot(hist.history["loss"], color = "red")
    axes["topleft"].plot(hist.history["val_loss"], color = "blue")
    axes["topleft"].set(title = "Loss")
    
    # plot Precision
    axes["topright"].plot(hist.history["precision"], color = "red")
    axes["topright"].plot(hist.history["val_precision"], color = "blue")
    axes["topright"].set(title = "Precision")
    
    # plot Recall
    axes["bottomright"].plot(hist.history["recall"], color = "red")
    axes["bottomright"].plot(hist.history["val_recall"], color = "blue")
    axes["bottomright"].set(title = "Recall")
    
    plt.show()
    
##################################################

# MAKE A PREDICTION ON A SINGLE AUDIO CLIP
##################################################

# get one batch and make a prediction
print("predicting")
X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)

# convert logits to classes
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

##################################################

# BUILD FOREST PARSING FUNCTION
##################################################

# load up MP3s
def load_mp3_16k_mono(filename):
    # load mp3 file, convert it to a float tensor, resample to 16kHz single-channel
    res = tfio.audio.AudioIOTensor(filename) # take MP3 file and convert to tensor
    # convert to tensor and combine channels
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis = 1) / 2 # take the average of two channels to reduce to a single channel value
    # extract sample rate and cast
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)
    # resample to 16 kHz
    wav = tfio.audio.resample(tensor, rate_in = sample_rate, rate_out = 16000)
    
    return wav

mp3 = "/Volumes/Seagate/data/capuchin/Forest\ Recordings/recording_00.mp3"
wav = load_mp3_16k_mono(mp3)

# convert file into audio slices
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length = a, sequence_stride = a, batch_size = 1) 
samples, index = audio_slices.as_numpy_iterator().next()

# build function to convert clips into windowed spectrograms
def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([a] - tf.shape(sample), dtype = tf.float32) # pad audio file if sample is not long enough
    wav = tf.concat([zero_padding, sample], 0) # concatenate pad to the wav file
    spectrogram = tf.signal.stft(wav, frame_length = 320, frame_step = 32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis = 2)
    
    return spectrogram

# convert longer clips into windows and make predictions
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length = a, sequence_stride = a, batch_size = 1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)

yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.99 else 0 for prediction in yhat]

# group consecutive detections
from itertools import groupby
yhat = [key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()

##################################################

# MAKE PREDICTIONS
##################################################

# loop over all recordings and make predictions
results = {}
for file in os.list_dir(os.path.dirname(mp3)):
    FILEPATH = os.path.dirname(mp3) + file
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length = a, sequence_stride = a, batch_size = 1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)

    yhat = model.predict(audio_slices)
    
    results[file] = yhat
    
# convert predictions into classes
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in yhat]
    
# group consecutive detections
postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(yhat)]).numpy()

##################################################