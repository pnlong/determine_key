# README
# Phillip Long
# February 16, 2023

# I just bought a Hercules Inpulse 200 DJ Controller with my friend Carson.
# However, one of the most important traits that a DJ must have is music curation -- after all, DJs are "vibe curators".
# To mix songs well, they need to have somewhat similar BPMs and keys.
# I want a program that can determine the BPM and key of all the songs in my library.
# Essentially, this program must edit the metadata of my mp3 files.

# My test files are as follows:
#
#   - /Users/philliplong/Desktop/Coding/determine_tempokey/test1.wav
#       - 174 bpm
#       - A Major
#
#   - /Users/philliplong/Desktop/Coding/determine_tempokey/test2.wav
#       - 105 bpm
#       - e minor
#

# Or not. This is just a test of the librosa package -- much thanks to this video: (https://www.youtube.com/watch?v=MhOdbtPhbLU).

# IMPORTS
##################################################

import sys # for system arguments
import os # for checking directories
import re # regular expressions
import librosa # manipulate audio
import numpy # for vectors and math
import pandas # for tables

##################################################

# ARGUMENTS
##################################################

# Do we want to display plots for me to see what is going on?
display_plots = True

# python /Users/philliplong/Desktop/Coding/determine_tempokey/determine_tempokey.py audio_file
sys.argv = ("/Users/philliplong/Desktop/Coding/determine_tempokey/determine_tempokey.py",
            "/Users/philliplong/Desktop/Coding/determine_tempokey/test1.wav")

# PATH TO SOUND FILE
path = str(sys.argv[1]) # path to sound file

# MAKE SURE PATH EXISTS AND IS MP3
if not os.path.exists(path): # if filepath is invalid
    print("Error: file path does not exist.")
    quit()
elif not re.match("^.*(mp3|wav)$", path): # if file is not an mp3 or wav
    print("Error: invalid file format.")
    quit()
# CONVERT WAV FILES TO MP3
# elif path.endswith("wav"):
#     import warnings # silence warnings
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore") # silence "RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work" Error
#         import pydub # for converting wav forms
#         pydub.AudioSegment.converter = "/Users/philliplong/Desktop/Coding/ffmpeg" # Download ffmpeg from: https://ffmpeg.org/download.html
#     sound = pydub.AudioSegment.from_wav(path)
#     path = path[:(len(path) - 3)] + "mp3"
#     if not os.path.exists(path):
#         sound.export(path, format = "mp3")
#     del pydub

##################################################

# READ IN AUDIO, DETERMINE WAVEFORM
##################################################

# load in audio
y, sr = librosa.load(path) # y = audio data, sr = sampling rate

# filter harmonic and percussive elements
y_harmonic, y_percussive = librosa.effects.hpss(y = y) # harmonic-percussive separation

##################################################

# CONVERT TIME DOMAIN TO FREQUENCY DOMAIN
##################################################

# some terminology...
#   - sr = sampling rate (Hz)
#   - frame = short audio clip
#   - n_fft = samples per frame
#   - hop_length = # samples between frames; how far the frame moves each time

# CQT measures the energy in each pitch (C1, C2, C3, C4)
# Chroma measures the energy in each pitch class (C)


# spectrogram data with direct log-frequency analysis
C = librosa.cqt(y = y_harmonic, sr = sr) # constant-Q transform, which is nice, because when plotted, one vertical move is one semitone
C_db = librosa.amplitude_to_db(S = numpy.abs(C), ref = numpy.max, top_db = 94) # convert C to decibels, top_db is the decibel cutoff, average use of personal audio device = 94 db/hr
C_db -= numpy.min(C_db) # force the quietest decibels equal to 0, shift the rest accordingly by subtracting the minimum value from everything

# determine chroma
chroma = librosa.feature.chroma_cqt(y = y_harmonic, sr = sr)

# onsets (new notes)
onset_envelope = librosa.onset.onset_strength(y = y_harmonic, sr = sr)
onsets = librosa.onset.onset_detect(onset_envelope = onset_envelope)

##################################################

# DETERMINE TEMPO
##################################################

# determine tempo
tempo, beats = librosa.beat.beat_track(onset_envelope = onset_envelope, sr = sr)
beat_times = librosa.frames_to_time(beats)
beats_to_display = numpy.arange(start = beat_times[0], stop = len(y_harmonic) / sr, step = (tempo / 60) ** -1)

##################################################

# IDENTIFY MOST COMMON PITCHES
##################################################

# article with approaches: https://stackoverflow.com/questions/57082826/how-can-a-chromagram-file-produced-by-librosa-be-interpreted-as-a-set-of-musical

# average decibel for each pitch (Mean Decibel Per Pitch)
mdpp = pandas.Series(data = numpy.asarray([abs(numpy.mean(a = pitch)) for pitch in chroma]), # take absolute value to get rid of complex numbers
                     index = librosa.key_to_notes(key = "C:maj")) # align to each note
mdpp = mdpp.sort_values(ascending = False) # arrange from highest mean decibel value to lowest

##################################################

# DISPLAY PLOTS?
##################################################

# show plots if display_plots is true
if display_plots:
    
    # imports
    from matplotlib import pyplot as plt # import for graphics
    import librosa.display # for audio visualization
        
    # set up full plot (https://librosa.org/doc/main/auto_examples/plot_display.html)
    figure, axes = plt.subplot_mosaic([["topleft", "topmiddle", "topright"],["bottomleft", "bottomright", "bottomright"]],
                                      constrained_layout = True, figsize = (15, 10))
    figure.canvas.manager.set_window_title(os.path.basename(path))
    colormap = "YlOrBr" # color map options: https://matplotlib.org/stable/tutorials/colors/colormaps.htmll
    

    # display original waveform, https://librosa.org/doc/main/generated/librosa.display.waveshow.html
    waveform = librosa.display.waveshow(y = y, sr = sr, ax = axes["topleft"], color = "orange")
    axes["topleft"].vlines(beats_to_display, -y.max(), y.max(), color = "sienna", alpha = 0.9)
    axes["topleft"].set(title = "Waveform", ylabel = "Amplitude")
    
    # display onsets and beats
    axes["topmiddle"].plot(onset_envelope, label = "Onset Strength", color = "orange")
    axes["topmiddle"].vlines(onsets, 0, onset_envelope.max(), color = "sienna", alpha = 0.9, label = "Onsets")
    axes["topmiddle"].legend(frameon = True)
    axes["topmiddle"].axis("tight")
    axes["topmiddle"].set(title = "Onsets", xticks = [], yticks = [])
    
    # display spectrogram of CQT
    spectrogram_cqt = librosa.display.specshow(data = C_db, sr = sr, x_axis = "time", y_axis = "cqt_note", ax = axes["topright"], cmap = colormap)
    axes["topright"].set(title = "Spectrogram CQT")    
    
    # display spectrogram of chroma
    spectrogram_chroma = librosa.display.specshow(data = numpy.abs(chroma), sr = sr, x_axis = "time", y_axis = "chroma", ax = axes["bottomright"], cmap = colormap)
    axes["bottomright"].set(title = "Spectrogram Chroma")
    
    # display average decibel level per pitch
    mdpp.plot(ax = axes["bottomleft"], kind = "barh", color = "sienna")
    axes["bottomleft"].set(title = "Mean Decibel Level per Pitch", xlabel = "Mean Decibel Level", ylabel = "Pitch Class")
    axes["bottomleft"].invert_yaxis()
           
    # make color bar
    figure.colorbar(spectrogram_cqt, ax = [axes["topright"], axes["bottomright"]], format = "%2.f dB")
    
    # zoom in on chroma spectrogram
    # axes["bottomright"].set(xlim = [0, 2]) 
    
    # display plot
    plt.show() 

##################################################

