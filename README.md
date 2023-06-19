# determine_key
Determine the musical key of a given audio sample.

---

## Background

Though perhaps not as important as tempo, musical key comes in at a close second. If DJs do not mix into songs with certain correct keys, to an untrained ear, their mix will sound cacauphonous. To a trained ear, it just sounds bad. Therefore, it is important that DJs know the keys of both the song they are mixing out of and the song they are mixing into. I will use machine learning to take an audio file (.MP3) as input and output a song's musical key through two values:

1. The song's relative key / key signature (Ex. "A Major / f# minor")
2. Whether the song is in a Major or minor key


## Data

I can use two approaches for collecting data:

- [Looperman.com](https://www.looperman.com/) offers a massive selection of audio samples and loops, all of which are labelled by key and BPM. Though I believe that these samples would not be beneficial for training a neural network to determine BPM as they are too short, they could be useful for training a neural network to determine key, as most songs repeat the same chord progression over and over and can thus be aptly described in a short audio sample. The main downside of this method is that key information might not extrapolate to longer audio files (i.e. a neural network trained on loops cannot determine the key of an entire song). The upside, however, is that this method has potential for generating a lot of data, since Looperman is a plentiful resource for audio data.
- I have to label the different sections (intro, verse, chorus, outro) of the songs in my collection for the next phase of my AI DJ project anyways. I could push this step forward and use this information as training data for detecting key and BPM. For example, with a track labelled as having an intro, verse, chorus, verse, chorus, and outro, I can divide the song into 6 separate audio files, one per section and still retaining the metadata of the original track. What this equates to is that for each song in my ideally ~1000 track collection, I will now have an average of ~6 files per song, meaning I have much more data for training. The main downside of this method is that the dataset is hard to construct, and thus it had less potential for a lot of data when compared to the Looperman method. The upside, however, is that this dataset is applicable to multiple phases of the project, and for the purposes of key determination, I believe it is more likely to extrapolate to longer audio files.

For either data approach, I would split the data 70%-20%-10% into training, cross-validation, and test datasets, respectively.

I will use Fourier transforms to convert raw-audio-waveform into melspectrogram data, which I will then use to train my neural networks. Raw audio waveforms represent amplitude (not a pitch-based value) as a function of time, and are thus hard to use for determining key; melspectrograms, on the other hand, display what is effectively pitch as a function of time. [Melspectrograms are better than normal spectrograms](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53) because the mel scale better represents the human perception of pitch than the frequency scale, and thus, musical notes and chords are more apparent.


## Strategies for Improving Data

On its own, audio data is hard to collect. Of the two methods mentioned previously, the latter is especially time consuming to label. Despite this, we can use certain strategies to improve data quality by creating larger datasets from our original audio files.

- **Key Extrapolation** is a term I have coined for creating a lot of new audio data from an audio file that has been labelled with a musical key. For a given loop in a certain key, I can transpose the audio file into the 11 other possible keys. I will create new audio files for the 6 keys transposed-up and 5 keys transposed-down from the original, which is better than the alternative of tranposing 11 keys up or down as both options will significantly distort later keys. For instance, if I receive a loop in D Major, I can transpose it up, creating a new file for D# Major all the way up to G# Major, and transpose it down, creating a new file for C# Major all the way down to A Major. This will generate 12 times more data than before. Given 1000 loops, there is the potential for 12000 easily-obtained data points.
- **Audio Shuffling** is another term I have coined for creating a lot of new audio data from an audio file, ideally a loop. Performing this action many times, I could identify each bar of a loop and shuffle them around, creating multiple new loops. Chord progressions would be altered, but because none of the notes changed, in theory, the audio will retain its original key. Though this strategy has the potential to massively increase the amount of data I have, in practice, it could ruin some chord progressions and actually hurt the accuracy of the dataset. I will stay away from it for now.


## Machine Learning

I will use PyTorch to create two Convolutional Neural Networks (CNNs) that take .MP3 files as input and each output a value, the song's relative key and whether the sample is Major or minor. The hope is that CNNs can better detect patterns in the audio data such as a bassline or chords that are useful in determining a song's key. I face the same debate as I do for [determining tempo](https://github.com/pnlong/determine_tempo): should I use a CNN with windowing and pick the median key value, or should I use Long Short Term Memory (LSTM)? The advantage of the former is the potential for a lot of data, though the latter seems like it is more suited for this challenge (not to mention I could learn a lot from implementing LSTM or even Attention).


## Output

As previously mentioned, the neural networks will each have a different output:

1. **Relative Key (Key Signature)**: As a multilabel classification problem, the final layer of this neural network will be a SoftMax function that outputs a vector of 12 probabilities -- indicies 1 through 12 represent A Major (f# minor) through G# Major (f minor), respectively. The index with the highest probability will decide the relative key (key signature) of the audio file. For instance, if index 4 yields the highest probability, then the song is either in C Major or a minor and it is up for the second neural network to decide between the two.
2. **Major or minor**: As a binary classification problem, the final layer of this neural network will be a single sigmoid activation function representing the probability that the file is in a Major key. A value >=0.5 suggests that the audio file is in a Major key, while a value <0.5 suggests the song is in a minor key.


---

## Software

### *.py*