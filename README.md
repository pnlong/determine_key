# determine_key
Determine the musical key of a given audio sample.

---

## Background

Though perhaps not as important as tempo, musical key comes in at a close second. If DJs do not mix into songs with certain correct keys, to an untrained ear, their mix will sound cacauphonous. To a trained ear, it just sounds bad. Therefore, it is important that DJs know the keys of both the song they are mixing out of and the song they are mixing into. I will use machine learning to take an audio file (.MP3) as input and output a song's musical key through two values:

1. The song's relative key / key signature (Ex. "A Major / f# minor")
2. Whether the song is in a Major or minor key


## Data

I can use two approaches:

- [Looperman.com](https://www.looperman.com/) offers a massive selection of audio samples and loops, most of which are labelled by key and BPM. Though I believe that these samples would not be beneficial for training a neural network to determine BPM, as they are too short, they could be useful for training a neural network to detect key, as most songs repeat the same chord progression over and over and can thus be aptly described in a short sample. Additionally, for each loop, I can transpose it into the 11 other possible keys. This means if I download 1000 loops, I will have 12000 data points. Going even further, I could put the first `n` seconds of a loop at the end of the loop for all my loops, repeating this many times -- this strategy has potential to massively increase the amount of data I have. However, this might ruin some chord progressions and actually hurt the accuracy of the dataset, so I will stay away from it for now. I could scrape maybe 1000 loops off of Looperman and use this as training data for a neural network to detect key, using a 70%-20%-10% split of training, cross-validation, and test data, respectively. The main downside of this method is that key information might not extrapolate to longer audio files (i.e. a neural network trained on loops cannot determine the key of an entire song). The upside, however, is that this method has potential for a lot of data.
- I have to label the different sections (intro, verse, chorus, outro) of songs for the next phase of my AI DJ project. I could push this step forward and use this information as training data for detecting key and BPM. For example, if I have labelled a song to have an intro, verse, chorus, verse, chorus, and outro, I will divide the song into 6 separate audio files, one for each section, still retaining the metadata of the original track. What this equates to is that for each song in my ideally ~1000 track collection, I will now have an average of ~6 files per song, meaning I have much more data for training. The main downside of this method is that the dataset is hard to make, and thus there will not be as much data when compared to the Looperman method. The upside, however, is that this dataset is applicable to multiple phases of the project, and for the purposes of key determination, it is more likely to extrapolate to longer audio files.


## Machine Learning


## Output




---

## Software

### *.py*