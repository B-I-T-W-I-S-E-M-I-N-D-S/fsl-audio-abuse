# Downloading Dataset

Given the storage constraint of GitHub, we use [gdown](https://pypi.org/project/gdown/) to download the features as csv files.

The csv files contain the following columns:
1. ```audio_feature_x```: Where x marks the feature index between (0-n), where n=768 for Wav2Vec and n=1024 for Whisper.
2. ```train_test```: Train or test split, as given by the ADIMA dataset.
3. ```language```: One of the 10 Languages of the dataset.
4. ```abuse```: 0 for Not Abusive, 1 for Abusive.