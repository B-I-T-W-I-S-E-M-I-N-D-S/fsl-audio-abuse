# Feature Extraction Module

This repository contains functions for feature extraction and feature normalization of the audio files.

Please note that you are required to download the Audio Files and the Annotations from the Original Repository of [ADIMA](https://github.com/ShareChatAI/ADIMA).

1. [```utils.py```](./utils.py): Contains normalisation and feature extraction modules.
2. [```wav2vec.py```](./wav2vec.py): Extracting features from audio using [Wav2Vec](https://huggingface.co/facebook/wav2vec2-xls-r-300m) from HuggingFace.
2. [```whisper.py```](./whisper.py): Extracting features from audio using [Whisper](https://huggingface.co/openai/whisper-large) from HuggingFace.