# Towards Cross-Lingual Audio Abuse Detection in Low-Resource Settings with Few-Shot Learning

This is the code repository for our paper published as part of the proceedings of COLING 2025.

## Abstract

Online abusive content detection, particularly in low-resource settings and within the audio modality, remains underexplored. We investigate the potential of pre-trained audio representations for detecting abusive language in low-resource languages, in this case, in Indian languages using Few Shot Learning (FSL). Leveraging powerful representations from models such as Wav2Vec and Whisper, we explore cross-lingual abuse detection using the ADIMA dataset with FSL. Our approach integrates these representations within the Model-Agnostic Meta-Learning (MAML) framework to classify abusive language in 10 languages. We experiment with various shot sizes (50-200) evaluating the impact of limited data on performance. Additionally, a feature visualization study was conducted to better understand model behaviour. This study highlights the generalization ability of pre-trained models in low-resource scenarios and offers valuable insights into detecting abusive language in multilingual contexts.


## Overview of the codebase

### Folders

1. [```data```](/data): Contains the script to download the dataset with the extracted embeddings.
2. [```feature-extraction```](/feature-extraction/): Contains scripts to extract features from audio files of the ADIMA dataset.
3. [```plots```](/plots/): Contains plots of Accuracy Scores and tSNE experiments.
3. [```results```](/results/): Contains csv files with FSL experiment results.
3. [```utils```](/utils/): Contains code for the FSL experiments.

### Code files
1. [```fsl-whisper.py```](/fsl-whisper.py): To run the Whisper-Related Experiments
2. [```fsl-wav2vec.py```](/fsl-wav2vec.py): To run the Wav2Vec-Related Experiments
3. [```results-plot.ipynb```](/results-plot.ipynb): Contains code to the plots presented in the paper.


Refer to the Original Authors Repository [ADIMA](https://github.com/ShareChatAI/ADIMA) for the audio files and annotations.

## Requirements

We performed the experiments with Python 3.11. The [```requriments.txt```](/requirements.txt) contains the necessary libraries needed for the task.

## Citation

Please use the following citation in case you use our work:
```
@inproceedings{sankaran2025towards,
    title = "Towards Cross-Lingual Audio Abuse Detection in Low-Resource Settings with Few-Shot Learning",
    author = "Sankaran, Aditya Narayan  and
      Farahbakhsh, Reza  and
      Crespi, Noel",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.373/",
    pages = "5558--5569",
    abstract = "Online abusive content detection, particularly in low-resource settings and within the audio modality, remains underexplored. We investigate the potential of pre-trained audio representations for detecting abusive language in low-resource languages, in this case, in Indian languages using Few Shot Learning (FSL). Leveraging powerful representations from models such as Wav2Vec and Whisper, we explore cross-lingual abuse detection using the ADIMA dataset with FSL. Our approach integrates these representations within the Model-Agnostic Meta-Learning (MAML) framework to classify abusive language in 10 languages. We experiment with various shot sizes (50-200) evaluating the impact of limited data on performance. Additionally, a feature visualization study was conducted to better understand model behaviour. This study highlights the generalization ability of pre-trained models in low-resource scenarios and offers valuable insights into detecting abusive language in multilingual contexts."
}
```
