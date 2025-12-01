import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

from utils import feature_extractor

# Path to the directory containing CSV files
csv_directory = "./annotations/"

# Path to the directory containing audio directories
audio_directory = "./content/audio/SC_audio_"

# List of language names
languages = ["Bengali", "Bhojpuri" ,"Gujarati", "Haryanvi","Hindi", "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"] 

# Initialize an empty list to store data
data = []

# Iterate through each language and train/test combination
for language in languages:
    for split in ["train", "test"]:
        # Read the CSV file
        csv_filename = f"{language}_{split}.csv"
        csv_path = os.path.join(csv_directory, csv_filename)
        csv_data = pd.read_csv(csv_path)
        
        # Iterate through each row in the CSV data
        for index, row in csv_data.iterrows():
            t = audio_directory+language+'/'
            audio_path = os.path.join(t, row['filename'])
            data.append({
                'path_to_audio': audio_path,
                'language': language,
                'train_test': split,
                'abuse': row['label']
            })

df = pd.DataFrame(data)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(42)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.manual_seed(42)
else:
    device = torch.device("cpu")
print(f"Running on {device}.")

model_name = "openai/whisper-base"

model = AutoModel.from_pretrained(model_name).to(device)
processor = AutoProcessor.from_pretrained(model_name)

feature_extractor(df, "whisper", "l2", languages, processor, model, device) 
feature_extractor(df, "whisper", "temp", languages, processor, model, device)
