import pandas as pd
import numpy as np
import os

# Path to the directory containing CSV files with labels
csv_directory = "./annotations/"

# Path to the directory containing audio directories
audio_directory = "./content/audio/SC_audio_"

# List of language names
languages = ["Bengali", "Bhojpuri", "Gujarati", "Haryanvi", "Hindi", 
             "Kannada", "Malayalam", "Odia", "Punjabi", "Tamil"]

def create_csv_from_npy(feature_dir, output_csv):
    """
    Combines .npy feature files with labels into a single CSV
    
    Args:
        feature_dir: Directory containing the .npy files (e.g., './features/whisper')
        output_csv: Output CSV filename (e.g., './data/whisper-feat.csv')
    """
    all_data = []
    
    for language in languages:
        print(f"Processing {language}...")
        
        # Load the embeddings
        npy_path = f"{feature_dir}/wav2vec-{language}-emb.npy"
        
        if not os.path.exists(npy_path):
            print(f"Warning: {npy_path} not found. Skipping...")
            continue
        
        embeddings = np.load(npy_path)
        print(f"  Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
        
        # Load corresponding labels from train and test CSV files
        for split in ["train", "test"]:
            csv_filename = f"{language}_{split}.csv"
            csv_path = os.path.join(csv_directory, csv_filename)
            
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found. Skipping...")
                continue
            
            csv_data = pd.read_csv(csv_path)
            
            # Get the number of samples for this split
            num_samples = len(csv_data)
            
            # Match embeddings with labels (assumes same order as original processing)
            if split == "train":
                start_idx = 0
                end_idx = num_samples
            else:
                # For test split, embeddings come after train split
                train_csv = pd.read_csv(os.path.join(csv_directory, f"{language}_train.csv"))
                start_idx = len(train_csv)
                end_idx = start_idx + num_samples
            
            # Get embeddings for this split
            split_embeddings = embeddings[start_idx:end_idx]
            
            # Create dataframe rows
            for i, (_, row) in enumerate(csv_data.iterrows()):
                if i >= len(split_embeddings):
                    print(f"Warning: Mismatch in {language} {split} - not enough embeddings")
                    break
                
                # Create a dictionary with embedding features and metadata
                row_data = {f'feature_{j}': split_embeddings[i][j] 
                           for j in range(split_embeddings.shape[1])}
                row_data['abuse'] = row['label']
                row_data['language'] = language
                row_data['train_test'] = split
                
                all_data.append(row_data)
    
    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save to CSV (with index to match expected format)
    df.to_csv(output_csv, index=True)
    print(f"\nSaved {len(df)} samples to {output_csv}")
    print(f"Columns: {df.columns.tolist()[:5]}... (showing first 5)")
    print(f"Shape: {df.shape}")
    
    return df

# Create both CSV files
print("Creating Temporal-Mean CSV...")
df_temporal = create_csv_from_npy('./features/whisper', './data/whisper-feat.csv')

print("\n" + "="*50)
print("Creating L2-Norm CSV...")
df_l2 = create_csv_from_npy('./features/whisper-l2', './data/whisper-l2-feats.csv')

print("\n" + "="*50)
print("Done! You can now run fsl-whisper.py")
