# utils.py - Add error handling and batch progress
from tqdm.auto import tqdm
import numpy as np
import torch
import librosa
import os

def temporal_mean(tensor_list, lang, name):
    if len(tensor_list) == 0:
        print(f"Warning: No data for language {lang}. Skipping...")
        return None
    
    vectors = []
    for tensor in tensor_list:
        tensor = np.squeeze(tensor, axis=0)
        vector = np.mean(tensor, axis=0)
        vectors.append(vector)
    result_array = np.vstack(vectors)
    
    # Create directory if it doesn't exist
    os.makedirs(f"./features/{name}", exist_ok=True)
    np.save(f"./features/{name}/wav2vec-{lang}-emb", result_array)
    return result_array

def l2_norm(tensor_list, lang, name):
    if len(tensor_list) == 0:
        print(f"Warning: No data for language {lang}. Skipping...")
        return None
    
    vectors = []
    for tensor in tensor_list:
        tensor = np.squeeze(tensor, axis=0)
        l2_norms = np.sqrt(np.sum(np.square(tensor), axis=1, keepdims=True))
        normalized_tensor = tensor / l2_norms
        vector = np.mean(normalized_tensor, axis=0)
        vectors.append(vector)
    result_array = np.vstack(vectors)
    
    # Create directory if it doesn't exist
    os.makedirs(f"./features/{name}-l2", exist_ok=True)
    np.save(f"./features/{name}-l2/wav2vec-{lang}-emb", result_array)
    return result_array

def feature_extractor(df, name, q, languages, processor, model, device):
    decoder_input_ids = torch.tensor([[1024, 1024]]).to(device)
    for lang in tqdm(languages):
        temp = list(df[df['language']==lang]['path_to_audio'])
        
        if len(temp) == 0:
            print(f"Warning: No audio files found for {lang}. Skipping...")
            continue
        
        emb = []
        processed_count = 0
        
        for idx, path in enumerate(tqdm(temp, desc=f"Processing {lang}"), start=1):
            if not os.path.exists(path):
                print(f"Warning: File not found - {path}")
                continue
                
            try:
                audio_sample, sr = librosa.load(path, sr=16000)
                inputs = processor(audio=audio_sample, return_tensors="pt", sampling_rate=sr)
                inputs = inputs.input_features.to(device)         
                outputs = model(inputs, decoder_input_ids=decoder_input_ids).last_hidden_state
                emb.append(outputs.detach().cpu().numpy())
                processed_count += 1
                
                # Print progress every 50 files
                if processed_count % 50 == 0:
                    print(f"\nProcessing {lang}: Completed {processed_count}/{len(temp)} audio files")
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        # Print final count
        print(f"\nProcessing {lang}: Completed {processed_count}/{len(temp)} audio files (Final)")
        
        if len(emb) == 0:
            print(f"Warning: No valid embeddings extracted for {lang}. Skipping...")
            continue
            
        if q == 'l2':
            emb = l2_norm(emb, lang, name)
        else:
            emb = temporal_mean(emb, lang, name)
