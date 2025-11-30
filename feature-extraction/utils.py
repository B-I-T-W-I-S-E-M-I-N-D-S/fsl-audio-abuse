from tqdm.auto import tqdm
import numpy as np
import torch
import librosa

def temporal_mean(tensor_list, lang, name):
    vectors = []
    for tensor in tensor_list:
        tensor = np.squeeze(tensor, axis=0)

        vector = np.mean(tensor, axis=0)
        vectors.append(vector)
    
    if not vectors:
        print(f"Warning: No vectors to process for {lang} in {name} (temporal_mean).")
        return None

    result_array = np.vstack(vectors)
    np.save(f"./features/{name}/wav2vec-{lang}-emb", result_array)
    
    return result_array

def l2_norm(tensor_list, lang, name):
    vectors = []
    for tensor in tensor_list:
        tensor = np.squeeze(tensor, axis=0)
        
        l2_norms = np.sqrt(np.sum(np.square(tensor), axis=1, keepdims=True))
        normalized_tensor = tensor / l2_norms
        
        vector = np.mean(normalized_tensor, axis=0)
        vectors.append(vector)
    
    if not vectors:
        print(f"Warning: No vectors to process for {lang} in {name} (l2_norm).")
        return None

    result_array = np.vstack(vectors)
    np.save(f"./features/{name}-l2/wav2vec-{lang}-emb", result_array)
    
    return result_array

def feature_extractor(df, name, q, languages, processor, model, device):
    decoder_input_ids = torch.tensor([[1024, 1024]]).to(device)
    for lang in tqdm(languages):

        temp = list(df[df['language']==lang]['path_to_audio'])

        emb = []

        for path in tqdm(temp):

            audio_sample, sr = librosa.load(path, sr=16000)

            inputs = processor(audio=audio_sample, return_tensors="pt", sampling_rate=sr)
            inputs = inputs.input_features.to(device)         
            outputs = model(inputs,  decoder_input_ids=decoder_input_ids).last_hidden_state

            emb.append(outputs.detach().cpu().numpy())
        if q=='l2':
            emb = l2_norm(emb, lang, name)
        else:
            emb = temporal_mean(emb, lang, name)