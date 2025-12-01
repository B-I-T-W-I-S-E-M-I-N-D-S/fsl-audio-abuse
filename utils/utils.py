import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import gdown

from .MAML import MAML
from .dataset import AudioDataset
from .model import SimpleNN


def stratified_sample(df, shot):
    dfs = []
    for abuse in df['abuse'].unique():
        class_df = df[df['abuse'] == abuse]
        dfs.append(class_df.sample(n=shot//2, random_state=1))
    return pd.concat(dfs)

def train_maml(train_loader, maml, n_epochs, device):
    steps = 0
    for epoch in range(n_epochs):
        for features, labels in train_loader:
            steps = steps+1
            features, labels = features.to(device), labels.to(device)
            # Perform inner update
            updated_params = maml.inner_update(features, labels)

            # Perform second forward pass with updated parameters
            updated_outputs = maml.forward(features, updated_params)
            updated_loss = F.cross_entropy(updated_outputs, labels)
            

            # Perform outer update
            maml.outer_update(updated_loss)
            
        if((epoch+1)%50==0):
            print(f"Step={steps}, Epoch {epoch + 1}/{n_epochs} - Loss: {updated_loss.item():.4f}")

def evaluate_model(test_loaders, maml, device):
    maml.model.eval()
    results = {}
    with torch.no_grad():
        for lang, test_loader in test_loaders.items():
            all_preds = []
            all_labels = []
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = maml.forward(features)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro')
            results[lang] = {'accuracy': accuracy, 'f1': f1}
            
#     res = pd.DataFrame(results).T
#     res.to_csv(f"")
    return pd.DataFrame(results).T

def maml_adima(batch_size, shot, path, input_dim, hidden_dim, output_dim, lr_inner, lr_outer, n_epochs, feat, norm, device):
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    
    df.abuse = np.where(df.abuse.values == 'Yes', 1, 0)
    languages = df['language'].unique()


#     logging.info(f"Hyperparameters: batch_size={batch_size}, hidden_dim={hidden_dim}, lr_inner={lr_inner}, lr_outer={lr_outer}, n_epochs={n_epochs}, shot={shot}")

    train_df = df[df['train_test']=='train'].drop(['train_test'], axis=1)
    train_list = []
    for lang in languages:
        temp = train_df[train_df['language']==lang].drop(['language'], axis=1).reset_index().drop(['index'], axis=1)
        temp = stratified_sample(temp, shot).reset_index().drop(['index'], axis=1)
        train_list.append(temp)
    train = pd.concat(train_list)
    train_loader= DataLoader(AudioDataset(train), batch_size=batch_size, shuffle=True)

    test_loaders = {}
    test_df = df[df['train_test']=='test'].drop(['train_test'], axis=1)
    for lang in languages:
        temp_df = test_df[test_df['language']==lang].drop(['language'], axis=1)
        test_dataset = AudioDataset(temp_df)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loaders[lang] = test_loader


    model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
    maml = MAML(model, lr_inner, lr_outer, device)
    
    print(f"---Shot Size: {shot}---")

    # FIX: Add device parameter here
    train_maml(train_loader, maml, n_epochs, device)

    # FIX: Add device parameter here too
    evaluation_results = evaluate_model(test_loaders, maml, device)
    
    evaluation_results.to_csv(f"./results/{norm}/{feat}/{shot}-result.csv")

    print(evaluation_results)
