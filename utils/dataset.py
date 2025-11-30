import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """
    Torch Dataset
    """
    def __init__(self, df, ):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio = torch.from_numpy(self.df.iloc[idx].drop(['abuse']).values).float()
        label = int(self.df.iloc[idx]['abuse'])

        return audio, label