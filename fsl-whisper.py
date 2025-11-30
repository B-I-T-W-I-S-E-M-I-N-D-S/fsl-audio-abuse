from utils.utils import maml_adima
from data.download import download_dataset
import torch
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(42)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.manual_seed(42)
else:
    device = torch.device("cpu")
print(f"Running on {device}.")

os.makedirs("./results/Temporal-Mean/whisper", exist_ok=True)
os.makedirs("./results/L2-Norm/whisper", exist_ok=True)

download_dataset()

shot_sizes = [50, 100, 150, 200]

path = "./data/whisper-feat.csv"
print("---Whisper Few Shot---")
for shot in shot_sizes:
    maml_adima(128, shot, path, 768, 128, 2, 0.001, 0.001, 150, "whisper", "Temporal-Mean", device=device)

path = "./data/whisper-l2-feats.csv"
print("---Whisper L2 Few Shot---")
for shot in shot_sizes:
    maml_adima(128, shot, path, 768, 128, 2, 0.001, 0.001, 150, "whisper", "L2-Norm", device=device)