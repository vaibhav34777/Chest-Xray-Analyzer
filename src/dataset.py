from torchvision import transforms,models
from torch.utils.data import Dataset,DataLoader
from glob import glob
import os
import pandas as pd
import numpy as np
from PIL import Image

# Disease List the model will predict
disease_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
                'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
# Disease Dictionary
disease_map = {disease: i for i, disease in enumerate(disease_list)}

# Encoding the labels to one-hot vectors
def encode_labels(labels):
    one_hot=np.zeros(len(disease_list),dtype=np.float32)
    for disease in labels.split('|'):
        if disease in disease_list:
            one_hot[disease_map[disease]]=1.0
    return one_hot

# Dataset class for loading images and labels 
class ChestXRayDataset(Dataset):
    def __init__(self, df, data_folder, transform=None):
        self.df = df
        self.transform = transform
        # Collect image paths
        self.image_paths = {}
        image_folders = sorted(
            [os.path.join(data_folder, d) for d in os.listdir(data_folder) if d.startswith("images_")],
            key=lambda x: int(os.path.basename(x).split('_')[-1])
        )
        for folder in image_folders:
            for path in glob(os.path.join(folder, "images", "*.[pj][np]g")):
                self.image_paths[os.path.basename(path).lower()] = path  # Store lowercase filenames
                
        self.df["Image Index"] = self.df["Image Index"].str.lower()

        def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["Image Index"]
        label = encode_labels(row["Finding Labels"])  # Convert label to tensor
        # Load image
        img_path = self.image_paths.get(filename)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
    
