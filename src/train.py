import torch 
from torch import nn
from torchvision import transforms,models
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataset import ChestXRayDataset
from utils import *
from model import ResNet50   # importing ResNet50 from model.py

# Loading the metadata from kaggle dataset
# This is a placeholder path. You should replace it with the actual path to your dataset.
csv_path = "/kaggle/input/data/Data_Entry_2017.csv"  
df = pd.read_csv(csv_path)



# Train and validation dataset and dataloader
train_df,val_df=train_test_split(df,test_size=0.01,random_state=43)
data_folder = "/kaggle/input/data"       # This is a the path to the kaggle dataset i used for training
train_dataset = ChestXRayDataset(train_df,data_folder,transform)
val_dataset = ChestXRayDataset(val_df,data_folder,transform)
BATCH_SIZE=32
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    
# Model instantiation
num_labels=14
model = ResNet50()
pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
pretrained_state_dict = pretrained_model.state_dict()
model.load_state_dict(pretrained_state_dict)

# Replacing the final layer with a new one for our specific task
in_features = model.fc.in_features
model.fc = nn.Linear(in_features,num_labels)  # replacing final layer

# Unfreezing last two layers and the output layer
for param in model.parameters():
    param.requires_grad=False
for param in model.layer2.parameters():
    param.requires_grad = True
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

# Device configuration
device='cpu'
if torch.cuda.is_available():
    device='cuda'
model.to(device);

# Loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
criterion = nn.BCEWithLogitsLoss()

# Training and validation loop
lr = 1e-5
num_epochs = 5  # total epochs are 5
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    step = 0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in loop:
        step += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            val_logits = model(images)
            val_loss += criterion(val_logits, labels).item()
            # Optional: Evaluate binary accuracy
            preds = torch.sigmoid(val_logits)
            predicted = (preds > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(avg_val_loss)

    print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
    scheduler.step(avg_val_loss)
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("Saved Best Model")
        
