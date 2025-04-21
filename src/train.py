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

# Model definition
 # 3x3 convolution with padding
def conv3x3(in_planes,out_planes,stride=1,dilation=1):  
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,
                     stride=stride,padding=dilation,
                     bias=False,dilation=dilation,)
 # 1x1 convolution
def conv1x1(in_planes,out_planes,stride=1):                     
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self,in_planes,planes,stride=1,dilation=1):  
        super().__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_planes,planes)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = conv3x3(planes,planes,stride,dilation)
        self.bn2 = self.norm_layer(planes)
        self.conv3 = conv1x1(planes,self.expansion*planes)
        self.bn3 = self.norm_layer(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if self.stride!=1 or in_planes!=planes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=self.stride,bias=False),
                nn.BatchNorm2d(planes*self.expansion),
            )
        else:
            self.downsample = None
    def forward(self,x):
        x_skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            x_skip = self.downsample(x_skip)
        x += x_skip
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()
        self.layers = [3,4,6,3]   # number of blocks in each layer
        self.expansion = 4
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3,self.in_planes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(Bottleneck,64,self.layers[0])
        self.layer2 = self._make_layer(Bottleneck,128,self.layers[1],stride=2)
        self.layer3 = self._make_layer(Bottleneck,256,self.layers[2],stride=2)
        self.layer4 = self._make_layer(Bottleneck,512,self.layers[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,block,planes,blocks,stride=1,dilation=1):
        layers=[]
        layers.append(block(self.in_planes,planes,stride,dilation))
        self.in_planes = planes * block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_planes,planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
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
        
