from utils import *
from dataset import ChestXRayDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
from model import ResNet50

num_classes = 14
disease_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
                'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

 # Getting the dataloaders for inference
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

device='cpu'
if torch.cuda.is_available():
    device='cuda'

# Getting the model for inference
model = ResNet50(num_classes=14)
model.to(device)
model.load_state_dict(torch.load('/kaggle/input/pytorch-model/pytorch/default/1/best_model (1).pt'))  # Load the fine tuned model 

# Getting the best thresholds for inference using F1 score
val_preds, val_labels = get_predictions_and_labels(model, val_loader, device, num_classes)
optimal_thresholds = find_optimal_thresholds(val_preds, val_labels, num_classes)
optimal_thresholds[7] = 0.2  # setting high for hernia explicitly after observation

optimal_thresholds=torch.tensor(optimal_thresholds)
optimal_thresholds=optimal_thresholds.to(device)
def inference(image_path,transform,device):
    img = Image.open(image_path).convert('RGB')
    input = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input)
        preds = torch.sigmoid(logits)
    predicted = (preds > optimal_thresholds).float()
    predicted_disease = [disease_list[i] for i, val in enumerate(predicted[0]) if val.item() == 1.0]
    return predicted_disease

pth = "/kaggle/input/data/"
# These are the paths to the images in kaggle dataset for testing the model
# You can replace them with your own images
img_path1 = f"{pth}images_006/images/00011558_013.png"
img_path2 = f"{pth}images_008/images/00016051_016.png"
img_path3 = f"{pth}images_004/images/00006585_011.png"
img_list = [img_path1, img_path2, img_path3]

for n, i in enumerate(img_list):
    image = Image.open(i)
    plt.figure(figsize=(5, 4))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    predicted_disease = inference(i, transform, device)
    if len(predicted_disease) == 0:
        print(f'Predicted Diseases in this image: No Finding')
    else:
        print(f'Predicted Disease in this image: {predicted_disease}')
    image_path = os.path.basename(i).lower()
    diseases = df[df['Image Index'] == image_path]['Finding Labels'].values[0]
    print("Origninal Diseases:", diseases)
