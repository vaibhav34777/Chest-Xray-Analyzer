import torch 
import streamlit as st
from torch import nn
from torchvision import transforms,models
from torch.utils.data import Dataset,DataLoader
from glob import glob
import os
import pandas as pd
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import gdown
def download_model():
    model_path = "model/model.pth"
    if not os.path.exists(model_path):
        os.makedirs("model", exist_ok=True)
        url = "https://huggingface.co/imvaibhavrana/chest-xray-analyzer/resolve/main/model.pt?download=true"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return model_path
device = 'cpu'
if torch.cuda.is_available():
    device='cuda'

disease_list = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 
                'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

disease_map = {disease: i for i, disease in enumerate(disease_list)}
def encode_labels(labels):
    one_hot=np.zeros(len(disease_list),dtype=np.float32)
    for disease in labels.split('|'):
        if disease in disease_list:
            one_hot[disease_map[disease]]=1.0
    return one_hot

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# MODEL ARCHITECTURE
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

model_path = download_model()
model = ResNet50(num_classes=14).to(device)
state_dict = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
model.load_state_dict(state_dict)
model.eval()
optimal_thresholds = torch.tensor([0.18181818, 0.52525253, 0.1010101 , 0.06060606, 0.3030303 ,
       0.12121212, 0.1010101 , 0.3       , 0.16161616, 0.19191919,
       0.13131313, 0.11111111, 0.03030303, 0.14141414])

    
def inference(img,transform,device):
    input = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input)
        preds = torch.sigmoid(logits)
    predicted = (preds > optimal_thresholds).float()
    predicted_disease = [disease_list[i] for i, val in enumerate(predicted[0]) if val.item() == 1.0]
    return predicted_disease


st.set_page_config(page_title="Chest X-ray Disease Analyzer", layout="centered")
st.title("🩺 Chest X-ray Disease Analyzer")
st.markdown("Upload a chest X-ray image and click **Predict** to analyze potential thoracic diseases.")
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300, use_container_width=False)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            predicted_diseases = inference(image, transform, device)

        st.subheader("Predicted Disease(s):")

        if not predicted_diseases:
            st.markdown("<h2 style='font-size:24px'>✅ No Finding</h2>", unsafe_allow_html=True)
        else:
            for disease in predicted_diseases:
                st.markdown(f"<h2 style='font-size:24px'>- <strong>{disease}</strong></h2>", unsafe_allow_html=True)



