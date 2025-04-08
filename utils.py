import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from torchvision import transforms

# It transform a image to be input to the model
def transform(image_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

# Function for getting predicted labels and true labels from the validation set
def get_predictions_and_labels(model, val_loader, device,num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)  # For multi-label BCE
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    val_preds = np.concatenate(all_preds, axis=0)
    val_labels = np.concatenate(all_labels, axis=0)
    return val_preds, val_labels

# Finding optimal threshold for each class using F1 score
def find_optimal_thresholds(val_preds, val_labels, num_classes):
    best_thresholds = []
    for i in range(num_classes):
        f1_scores = []
        thresholds = np.linspace(0, 1, 100)
        for thresh in thresholds:
            bin_preds = (val_preds[:, i] > thresh).astype(int)
            f1 = f1_score(val_labels[:, i], bin_preds, zero_division=0)
            f1_scores.append(f1)
        best_thresh = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_thresh)
    return np.array(best_thresholds)

def plot_loss_curve(train_losses, window_size=200, filename="loss_curves.png"):
    """
    Plots the smoothed training loss curve.
    
    Parameters:
        train_losses (list or np.array): List of training loss values.
        window_size (int): The window size for smoothing (default is 200).
        filename (str): The filename where the plot will be saved.
    """
    train_losses_np = np.array(train_losses)
    smoothed_train_losses = np.convolve(train_losses_np, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(6, 4))
    plt.plot(smoothed_train_losses, label='Train Loss', color='blue', alpha=0.9, linewidth=2)
    plt.xlabel('Steps / Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    