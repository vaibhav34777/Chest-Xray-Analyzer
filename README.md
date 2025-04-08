# Chest X-ray Disease Analyzer

A deep learning-based multi-label classification model for detecting thoracic diseases from chest X-ray images using a fine-tuned ResNet-50 architecture.

## Overview

This project leverages the NIH Chest X-ray dataset and a fine-tuned ResNet-50 model to predict the presence of 14 thoracic diseases from chest radiographs. The model is trained using a combination of transfer learning, learning rate scheduling, and threshold optimization to achieve high performance on multi-label classification.

## Data Preprocessing

The dataset used is the NIH Chest X-ray Dataset, available on Kaggle, which contains 112,000 X-ray images. Each image can be associated with zero or more labels from a list of 14 diseases. Due to the multi-label nature, the output of the model consists of 14 units.

The 14 disease classes are:
- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural_Thickening
- Hernia

The preprocessing steps include:
- Resizing each image to 224x224 pixels.
- Converting the image to a PyTorch tensor.
- Using a custom dataset class that loads images and one-hot encodes their corresponding labels.
- Utilizing PyTorch DataLoader to efficiently batch and shuffle the training and validation data.

## Model Architecture and Training

- A ResNet-50 model was used as the backbone.
- The final fully connected layer was replaced to output 14 logits corresponding to each of the disease classes.
- Only the `layer2`, `layer3`, and `layer4` of the ResNet-50 were unfrozen to allow fine-tuning, while the earlier layers remained frozen.
- The model was trained using the `BCEWithLogitsLoss` loss function since the final layer outputs raw logits suitable for multi-label classification.
- A low learning rate of `1e-5` was used to avoid disrupting the pretrained weights significantly.
- A learning rate scheduler was implemented to reduce the learning rate when the validation loss plateaued.
- The model was trained for 5 epochs, reaching a validation loss of 0.15, a training loss of 0.12, and an overall accuracy of 94.7%.

## Threshold Optimization

- Instead of applying a fixed threshold (e.g., 0.5) across all classes, F1 scores were computed across a range of threshold values for each class individually.
- The optimal threshold for each class was selected based on the value that provided the best F1 score, significantly improving the performance of the multi-label classification.

## Inference

- During inference, the trained model uses the precomputed optimal thresholds for each class to determine which diseases are present in a given image.
- The image undergoes the same preprocessing steps and is passed through the model to generate logits.
- A sigmoid activation is applied, and values above the respective thresholds are selected as the predicted disease labels.

## Results and Evaluation

- The training process and model evaluation are showcased in a Kaggle notebook.
- Several example predictions are included, comparing the model’s output against the actual ground truth labels to visually assess its performance.

📌 **Kaggle Notebook Link:** https://www.kaggle.com/code/vaibhav1908/chestxrayanalysis

## Future Improvements

- Apply data augmentation techniques such as rotation, flipping, and contrast adjustment to improve model generalization.
- Experiment with other model architectures or ensemble methods for potentially better performance.
- Integrate Grad-CAM or other visualization tools to improve explainability.

## Acknowledgements

- **NIH Chest X-ray Dataset**: For providing the high-quality annotated dataset.
- **Kaggle**: For hosting the dataset and providing the computational resources.
- **Torchvision**: For the pretrained ResNet-50 model used as the base architecture.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

