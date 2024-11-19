import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Code for making images greyscale and data augmentation
# Uses online augmentation (pytorch) so augmented images are not saved
# Also does "random" augmentations each iteration

# Set of augmentations
# Define the augmentations with no change in image size
# Updated augmentations for grayscale images of size 150x150
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),    # Random vertical flip
    transforms.RandomRotation(30),      # Random rotation between -30 and +30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random brightness/contrast adjustments
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),  # Random Gaussian Blur
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (after augmentations)
    transforms.ToTensor(),              # Convert to Tensor
])


def augment_data(train_data_path):
    return datasets.ImageFolder(root=train_data_path, transform=transform_augment)


# TEST CODE

path = 'CNN/seg_train/seg_train'

train_dataset = augment_data(path)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

def imshow(tensor):
    image = tensor.numpy().squeeze(0)  # Remove the channel dimension
    plt.imshow(image, cmap='gray')    # Display as grayscale
    plt.axis('off')
    plt.show()

# Visualize one batch of augmented images
data_iter = iter(train_loader)
images, labels = next(data_iter)
class_names = train_dataset.classes

# Display augmented grayscale images
for i in range(len(images)):  # Display each image in the batch
    imshow(images[i])
    print(f"Class: {class_names[labels[i].item()]}")  # Print the corresponding class name
