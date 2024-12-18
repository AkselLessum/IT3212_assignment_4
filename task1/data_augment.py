import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
import matplotlib.pyplot as plt
from PIL import Image
import random

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

# Add grayscale transformation for original images (only for making channel 1 again)
transform_original = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (before augmentation)
    transforms.ToTensor(),              # Convert to Tensor
])


def augment_data(train_data_path):
    # Transform data
    data_original = datasets.ImageFolder(root=train_data_path, transform=transform_original)
    data_augmented = datasets.ImageFolder(root=train_data_path, transform=transform_augment)

    # Combine the augmented data with the original to create new training set
    combined_data = ConcatDataset([data_original, data_augmented])

    # Get a random subset of the data (TOO MUCH DATAAAAAA)
    total_size = len(combined_data)
    subset_size = 15000
    indices = random.sample(range(total_size), subset_size)
    combined_data = Subset(combined_data, indices)

    return combined_data

def augment_test_data(test_data_path):
    test_data = datasets.ImageFolder(root=test_data_path, transform=transform_original)

    # Get a random subset of the data (TOO MUCH DATAAAAAA)
    total_size = len(test_data)
    subset_size = 3000
    indices = random.sample(range(total_size), subset_size)
    test_data = Subset(test_data, indices)

    return test_data


# TEST CODE DEPRICATED
'''
path = 'task1/seg_train/seg_train'

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

# Display augmented grayscale images
for i in range(len(images)):  # Display each image in the batch
    imshow(images[i])
'''