from data_augment import augment_data
from skimage.feature import hog
from skimage import exposure
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Get the training and test data
train_path = 'CNN/seg_train/seg_train'
train_data = augment_data(train_path)


# Perform HOG (Histogram of gradients) and PCA
# Expects data as pytorch datasets
def select_extract(data):
    hog_features = []
    labels = []

    for image, label in tqdm(data, desc="Extracting HOG features"):
        features = get_hog_features(image)
        hog_features.append(features)
        labels.append(label)
    
    hog_features = np.array(hog_features)
    labels = np.array(labels)


def get_hog_features(image):
    # Need to make the tensor image data into numpy arrays for HOG to work
    #image_np = image.numpy().transpose((1, 2, 0))
    image_np = image.permute(1, 2, 0)
    image_np = np.squeeze(image_np)
    
    # Get HOG features
    features, hog_img = hog(image_np, pixels_per_cell=(10,10), cells_per_block=(2,2), visualize=True)
    
    '''# Rescale histogram for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    plt.show()'''
    
    return features

select_extract(train_data)