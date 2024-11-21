from data_select_extract import *
from data_augment import *
import numpy as np

# Function to get training and tested formatted with labels
def main(train_data_path, test_data_path):
    print("TRAINING DATA")
    train_data = augment_data(train_data_path)
    X_train, y_train = select_extract(train_data)
    print("------------------------------------------")
    print("TEST DATA")
    test_data = augment_test_data(test_data_path)
    X_test, y_test = select_extract(test_data)

    print("------------------------------------------")
    print("PERFORMING PCA ON TRAINING AND TEST DATA")
    X_train, X_test = get_pca_features(X_train, X_test, 600)
    print("FINISHED PREFORMINC PCA")
    print("------------------------------------------")
    return X_train, X_test, y_train, y_test


#main('task1/seg_train/seg_train', 'task1/seg_test/seg_test')