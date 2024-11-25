#TODO, ensemble learning
# remember: Bagging, boosting, stacking

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers, optimizers
from sklearn.metrics import accuracy_score
import matplotlib.image as mpimg
import math
import os
import warnings
warnings.filterwarnings('ignore')

#Borrowing quite a bit from lecture 7 and 8

class ensemble:
    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier()
    model3 = LogisticRegression()
    
    train_directory = 'task1/seg_train/seg_train'
    test_directory = 'task1/seg_test/seg_test'
    
    
    def load_data(self, train_data_path, test_data_path):
        pass
    
    #Layer 1
    def bagging(self, X_train, y_train, X_test, y_test): #TODO
        bagging_model = RandomForestClassifier(n_estimators=100, random_state=42)
        bagging_model.fit(X_train, y_train)
        prediction = bagging_model.predict(X_test)
        print('Accuracy:', accuracy_score(y_test, prediction))
        return bagging_model
    
    #use embeddings from VGG16?
    # https://www.kaggle.com/code/arshadali12/image-classification-vgg16-fine-tuned
    def boosting(self): #TODO
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 1))
        def preprocess_input(x):
            return np.repeat(x, 3, axis=-1)
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    
    #I don't fully understand this
    def transer_learning(self):
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 1))
        def preprocess_input(x):
            return np.repeat(x, 3, axis=-1)
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        def extract_features(directory, sample_count):
            features = np.zeros(shape=(sample_count, 4, 4, 512))
            labels = np.zeros(shape=(sample_count))
            generator = datagen.flow_from_directory(
                directory,
                target_size=(150, 150),
                batch_size=20,
                class_mode='binary')
            i = 0
            for inputs_batch, labels_batch in generator:
                features_batch = conv_base.predict(inputs_batch)
                features[i * 20 : (i + 1) * 20] = features_batch
                labels[i * 20 : (i + 1) * 20] = labels_batch
                i += 1
                if i * 20 >= sample_count:
                    break
            return features, labels
        train_features, train_labels = extract_features(self.train_directory, sample_count=2000)
        test_features, test_labels = extract_features(self.test_directory, sample_count=1000)
        return train_features, train_labels, test_features, test_labels
    
    
    #Layer 2
    # This falls out if too computationally expensive
    # From lecture 7
    def stacking(model, train, y, test, n_fold): #TODO
        folds = StratifiedKFold(n_splits=n_fold, random_state=42)
        test_pred = np.empty((test.shape[0], 1), float)
        train_pred = np.empty((0, 1), float)
        for train_indicies, val_indicies in folds.split(train, y.values):
            x_train, x_val = train.iloc[train_indicies], train.iloc[val_indicies]
            y_train, y_val = y.iloc[train_indicies], y.iloc[val_indicies]
            model.fit(X=x_train, y=y_train)
            train_pred = np.append(train_pred, model.predict(x_val))
            test_pred = np.append(test_pred, model.predict(test))   
        return test_pred.reshape(-1, 1), train_pred
    
    def ensemble(self): #TODO
        pass
    
    def __init__(self):
        pass
