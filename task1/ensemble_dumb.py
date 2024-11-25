# Simple ensemble, 3 models and boosting

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import main
import numpy as np
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode
from sklearn.metrics import accuracy_score

#TODO: Fix this garbage
class ensemble_Dumb:
    
    def __init__(self):
        self.model1 = DecisionTreeClassifier(random_state=42)
        self.model2 = KNeighborsClassifier()
        self.model3 = LogisticRegression(random_state=42)
        self.model4 = XGBClassifier(random_state=42)
        self.finalpred = None
        self.X_train, self.X_test, self.y_train, self.y_test = main.main('task1/seg_train/seg_train', 'task1/seg_test/seg_test')

    
    def do_ensemble(self):
        self.model1.fit(self.X_train, self.y_train)
        self.model2.fit(self.X_train, self.y_train)
        self.model3.fit(self.X_train, self.y_train)
        self.model4.fit(self.X_train, self.y_train)
        
        pred1 = self.model1.predict(self.X_test)  
        pred2 = self.model2.predict(self.X_test)
        pred3 = self.model3.predict(self.X_test)
        pred4 = self.model4.predict(self.X_test)
        
        self.finalpred = np.array([])
        for i in range(len(self.X_test)):
            self.finalpred = np.append(self.finalpred, mode([pred1[i], pred2[i], pred3[i], pred4[i]])[0][0])
            
        
        accuracy = accuracy_score(self.y_test, self.finalpred)
        print(f'Accuracy: {accuracy * 100:.2f}%')
        
        return self.finalpred    
        
    def plot(self):
        if self.finalpred is None or self.y_test is None:
            print("Please run do_ensemble() first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.y_test, label='True Values')
        plt.plot(self.finalpred, label='Predicted Values', linestyle='--')
        plt.legend()
        plt.title('Ensemble Model Predictions vs True Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.show()
        
        class_names = ['Building', 'Forests', 'Glacier', 'Mountain', 'Sea', 'Street']
        cm = confusion_matrix(self.y_test, self.finalpred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.show()
        
        
ensemble = ensemble_Dumb()

ensemble.do_ensemble()

ensemble.plot()