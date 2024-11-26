from main import *
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load data
X_train, X_test, y_train, y_test = main('task1/seg_train/seg_train', 'task1/seg_test/seg_test')

# Initialize classifiers
dc_classifier = DecisionTreeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb_classifier = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss' ,random_state=42)
lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)

# List of classifiers
classifiers = {
    "Decision Tree": dc_classifier,
    "Random Forest": rf_classifier,
    #"Gradient Boosting": gb_classifier
    "Light GBM": lgbm_classifier,
    "XGBoost": xgb_classifier
}

# Train all classifiers and collect their predictions
predictions = []
for name, clf in classifiers.items():
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"{name} training time: {training_time:.4f} seconds")

    # Make predictions
    start_time = time.time()
    y_pred = clf.predict(X_test)
    prediction_time = time.time() - start_time
    print(f"{name} prediction time: {prediction_time:.4f} seconds")

    # Append predictions
    predictions.append(y_pred)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Combine predictions (majority voting)
predictions = np.array(predictions)
combined_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# Evaluate combined predictions
accuracy = accuracy_score(y_test, combined_predictions)
print(f"\nCombined Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report for Combined Model:")
print(classification_report(y_test, combined_predictions))

# Confusion matrix for combined predictions
class_names = ['Building', 'Forests', 'Glacier', 'Mountain', 'Sea', 'Street']
cm = confusion_matrix(y_test, combined_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Combined Model - Confusion Matrix')
plt.show()




"""""

--- Training Decision Tree ---
Decision Tree training time: 16.1599 seconds
Decision Tree prediction time: 0.0156 seconds
Decision Tree Accuracy: 49.77%
Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.48      0.49       437
           1       0.71      0.77      0.74       474
           2       0.38      0.35      0.36       553
           3       0.38      0.38      0.38       525
           4       0.49      0.49      0.49       510
           5       0.54      0.55      0.55       501

    accuracy                           0.50      3000
   macro avg       0.50      0.50      0.50      3000
weighted avg       0.49      0.50      0.49      3000


--- Training Random Forest ---
Random Forest training time: 46.7561 seconds
Random Forest prediction time: 0.1080 seconds
Random Forest Accuracy: 59.63%
Classification Report:
              precision    recall  f1-score   support

           0       0.74      0.51      0.60       437
           1       0.78      0.92      0.84       474
           2       0.48      0.43      0.45       553
           3       0.41      0.51      0.46       525
           4       0.56      0.52      0.54       510
           5       0.71      0.71      0.71       501

    accuracy                           0.60      3000
   macro avg       0.61      0.60      0.60      3000
weighted avg       0.60      0.60      0.59      3000


--- Training Gradient Boosting ---
Gradient Boosting training time: 2072.0973 seconds
Gradient Boosting prediction time: 0.0453 seconds
Gradient Boosting Accuracy: 64.30%
Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.66      0.70       437
           1       0.77      0.92      0.84       474
           2       0.52      0.52      0.52       553
           3       0.48      0.51      0.49       525
           4       0.62      0.58      0.60       510
           5       0.75      0.70      0.73       501

    accuracy                           0.64      3000
   macro avg       0.65      0.65      0.65      3000
weighted avg       0.64      0.64      0.64      3000


Combined Model Accuracy: 62.93%
Classification Report for Combined Model:
              precision    recall  f1-score   support

           0       0.67      0.68      0.67       437
           1       0.77      0.92      0.83       474
           2       0.48      0.51      0.50       553
           3       0.48      0.47      0.48       525
           4       0.65      0.54      0.59       510
           5       0.77      0.70      0.73       501

    accuracy                           0.63      3000
   macro avg       0.63      0.64      0.63      3000
weighted avg       0.63      0.63      0.63      3000


run 2:


"""""