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

--- Training Decision Tree ---
Decision Tree training time: 17.0043 seconds
Decision Tree prediction time: 0.0097 seconds
Decision Tree Accuracy: 48.33%
Classification Report:
              precision    recall  f1-score   support

           0       0.46      0.47      0.47       437
           1       0.70      0.75      0.72       474
           2       0.38      0.34      0.36       553
           3       0.36      0.38      0.37       525
           4       0.46      0.44      0.45       510
           5       0.54      0.55      0.55       501

    accuracy                           0.48      3000
   macro avg       0.48      0.49      0.49      3000
weighted avg       0.48      0.48      0.48      3000


--- Training Random Forest ---
Random Forest training time: 48.2772 seconds
Random Forest prediction time: 0.1028 seconds
Random Forest Accuracy: 59.20%
Classification Report:
              precision    recall  f1-score   support

           0       0.77      0.51      0.61       437
           1       0.77      0.90      0.83       474
           2       0.48      0.44      0.46       553
           3       0.40      0.52      0.45       525
           4       0.57      0.51      0.54       510
           5       0.69      0.70      0.70       501

    accuracy                           0.59      3000
   macro avg       0.61      0.60      0.60      3000
weighted avg       0.60      0.59      0.59      3000


--- Training Light GBM ---
[LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.063859 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 153000
[LightGBM] [Info] Number of data points in the train set: 15000, number of used features: 600
[LightGBM] [Info] Start training from score -1.867345
[LightGBM] [Info] Start training from score -1.807889
[LightGBM] [Info] Start training from score -1.781809
[LightGBM] [Info] Start training from score -1.715169
[LightGBM] [Info] Start training from score -1.843053
[LightGBM] [Info] Start training from score -1.743732
Light GBM training time: 20.6478 seconds
Light GBM prediction time: 0.0247 seconds
Light GBM Accuracy: 66.43%
Classification Report:
              precision    recall  f1-score   support

           0       0.77      0.69      0.73       437
           1       0.84      0.93      0.88       474
           2       0.54      0.52      0.53       553
           3       0.47      0.51      0.49       525
           4       0.63      0.61      0.62       510
           5       0.78      0.77      0.77       501

    accuracy                           0.66      3000
   macro avg       0.67      0.67      0.67      3000
weighted avg       0.67      0.66      0.66      3000


--- Training XGBoost ---
C:\Users\morom\Documents\git repos\datadrevet\IT3212_assignment_4\.venv\Lib\site-packages\xgboost\core.py:158: UserWarning: [11:25:37] WARNING: C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0ed59c031377d09b8-1\xgboost\xgboost-ci-windows\src\learner.cc:740:
Parameters: { "use_label_encoder" } are not used.

  warnings.warn(smsg, UserWarning)
XGBoost training time: 59.2583 seconds
XGBoost prediction time: 0.0147 seconds
XGBoost Accuracy: 66.40%
Classification Report:
              precision    recall  f1-score   support

           0       0.74      0.68      0.71       437
           1       0.82      0.94      0.88       474
           2       0.55      0.53      0.54       553
           3       0.49      0.54      0.52       525
           4       0.65      0.62      0.63       510
           5       0.77      0.72      0.74       501

    accuracy                           0.66      3000
   macro avg       0.67      0.67      0.67      3000
weighted avg       0.66      0.66      0.66      3000


Combined Model Accuracy: 66.03%
Classification Report for Combined Model:
              precision    recall  f1-score   support

           0       0.75      0.72      0.74       437
           1       0.80      0.94      0.87       474
           2       0.53      0.57      0.55       553
           3       0.48      0.52      0.50       525
           4       0.66      0.55      0.60       510
           5       0.81      0.70      0.75       501

    accuracy                           0.66      3000
   macro avg       0.67      0.67      0.67      3000
weighted avg       0.66      0.66      0.66      3000

"""""