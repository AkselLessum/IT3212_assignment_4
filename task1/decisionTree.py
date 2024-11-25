from main import *
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = main('task1/seg_train/seg_train', 'task1/seg_test/seg_test')

# Initialize the Random Forest Classifier
dc_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier
start_time = time.time()
dc_classifier.fit(X_train, y_train)
end_time = time.time()

prediction_time = end_time - start_time
print(f'Prediction time: {prediction_time:.4f} seconds')

# Make predictions
start_time = time.time()
y_pred = dc_classifier.predict(X_test)
end_time = time.time()

prediction_time = end_time - start_time
print(f'Prediction time: {prediction_time:.4f} seconds')

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


class_names = ['Building', 'Forests', 'Glacier', 'Mountain', 'Sea', 'Street']
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.show()