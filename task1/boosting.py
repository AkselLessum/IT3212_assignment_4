import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from main import *

X_train, X_test, y_train, y_test = main('task1/seg_train/seg_train', 'task1/seg_test/seg_test')

# Initialize the XGBoost Classifier with tuned parameters
xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',   # Multi-class classification
    num_class=6,                # Number of classes
    max_depth=10,               # Depth of the trees
    learning_rate=0.1,          # Learning rate
    n_estimators=200,           # Number of trees
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,       # Feature sampling
    eval_metric='mlogloss',     # Multi-class log-loss
)

# Train the model
xgb_clf.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_clf.predict(X_test)


# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
