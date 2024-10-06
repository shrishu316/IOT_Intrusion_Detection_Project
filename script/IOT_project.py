import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("D:\\IOT_project\\IOT_data.csv")

print(data.shape)
print(data.info())
print(data.head())
print(data.describe())
print(data.isnull().sum())

# You can drop or fill missing values as needed
data = data.fillna(method='ffill')  # Example of filling missing values

# Check unique labels in the 'label' column
unique_labels = data['label'].unique()
print("Unique labels in the dataset:")
print(unique_labels)

# Count of unique labels
label_counts = data['label'].value_counts()
print("\nCount of each label:")
print(label_counts)

# Define the labels to keep
labels_to_keep = [
    'DDoS-RSTFINFlood', 
    'DoS-TCP_Flood', 
    'DDoS-ICMP_Flood', 
    'DoS-UDP_Flood', 
    'DoS-SYN_Flood', 
    'Mirai-greeth_flood', 
    'DDoS-SynonymousIP_Flood', 
    'DDoS-PSHACK_Flood', 
    'DDoS-HTTP_Flood', 
    'DDoS-SLOWLORIS', 
    'Backdoor_Malware', 
    'CommandInjection', 
    'SqlInjection', 
    'DictionaryBruteForce', 
    'BrowserHijacking', 
    'Recon-PortScan', 
    'Recon-OSScan', 
    'Recon-HostDiscovery', 
    'Recon-PingSweep', 
    'BenignTraffic'
]

# Filter the dataset to keep only the selected labels
filtered_data = data[data['label'].isin(labels_to_keep)]

# Display the filtered dataset
print("\nFiltered dataset:")
print(filtered_data.head())

# Check the counts of the filtered labels
filtered_label_counts = filtered_data['label'].value_counts()
print("\nCount of filtered labels:")
print(filtered_label_counts)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Count plot of filtered labels
plt.figure(figsize=(12, 6))
# Set alpha for transparency (0 is fully transparent, 1 is fully opaque)
sns.countplot(y='label', data=filtered_data, order=filtered_data['label'].value_counts().index, alpha=0.8)

# Set title and labels with a contrasting color
plt.title('Distribution of Filtered Labels', fontsize=16, color='black')
plt.xlabel('Count', fontsize=14, color='black')
plt.ylabel('Label', fontsize=14, color='black')

# Set the x and y ticks to have black color
plt.xticks(color='black')
plt.yticks(color='black')

# Add gridlines with a faint color for better visibility
plt.grid(color='gray', linestyle='--', linewidth=0.5)

plt.savefig('D:\\IOT_project\\Distribution_of_Filtered_Labels.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Check for missing values
missing_values = filtered_data.isnull().sum()
print(missing_values)

from sklearn.preprocessing import StandardScaler

# Separate features and labels
X = filtered_data.drop('label', axis=1)
y = filtered_data['label']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier()

# Train the model
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
from sklearn.metrics import classification_report, confusion_matrix
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

from sklearn.svm import SVC

# Initialize the model
svm_model = SVC()

# Train the model
svm_model.fit(X_train, y_train)

# Predict
y_pred_svm = svm_model.predict(X_test)

# Evaluate
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

from sklearn.linear_model import LogisticRegression

# Initialize the model
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test)

# Evaluate
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Confusion matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Random Forest')
plt.savefig('D:\\IOT_project\\Confusion Matrix for Random Forest.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Convert the labels to binary: 'BenignTraffic' vs. All Attacks
y_train_binary = y_train.apply(lambda x: 1 if x == 'BenignTraffic' else 0)
y_test_binary = y_test.apply(lambda x: 1 if x == 'BenignTraffic' else 0)

# Train your model (e.g., Random Forest) with binary labels
rf_model.fit(X_train, y_train_binary)

# Predict probabilities for the test set
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (BenignTraffic)

# Compute ROC curve and AUC for Random Forest
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)  # Using binary labels
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('D:\\IOT_project\\Receiver Operating Characteristic (ROC) Curve', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_proba_multiclass[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {classes[i]} (area = {roc_auc:.2f})')

# Plot the diagonal line
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')

# Adjust legend position outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
plt.savefig('D:\\IOT_project\\Receiver Operating Characteristic (ROC) Curve for Multiclass.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

from sklearn.metrics import classification_report

# Predict class labels for the test set
y_pred_multiclass = rf_model_ovr.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_binary, y_pred_multiclass, target_names=classes))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(np.argmax(y_test_binary, axis=1), np.argmax(y_pred_multiclass, axis=1))

# Plot confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig('D:\\IOT_project\\Confusion Matrix.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

rf_model_ovr = OneVsRestClassifier(RandomForestClassifier(class_weight='balanced'))
rf_model_ovr.fit(X_train, y_train_binary)

from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, np.argmax(y_train_binary, axis=1))

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Get feature importances from Random Forest
feature_importances = rf_model_ovr.estimators_[0].feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Feature Importance for Random Forest')
plt.savefig('D:\\IOT_project\\Feature Importance for Random Forest.png', dpi=300, bbox_inches='tight', transparent=True)  
plt.show()

import joblib

joblib.dump(rf_model_ovr, 'random_forest_model.pkl')



