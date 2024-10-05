# IOT_Intrusion_Detection_Project
# IoT Intrusion Detection Using Machine Learning

## Overview

The rapid growth of Internet of Things (IoT) devices has introduced a significant increase in the potential attack surface for cyber threats. This project focuses on building a machine learning model to classify IoT network traffic into various attack types, such as DDoS, DoS, malware, and benign traffic. Using this model, IoT devices can detect abnormal traffic patterns and defend against network intrusions in real-time.

In this project, I used several machine learning models (Random Forest, Support Vector Machine, and Logistic Regression) to classify network traffic. The model's performance was evaluated using classification metrics, including precision, recall, F1-score, confusion matrices, and ROC curves.

## Dataset

The dataset contains network traffic labeled as either benign or specific attack types, such as:

- DDoS-RSTFINFlood
- DoS-TCP_Flood
- DDoS-ICMP_Flood
- Mirai-greeth_flood
- BenignTraffic
- ... (and other attack types)

The dataset includes multiple features like packet sizes, protocols, flow duration, etc., which are used as inputs to classify the type of network traffic.

### File: `IOT_data.csv`

- **Features:** Various features that describe the characteristics of the network traffic.
- **Labels:** Different types of attacks and benign traffic.

## Data Preprocessing

1. **Handling Missing Values:**  
   Forward fill (`fillna(method='ffill')`) was applied to fill in missing values. This is necessary to ensure that all data points are complete for model training.
   
2. **Label Filtering:**  
   I filtered out irrelevant classes and retained only the most critical attack labels and benign traffic. This helped focus the classification task on meaningful traffic patterns.

3. **Feature Scaling:**  
   Features were standardized using `StandardScaler` to ensure that they all have a mean of 0 and a standard deviation of 1. This step was crucial for models like SVM, which are sensitive to feature scaling.

## Models Used

### 1. **Random Forest Classifier**
Random Forest is an ensemble model that aggregates the decisions from multiple decision trees. It is well-suited for this classification problem because it handles high-dimensional data effectively and is robust against overfitting.

### 2. **Support Vector Machine (SVM)**
SVM is a powerful classifier that works well in high-dimensional spaces. It was included to compare performance, particularly in cases where there may be non-linear boundaries between the classes.

### 3. **Logistic Regression**
Logistic regression was used as a baseline linear model to evaluate the performance against more complex algorithms.

## Evaluation Metrics

### Classification Reports:
The precision, recall, F1-score, and support for each class were calculated to compare model performance. These metrics provide insight into how well the model predicts each class (e.g., detecting attacks vs. benign traffic).

### Confusion Matrix:
A confusion matrix was generated for each model to visualize the number of true positives, false positives, true negatives, and false negatives. This helps identify which classes are more difficult to predict accurately.

### ROC Curve and AUC:
ROC curves were plotted to evaluate the trade-off between true positive and false positive rates across different threshold values. The Area Under the Curve (AUC) was used to summarize model performance.

## Results

### 1. **Random Forest Results**
- **Precision:** The model performed well on most classes, especially for the benign traffic class.
- **Recall:** The recall for some attack classes was lower, indicating room for improvement in detecting rare attacks.
- **Confusion Matrix:** Visualized how well the model predicted each label.

### 2. **Support Vector Machine Results**
- **Performance:** While SVM performed comparably to Random Forest, it required more computational resources due to the dataset's size and high-dimensional nature.

### 3. **Logistic Regression Results**
- **Baseline Comparison:** Logistic Regression provided a reasonable baseline, but it lacked the flexibility to capture complex patterns compared to Random Forest and SVM.

### Best Performing Model: **Random Forest**  
The Random Forest model provided the best balance between performance and computational efficiency.

### ROC Curve Example (Binary Classification):
![Receiver Operating Characteristic (ROC) Curve](https://github.com/user-attachments/assets/9d80bef9-ed72-4bc4-82bc-8946574997ce)

### Confusion Matrix Example (Random Forest):
![Confusion Matrix](https://github.com/user-attachments/assets/4f1fd862-453a-4aae-bb91-09d52cfc251e)

## Hyperparameter Tuning

I used Grid Search with Cross-Validation (`GridSearchCV`) to find the optimal hyperparameters for the Random Forest model. The following parameters were tuned:

- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of each tree
- `min_samples_split`: Minimum number of samples required to split an internal node

### Best Parameters:

```json
{
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5
}
