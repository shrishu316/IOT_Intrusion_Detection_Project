# IoT Intrusion Detection Using Machine Learning

---
## Overview

The rapid expansion of **IoT devices** has increased the risk of cyberattacks. This project builds a **machine learning-based Intrusion Detection System (IDS)** to classify IoT network traffic into categories such as **DDoS**, **DoS**, **malware**, and **benign traffic**. Using models like **Random Forest**, **SVM**, and **Logistic Regression**, we detect abnormal traffic patterns to safeguard IoT networks.

---

## Dataset

The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset), contains labeled IoT network traffic, including:

- **BenignTraffic**
- **DDoS-RSTFINFlood**
- **DoS-TCP_Flood**
- **DDoS-ICMP_Flood**
- **Mirai-greeth_flood**
- ... and other attack types.

### Data Preprocessing

- **Missing values** filled using forward fill (`ffill`).
- **Feature scaling** using `StandardScaler` for normalization.

---

## Key Figures

### Distribution of Filtered Labels
![Distribution_of_Filtered_Labels](https://github.com/user-attachments/assets/637eae6e-7207-4c66-b272-7a052baf3664)
*Fig.1:* The bar chart shows the frequency of various network traffic labels, with DDoS-ICMP_Flood being the most prevalent attack type, followed by DDoS-PSHACK_Flood and DDoS-RSTFINFlood. Less frequent events include reconnaissance activities (e.g., Recon-HostDiscovery) and SQL injection attempts, indicating a diverse range of attack types in the dataset.

### Feature Importance for Random Forest
![Feature Importance for Random Forest](https://github.com/user-attachments/assets/c01cfa94-840d-43a8-a5d8-ed46e617d353)

### Confusion Matrix (Random Forest):
![Confusion Matrix](https://github.com/user-attachments/assets/4f1fd862-453a-4aae-bb91-09d52cfc251e)

### Receiver Operating Characteristic (ROC) Curve for Multiclass
![Receiver Operating Characteristic (ROC) Curve for Multiclass](https://github.com/user-attachments/assets/9164904f-ced2-4e87-a7ce-ef97652d1914)

### ROC Curve (Random Forest):
![ROC Curve](https://github.com/user-attachments/assets/9d80bef9-ed72-4bc4-82bc-8946574997ce)

---

## Machine Learning Models

1. **Random Forest Classifier** – Best performing model, balances accuracy and efficiency.
2. **SVM** – Competitive but computationally expensive.
3. **Logistic Regression** – Used as a baseline.

---

## Results

### Best Model: **Random Forest**
- **Precision**: High for benign traffic and most attack types.
- **Recall**: Lower for rare attack types.
- **Hyperparameter Tuning**: Best parameters achieved with `n_estimators=200`, `max_depth=20`, and `min_samples_split=5`.

---

## Conclusion

The **Random Forest** model proved most effective for IoT traffic classification, offering strong precision and computational efficiency for **real-time intrusion detection**. Future work will focus on improving recall for rare attack types.

