# 💳 Credit Card Fraud Detection using Deep Learning (RNN)

Building a **Recurrent Neural Network (RNN)** model to detect fraudulent credit card transactions using a real-world dataset.

---

## 👥 Group 6 Members

1. Khalif Ziran Maulana 
2. Daniel Ricardo Hadirahardja
3. Muhammad Ravindra Mahesandra 
4. Vinsensius Paulo Ryananda Virgiawan

---

## 🎯 Project Objective

To build a deep learning model capable of effectively detecting fraudulent credit card transactions using temporal data.  
The model should be able to identify fraudulent patterns with good precision and recall, even under highly imbalanced data conditions.

---

## 📁 Dataset

- **Source**: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
- **Samples**: 284,807 transactions
- **Fraud cases**: 492 (≈0.172%)
- **Features**:
  - `Time`, `Amount`, `Class` (target)
  - `V1` to `V28` (PCA-transformed features)

---

## 🧪 Methodology

1. **Data Preprocessing**
   - Checked for null values (none found)
   - Normalized `Amount` using StandardScaler
   - Used **SMOTE** to handle data imbalance
   - Created time-series sequences with a window of 30
   - Data split: 80% training, 20% testing

2. **Model Architecture**
   - Two LSTM layers with dropout
   - Dense layer with ReLU + L2 regularization
   - Final sigmoid output layer
   - Optimizer: Adam (LR: 0.0001)
   - Loss: Binary Crossentropy
   - EarlyStopping based on validation loss

3. **Evaluation Metrics**
   - Accuracy: 71%
   - Precision: 75%
   - Recall: 63%
   - F1-Score: 69%
   - Visualized loss and accuracy curves

---

## 🧠 Technologies Used

- Python
- Jupyter Notebook
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib, Scikit-learn
- SMOTE (imbalanced-learn)

---

## ⚙️ How to Run

> Make sure you have Python and Jupyter Notebook installed. And clone this repository.
