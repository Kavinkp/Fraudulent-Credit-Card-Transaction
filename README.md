# Fraudulent-Credit-Card-Transaction
# Fraudulent Credit Card Transaction Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques on an imbalanced dataset. The goal is to develop a simple but effective classifier that can distinguish between legitimate and fraudulent transactions.

## 🧠 Problem Statement

Credit card fraud is a major issue in financial institutions. Detecting fraudulent transactions quickly and accurately is critical. The dataset used is highly imbalanced — only 492 out of 284,807 transactions are frauds.

## 📁 Dataset

- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains anonymized features `V1` to `V28`, `Time`, `Amount`, and the target `Class`:
  - `0`: Legitimate transaction
  - `1`: Fraudulent transaction

## ⚙️ Approach

1. **Data Exploration & Cleaning**
   - Checked for null values
   - Analyzed class imbalance

2. **Data Balancing**
   - Used **under-sampling** to create a balanced dataset of 492 fraud and 492 legit transactions.

3. **Feature-Target Split**
   - Input features: All columns except `Class`
   - Target: `Class`

4. **Model Training**
   - **Logistic Regression**
   - **Support Vector Classifier (SVC)** with linear kernel

5. **Evaluation**
   - Evaluated on training and test sets using accuracy.

## 📊 Results

| Model                | Train Accuracy | Test Accuracy |
|---------------------|----------------|----------------|
| Logistic Regression | 93.9%          | 92.9%          |
| SVC (Linear Kernel) | 90.2%          | 89.8%          |

Logistic Regression outperformed SVC on both training and test sets.

## 🛠️ Tech Stack

- Python 3
- NumPy
- Pandas
- Scikit-learn

## 🚀 How to Run

1. Clone the repo or download the notebook
2. Make sure to download the dataset (`creditcard.csv`) and place it in the correct path.
3. Open the Jupyter Notebook:  
   `Fraudulent_Credit_Card_Transaction.ipynb`
4. Run all cells to reproduce the results

## 📌 Note

- The project uses **under-sampling** for simplicity. For better real-world performance, try advanced techniques like SMOTE, Ensemble Models, or Anomaly Detection.
