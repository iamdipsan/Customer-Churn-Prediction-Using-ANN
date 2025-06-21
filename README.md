# Customer Churn Prediction using Artificial Neural Networks (ANN)

This project is a simple yet practical implementation of an Artificial Neural Network to predict customer churn in the banking sector. The main objective of this project was to understand how neural networks are built and trained using Keras and TensorFlow.

## 🔍 Dataset

- **Source**: [Kaggle - Credit Card Customer Churn Prediction](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction)
- The dataset contains various features like customer demographics, account information, and credit card usage behavior.
- Target variable: `Attrition_Flag` (whether a customer has churned or not)

## 📌 Objective

The primary goal of this project is **learning**:
- How to build a neural network from scratch using Keras
- Apply it to a real-world binary classification problem

## 🧠 Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn (for EDA)
- TensorFlow / Keras (for ANN)
- Scikit-learn (for preprocessing and metrics)

## 🧪 Steps Followed

1. **Data Loading & Preprocessing**
   - Handling categorical variables using one-hot encoding
   - Feature scaling using StandardScaler
   - Splitting into training and test sets

2. **Building the Neural Network**
   - Used `Sequential` API
   - Added hidden layers with ReLU activation
   - Output layer with sigmoid activation (binary classification)
   - Compiled with binary crossentropy loss and Adam optimizer

3. **Training & Evaluation**
   - Trained the model using `.fit()`
   - Evaluated on test data using accuracy and confusion matrix

## 🧾 Model Summary

```python
model = Sequential()
model.add(Dense(units=11, activation='relu', input_dim=11))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
