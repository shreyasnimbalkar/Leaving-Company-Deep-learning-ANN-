# Employee Churn Prediction using ANN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Latest-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Deep learning model to predict employee churn using Artificial Neural Networks (ANN), helping organizations identify at-risk employees and reduce turnover costs.

## Overview

This project implements a binary classification neural network to predict whether an employee will leave the company based on various demographic and professional features. The model achieves 85.85% accuracy on test data.

## Features

- Deep learning architecture with 2 hidden layers
- Feature engineering with one-hot encoding
- Proper data preprocessing and normalization
- Training/validation split for model evaluation
- Confusion matrix and accuracy metrics
- Visualization of training progress

## Tech Stack

- **Python 3.8+**
- **TensorFlow/Keras** - Deep learning framework
- **scikit-learn** - Preprocessing and evaluation
- **Pandas & NumPy** - Data manipulation
- **Matplotlib** - Visualization

## Dataset

**Churn Modelling Dataset**
- Samples: 10,000 bank customers
- Target: Binary (Exited: 0 = Stayed, 1 = Left)

**Features:**
- CreditScore
- Geography (France, Germany, Spain)
- Gender (Male, Female)
- Age
- Tenure (years with bank)
- Balance
- NumOfProducts
- HasCrCard (Has credit card: 0/1)
- IsActiveMember (Active: 0/1)
- EstimatedSalary

## Model Architecture

```
Input Layer (11 features after encoding)
    ↓
Dense Layer 1 (7 neurons, ReLU activation)
    ↓
Dense Layer 2 (6 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

**Training Configuration:**
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Batch Size: 10
- Epochs: 100
- Validation Split: 33% of training data
- Train-Test Split: 80-20

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shreyasnimbalkar/Leaving-Company-Deep-learning-ANN-.git
cd Leaving-Company-Deep-learning-ANN-
```

2. Install dependencies:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook DL_Practice_(ANN).ipynb
```

## Usage

### Data Preprocessing
```python
# Load dataset
df = pd.read_csv('Churn_Modelling.csv')

# One-hot encoding for categorical variables
# Geography: France, Germany, Spain
# Gender: Male, Female

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Model Training
```python
# Build model
model = Sequential([
    Dense(7, activation='relu', input_shape=(11,)),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train,
                   batch_size=10,
                   epochs=100,
                   validation_split=0.33)
```

### Prediction
```python
# Make predictions
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(y_test, y_pred_binary)
```

## Results

**Model Performance:**
- **Test Accuracy:** 85.85%
- **Training Epochs:** 100
- **Validation Accuracy:** Monitored during training

**Confusion Matrix:**
```
[[1512   83]   True Negatives: 1512 | False Positives: 83
 [ 200  205]]  False Negatives: 200 | True Positives: 205
```

**Insights:**
- Model correctly identifies 1512 employees who stayed
- 205 churned employees correctly predicted
- False positive rate: 5.2% (83 false alarms)
- False negative rate: 49.4% (200 missed churns)

## Visualizations

The notebook includes:
- Training/validation accuracy curves
- Training/validation loss curves
- Confusion matrix heatmap
- Feature distribution analysis

## Data Preprocessing Details

1. **Categorical Encoding:**
   - Geography: One-hot encoded (3 columns)
   - Gender: Binary encoded (1 column)

2. **Feature Scaling:**
   - StandardScaler applied to all features
   - Fitted on training data only
   - Transform applied to both train and test

3. **Data Split:**
   - Training: 80% (8,000 samples)
   - Testing: 20% (2,000 samples)
   - Validation: 33% of training (within model.fit)

## Business Impact

Understanding employee churn helps organizations:
- Identify at-risk employees early
- Implement targeted retention strategies
- Reduce recruitment and training costs
- Improve workplace culture
- Optimize compensation packages

## Future Improvements

- Hyperparameter tuning (learning rate, layers, neurons)
- Add dropout layers to prevent overfitting
- Implement class weighting for imbalanced data
- Feature importance analysis
- Deploy as REST API for real-time predictions
- Add SHAP values for model interpretability

## Contributing

Contributions are welcome! Feel free to:
- Suggest model improvements
- Add new features
- Report bugs
- Submit pull requests

## License

MIT License

## Author

**Shreyas Nimbalkar**
- GitHub: [@shreyasnimbalkar](https://github.com/shreyasnimbalkar)
- Email: shreyasgnimbalkarwork@gmail.com
- LinkedIn: [Shreyas Nimbalkar](https://www.linkedin.com/in/shreyas-nimbalkar-713bb9267/)
