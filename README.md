# Customer churn prediction with an ANN

A small Keras neural network that predicts whether a bank customer will leave (churn),
based on things like credit score, age, balance and how active they are. It reaches
85.85% accuracy on a test set of 2,000 customers.

The repo name says "leaving company". The dataset is about customers leaving a bank,
not employees leaving a job.

## Data

`Churn_Modelling.csv` (included): 10,000 bank customers, one row each. The target is
`Exited` (1 = left the bank, 0 = stayed).

I used 10 columns as input: CreditScore, Geography, Gender, Age, Tenure, Balance,
NumOfProducts, HasCrCard, IsActiveMember and EstimatedSalary. Row number, customer ID
and surname were dropped.

## Approach

1. One-hot encode Geography and Gender with `drop_first=True`. That leaves 11
   features: 8 numeric, 2 for geography, 1 for gender.
2. Split 80/20 into 8,000 training and 2,000 test rows (`random_state=0`).
3. Standardize with `StandardScaler`, fitted on the training set only.
4. Train the network:

```
Dense(11, relu)
Dense(7, relu)
Dense(6, relu)
Dense(1, sigmoid)
```

It uses Adam and binary cross-entropy, batch size 10, for 100 epochs. A third of the
training data is held out for validation. The notebook creates an `EarlyStopping`
callback, but it isn't passed to `fit`, so all 100 epochs run.

5. Predict "left" when the output is 0.5 or higher.

## Results

Test accuracy: **85.85%**

Confusion matrix on the 2,000 test customers:

| | Predicted stayed | Predicted left |
|---|---|---|
| Actually stayed (1,595) | 1,512 | 83 |
| Actually left (405) | 200 | 205 |

For context, 1,595 of the 2,000 test customers stayed. So a model that always
predicts "stays" would already score 79.75%. The model catches 205 of the 405
customers who left, about half, with 83 false alarms. If the goal is to find
customers before they leave, the next steps would be class weights or a lower
threshold, at the cost of more false alarms.

## Running it

The notebook was written in Google Colab and reads the CSV from
`/content/Churn_Modelling.csv`. In Colab, upload the CSV and run all cells. To run it
locally, change that path to `Churn_Modelling.csv`, then:

```
pip install tensorflow scikit-learn pandas numpy matplotlib jupyter
jupyter notebook "DL_Practice_(ANN).ipynb"
```

## Files

- `DL_Practice_(ANN).ipynb`: preprocessing, training, plots and evaluation
- `Churn_Modelling.csv`: the dataset
- `LICENSE`: MIT
