# Dataset

This project uses the **Bank Customer Churn Dataset** (ABC Multistate Bank) published on Kaggle:
https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

The CSV is not included in this repository. To run the notebooks:

1. Download `Bank Customer Churn Prediction.csv` from the Kaggle page above.
2. Place it in this folder: `dataset/Bank Customer Churn Prediction.csv`.

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier (dropped before modeling) |
| `credit_score` | Credit score |
| `country` | France, Germany or Spain |
| `gender` | Female or Male |
| `age` | Age in years |
| `tenure` | Years as a customer |
| `balance` | Account balance |
| `products_number` | Number of bank products held (1-4) |
| `credit_card` | 1 if the customer has a credit card |
| `active_member` | 1 if the customer is an active member |
| `estimated_salary` | Estimated annual salary |
| `churn` | **Target**: 1 if the customer left the bank |

10,000 rows, 12 columns, no missing values, churn rate 20.4%.
