# Fraud Detection Pipeline

## Business Problem
Detecting fraudulent credit card transactions from 284,807 real transactions 
where only 0.17% are fraud. The core challenge: a model that predicts 
"not fraud" for every transaction achieves 99.83% accuracy — making 
accuracy a useless metric here.

## Dataset Challenges
- Extreme class imbalance: (0.17% fraud) — accuracy is meaningless
- Anonymised features: (V1-V28 are PCA-transformed) — limits 
  business interpretability of model explanations
- Skewed Amount distribution: — requires careful scaling to prevent 
  large transactions from dominating the model

## Approach
Built a scikit-learn Pipeline with a ColumnTransformer that scales 
`Amount` and `Time` with StandardScaler while passing V1-V28 through 
unchanged (already PCA-transformed).

`scale_pos_weight` was chosen over SMOTE to handle class imbalance — 
it adjusts the model's loss function without creating synthetic data, 
preserving the original data distribution.

Hyperparameters tuned with RandomizedSearchCV (10 iterations, 3-fold CV) 
over: `n_estimators`, `max_depth`, `learning_rate`, `colsample_bytree`.

## Evaluation Metrics
AUC-PR is the primary metric over AUC-ROC — AUC-ROC is inflated by the 
model's ability to correctly classify the massive legitimate class. 
AUC-PR focuses exclusively on minority class (fraud) performance.

## Results
| Metric | Baseline | Tuned |
|--------|----------|-------|
| Precision | 0.3037 | 0.8750 |
| Recall | 0.7838 | 0.7568 |
| F1 | 0.4377 | 0.8116 |
| AUC-ROC | 0.9559 | 0.9773 |
| AUC-PR | 0.6416 | 0.8337 |

## How to Run
git clone https://github.com/adityapatil371/fraud-detection-pipeline
cd fraud-detection-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/train.py

## What I'd Improve
- Add a threshold tuning step to optimise the precision-recall tradeoff 
  for a specific business cost of missed fraud vs false alarms
- Explore original unmasked features for better SHAP interpretability