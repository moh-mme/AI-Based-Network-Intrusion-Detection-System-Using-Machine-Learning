# Multiclass classification results

Data: CIC-IDS2017 | train flows: 735,588 | test flows: 499,596 | best model (f1_macro): XGBoost

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Detection rate | False alarm rate | ROC AUC | Train time (s) | Predict (ms / 1k flows) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9954 | 0.9285 | 0.8932 | 0.9074 | 0.9956 | 0.0046 | 0.9996 | 40.8 | 1.29 |
| Decision Tree | 0.9986 | 0.8874 | 0.9511 | 0.9141 | 0.9981 | 0.0013 | 0.9987 | 33.4 | 0.18 |
| Random Forest | 0.9989 | 0.9637 | 0.9496 | 0.9525 | 0.9989 | 0.0011 | 1.0000 | 35.7 | 3.25 |
| K-Nearest Neighbors | 0.9961 | 0.7879 | 0.8371 | 0.8072 | 0.9965 | 0.0038 | 0.9991 | 0.1 | 61.67 |
| XGBoost | 0.9991 | 0.9597 | 0.9753 | 0.9645 | 0.9997 | 0.0010 | 1.0000 | 65.2 | 4.03 |
| Neural Network (MLP) | 0.9978 | 0.9293 | 0.9171 | 0.9008 | 0.9993 | 0.0025 | 0.9999 | 89.4 | 2.02 |
