# Binary classification results

Data: CIC-IDS2017 | train flows: 735,588 | test flows: 499,596 | best model (f1_macro): XGBoost

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Detection rate | False alarm rate | ROC AUC | Train time (s) | Predict (ms / 1k flows) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9808 | 0.9544 | 0.9808 | 0.9670 | 0.9808 | 0.0192 | 0.9959 | 28.3 | 1.16 |
| Decision Tree | 0.9987 | 0.9968 | 0.9985 | 0.9976 | 0.9982 | 0.0012 | 0.9987 | 30.0 | 0.18 |
| Random Forest | 0.9989 | 0.9972 | 0.9989 | 0.9980 | 0.9990 | 0.0011 | 1.0000 | 35.7 | 1.80 |
| K-Nearest Neighbors | 0.9960 | 0.9897 | 0.9964 | 0.9930 | 0.9969 | 0.0042 | 0.9989 | 0.2 | 62.54 |
| XGBoost | 0.9991 | 0.9973 | 0.9994 | 0.9983 | 0.9998 | 0.0011 | 1.0000 | 9.8 | 0.70 |
| Neural Network (MLP) | 0.9978 | 0.9943 | 0.9981 | 0.9961 | 0.9984 | 0.0023 | 0.9999 | 66.9 | 1.76 |
