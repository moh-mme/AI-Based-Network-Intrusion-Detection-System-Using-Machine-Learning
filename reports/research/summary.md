# Research results: realistic evaluation and zero-day detection

Data: CIC-IDS2017. Headline numbers use **XGBoost** (without destination port). Alarm budget for calibrated detectors: 1.0% of benign flows. Feature sets compared: All features; Without destination port; Without port and TCP window.

## 1. Reported vs realistic detection

| Setting | Attacks detected | False alarms |
|---|---|---|
| Random split, all features | 100.0% | 0.11% |
| Random split, without destination port | 100.0% | 0.14% |
| Unseen attack family, supervised | 23.5% | 0.12% |
| Unseen attack family, hybrid with Autoencoder | 40.0% | 0.97% |
| Later days (temporal), supervised | 0.4% | 0.04% |
| Later days (temporal), hybrid with Autoencoder | 15.3% | 3.70% |
| Later days (temporal), hybrid with Isolation Forest, recalibrated on recent benign | 1.6% | 0.79% |

Unseen-family figures are averaged over families, each counting equally.

## 2. E1: random split (the usual setup)

| Task | Features | Model | F1 macro | Detection | False alarms |
|---|---|---|---|---|---|
| binary | All features | XGBoost | 0.9983 | 99.98% | 0.11% |
| binary | All features | Random Forest | 0.9979 | 99.89% | 0.12% |
| binary | Without destination port | XGBoost | 0.9978 | 99.96% | 0.14% |
| binary | Without destination port | Random Forest | 0.9973 | 99.82% | 0.15% |
| binary | Without port and TCP window | XGBoost | 0.9956 | 99.44% | 0.19% |
| binary | Without port and TCP window | Random Forest | 0.9955 | 99.41% | 0.19% |
| multiclass | All features | XGBoost | 0.9634 | 99.97% | 0.11% |
| multiclass | All features | Random Forest | 0.9511 | 99.89% | 0.12% |
| multiclass | Without destination port | XGBoost | 0.9462 | 99.94% | 0.14% |
| multiclass | Without destination port | Random Forest | 0.9465 | 99.80% | 0.15% |
| multiclass | Without port and TCP window | XGBoost | 0.9218 | 99.46% | 0.18% |
| multiclass | Without port and TCP window | Random Forest | 0.9224 | 99.41% | 0.18% |

## 3. E2: train on Monday, Tuesday, Wednesday; test on Thursday, Friday

Attack types in the test days never seen in training: Botnet, DDoS, Infiltration, PortScan, WebAttack. Thresholds are calibrated on benign flows from the training data or on recent benign flows from the test days.

| Features | Model | Detector | Calibration | Unseen detected | False alarms | AUC |
|---|---|---|---|---|---|---|
| All features | none | Isolation Forest | training data | 0.1% | 3.82% | 0.816 |
| All features | none | Isolation Forest | recent benign | 0.0% | 1.02% | 0.816 |
| All features | none | Autoencoder | training data | 54.2% | 9.32% | 0.909 |
| All features | none | Autoencoder | recent benign | 0.0% | 1.06% | 0.909 |
| All features | XGBoost | Supervised, threshold 0.5 | — | 1.3% | 0.02% | 0.836 |
| All features | XGBoost | Supervised, calibrated | training data | 59.9% | 1.15% | 0.836 |
| All features | XGBoost | Hybrid: supervised + Isolation Forest | training data | 56.5% | 3.79% | 0.836 |
| All features | XGBoost | Hybrid: supervised + Autoencoder | training data | 61.8% | 5.25% | 0.955 |
| All features | XGBoost | Supervised, calibrated | recent benign | 59.8% | 1.02% | 0.836 |
| All features | XGBoost | Hybrid: supervised + Isolation Forest | recent benign | 43.5% | 0.96% | 0.836 |
| All features | XGBoost | Hybrid: supervised + Autoencoder | recent benign | 43.5% | 1.02% | 0.943 |
| All features | Random Forest | Supervised, threshold 0.5 | — | 36.7% | 0.04% | 0.939 |
| All features | Random Forest | Supervised, calibrated | training data | 62.3% | 5.07% | 0.939 |
| All features | Random Forest | Hybrid: supervised + Isolation Forest | training data | 57.8% | 4.01% | 0.935 |
| All features | Random Forest | Hybrid: supervised + Autoencoder | training data | 63.0% | 5.32% | 0.955 |
| All features | Random Forest | Supervised, calibrated | recent benign | 49.7% | 0.93% | 0.939 |
| All features | Random Forest | Hybrid: supervised + Isolation Forest | recent benign | 46.6% | 0.93% | 0.926 |
| All features | Random Forest | Hybrid: supervised + Autoencoder | recent benign | 46.6% | 0.99% | 0.956 |
| Without destination port | none | Isolation Forest | training data | 0.2% | 3.61% | 0.816 |
| Without destination port | none | Isolation Forest | recent benign | 0.1% | 1.03% | 0.816 |
| Without destination port | none | Autoencoder | training data | 34.0% | 6.31% | 0.914 |
| Without destination port | none | Autoencoder | recent benign | 0.0% | 1.09% | 0.914 |
| Without destination port | XGBoost | Supervised, threshold 0.5 | — | 0.4% | 0.04% | 0.928 |
| Without destination port | XGBoost | Supervised, calibrated | training data | 33.4% | 2.48% | 0.928 |
| Without destination port | XGBoost | Hybrid: supervised + Isolation Forest | training data | 7.5% | 2.18% | 0.875 |
| Without destination port | XGBoost | Hybrid: supervised + Autoencoder | training data | 15.3% | 3.70% | 0.920 |
| Without destination port | XGBoost | Supervised, calibrated | recent benign | 6.1% | 1.01% | 0.928 |
| Without destination port | XGBoost | Hybrid: supervised + Isolation Forest | recent benign | 1.6% | 0.79% | 0.882 |
| Without destination port | XGBoost | Hybrid: supervised + Autoencoder | recent benign | 1.6% | 0.76% | 0.911 |
| Without destination port | Random Forest | Supervised, threshold 0.5 | — | 36.6% | 0.06% | 0.950 |
| Without destination port | Random Forest | Supervised, calibrated | training data | 47.8% | 3.15% | 0.950 |
| Without destination port | Random Forest | Hybrid: supervised + Isolation Forest | training data | 38.3% | 2.60% | 0.937 |
| Without destination port | Random Forest | Hybrid: supervised + Autoencoder | training data | 38.3% | 3.86% | 0.939 |
| Without destination port | Random Forest | Supervised, calibrated | recent benign | 37.9% | 0.84% | 0.950 |
| Without destination port | Random Forest | Hybrid: supervised + Isolation Forest | recent benign | 37.7% | 0.79% | 0.923 |
| Without destination port | Random Forest | Hybrid: supervised + Autoencoder | recent benign | 37.7% | 0.91% | 0.943 |
| Without port and TCP window | none | Isolation Forest | training data | 0.1% | 3.00% | 0.798 |
| Without port and TCP window | none | Isolation Forest | recent benign | 0.0% | 1.11% | 0.798 |
| Without port and TCP window | none | Autoencoder | training data | 31.8% | 8.17% | 0.904 |
| Without port and TCP window | none | Autoencoder | recent benign | 0.0% | 1.13% | 0.904 |
| Without port and TCP window | XGBoost | Supervised, threshold 0.5 | — | 23.8% | 0.14% | 0.884 |
| Without port and TCP window | XGBoost | Supervised, calibrated | training data | 37.6% | 1.02% | 0.884 |
| Without port and TCP window | XGBoost | Hybrid: supervised + Isolation Forest | training data | 35.6% | 2.02% | 0.799 |
| Without port and TCP window | XGBoost | Hybrid: supervised + Autoencoder | training data | 36.0% | 4.18% | 0.923 |
| Without port and TCP window | XGBoost | Supervised, calibrated | recent benign | 36.5% | 0.81% | 0.884 |
| Without port and TCP window | XGBoost | Hybrid: supervised + Isolation Forest | recent benign | 33.6% | 0.88% | 0.810 |
| Without port and TCP window | XGBoost | Hybrid: supervised + Autoencoder | recent benign | 33.6% | 0.98% | 0.898 |
| Without port and TCP window | Random Forest | Supervised, threshold 0.5 | — | 36.6% | 0.11% | 0.956 |
| Without port and TCP window | Random Forest | Supervised, calibrated | training data | 41.9% | 1.30% | 0.956 |
| Without port and TCP window | Random Forest | Hybrid: supervised + Isolation Forest | training data | 37.9% | 1.96% | 0.937 |
| Without port and TCP window | Random Forest | Hybrid: supervised + Autoencoder | training data | 38.1% | 4.11% | 0.930 |
| Without port and TCP window | Random Forest | Supervised, calibrated | recent benign | 38.7% | 0.85% | 0.956 |
| Without port and TCP window | Random Forest | Hybrid: supervised + Isolation Forest | recent benign | 37.8% | 0.91% | 0.931 |
| Without port and TCP window | Random Forest | Hybrid: supervised + Autoencoder | recent benign | 37.8% | 1.01% | 0.946 |

## 4. E3: leave one attack family out (zero-day protocol)

XGBoost, without destination port. Share of the left-out family's flows that raised an alarm.

| Family left out | Supervised, threshold 0.5 | Supervised, calibrated | Isolation Forest | Autoencoder | Hybrid: supervised + Isolation Forest | Hybrid: supervised + Autoencoder |
|---|---|---|---|---|---|---|
| Botnet | 0.0% | 0.9% | 0.4% | 0.0% | 0.1% | 0.1% |
| BruteForce | 0.2% | 2.2% | 0.0% | 0.0% | 1.6% | 1.6% |
| DDoS | 62.0% | 63.6% | 0.1% | 0.0% | 63.6% | 63.6% |
| DoS | 16.4% | 80.7% | 2.4% | 30.0% | 72.4% | 89.5% |
| Infiltration | 0.0% | 13.9% | 75.0% | 30.6% | 41.7% | 33.3% |
| PortScan | 0.4% | 1.1% | 0.0% | 0.0% | 1.1% | 1.1% |
| WebAttack | 85.7% | 92.5% | 0.0% | 0.0% | 91.2% | 91.2% |
| **Mean** | 23.5% | 36.4% | 11.1% | 8.7% | 38.8% | 40.0% |
| False alarms (mean) | 0.12% | 0.95% | 0.93% | 1.00% | 0.89% | 0.97% |

Share of unseen-family flows the model calls benign with over 99% confidence:

| Family left out | Confidently missed |
|---|---|
| Botnet | 99.7% |
| BruteForce | 98.1% |
| DDoS | 36.4% |
| DoS | 32.4% |
| Infiltration | 88.9% |
| PortScan | 98.9% |
| WebAttack | 7.7% |

## 5. E4: reliance on candidate shortcut features (TreeSHAP, XGBoost, binary)

Mean share of each flow's attribution, model trained with all features.

| Category | Destination Port | Init_Win_bytes_forward | Init_Win_bytes_backward | Top features (all) | Top features (without port and tcp window) |
|---|---|---|---|---|---|
| Benign | 12.6% | 10.6% | 10.7% | Destination Port, Init_Win_bytes_backward, Init_Win_bytes_forward | Average Packet Size, Bwd Packet Length Std, Fwd Packet Length Max |
| Botnet | 4.1% | 7.4% | 18.3% | Init_Win_bytes_backward, Bwd Packet Length Mean, Init_Win_bytes_forward | Average Packet Size, Bwd Packet Length Mean, Fwd Packet Length Max |
| BruteForce | 42.9% | 1.3% | 4.4% | Destination Port, Bwd Packet Length Std, Init_Win_bytes_backward | Bwd Packet Length Mean, Packet Length Std, Bwd Header Length |
| DDoS | 6.7% | 4.8% | 9.4% | Bwd Packet Length Std, Init_Win_bytes_backward, Average Packet Size | Bwd Packet Length Std, Fwd Packet Length Max, Bwd Packet Length Mean |
| DoS | 6.3% | 3.3% | 11.2% | Bwd Packet Length Std, Init_Win_bytes_backward, Destination Port | Bwd Packet Length Std, Bwd Packet Length Mean, Idle Max |
| Infiltration | 3.9% | 4.4% | 4.9% | Bwd Packet Length Mean, Bwd Packet Length Std, Average Packet Size | Bwd Packet Length Mean, Fwd Packet Length Max, Average Packet Size |
| PortScan | 8.7% | 4.7% | 5.8% | Average Packet Size, Destination Port, Bwd Header Length | Average Packet Size, Packet Length Std, Bwd Header Length |
| WebAttack | 10.1% | 3.8% | 3.8% | Fwd IAT Min, Average Packet Size, Destination Port | Fwd IAT Min, Fwd Header Length, Fwd IAT Total |


Figures: headline.png, loao_heatmap.png, overconfidence.png, shap_shortcut_share.png
