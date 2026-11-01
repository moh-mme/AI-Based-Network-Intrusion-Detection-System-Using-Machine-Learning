# Research results: realistic evaluation and zero-day detection

Data: CIC-IDS2017. Headline numbers use **XGBoost** (without shortcut features). Alarm budget for calibrated detectors: 1.0% of benign flows. Shortcut features: Destination Port.

> Quick run (small samples) for: random, temporal, loao, shap. Rerun without --quick for final numbers.

## 1. Reported vs realistic detection

| Setting | Attacks detected | False alarms |
|---|---|---|
| Random split, all features | 100.0% | 0.17% |
| Random split, no shortcut features | 100.0% | 0.23% |
| Unseen attack family, supervised | 23.4% | 0.21% |
| Unseen attack family, hybrid with Autoencoder | 40.2% | 1.21% |
| Later days (temporal), supervised | 4.7% | 0.04% |
| Later days (temporal), hybrid with Autoencoder | 37.6% | 4.02% |

Unseen-family figures are averaged over families, each counting equally.

## 2. E1: random split (the usual setup)

| Task | Features | Model | F1 macro | Detection | False alarms |
|---|---|---|---|---|---|
| binary | All features | XGBoost | 0.9974 | 99.97% | 0.17% |
| binary | All features | Random Forest | 0.9971 | 99.89% | 0.17% |
| binary | Without shortcut features | XGBoost | 0.9966 | 99.96% | 0.23% |
| binary | Without shortcut features | Random Forest | 0.9960 | 99.78% | 0.23% |
| multiclass | All features | XGBoost | 0.9633 | 99.97% | 0.17% |
| multiclass | All features | Random Forest | 0.8385 | 99.84% | 0.16% |
| multiclass | Without shortcut features | XGBoost | 0.8278 | 99.92% | 0.21% |
| multiclass | Without shortcut features | Random Forest | 0.8340 | 99.72% | 0.22% |

## 3. E2: train on Monday, Tuesday, Wednesday; test on Thursday, Friday

Attack types in the test days never seen in training: Botnet, DDoS, Infiltration, PortScan, WebAttack.

| Features | Model | Detector | Unseen detected | False alarms | AUC |
|---|---|---|---|---|---|
| All features | none | Isolation Forest | 0.1% | 3.47% | 0.821 |
| All features | none | Autoencoder | 19.9% | 4.40% | 0.927 |
| All features | XGBoost | Supervised, threshold 0.5 | 37.4% | 0.04% | 0.817 |
| All features | XGBoost | Supervised, calibrated | 59.1% | 1.10% | 0.817 |
| All features | XGBoost | Hybrid: supervised + Isolation Forest | 56.3% | 2.69% | 0.842 |
| All features | XGBoost | Hybrid: supervised + Autoencoder | 56.3% | 4.31% | 0.956 |
| All features | Random Forest | Supervised, threshold 0.5 | 36.7% | 0.05% | 0.892 |
| All features | Random Forest | Supervised, calibrated | 60.0% | 2.39% | 0.892 |
| All features | Random Forest | Hybrid: supervised + Isolation Forest | 58.1% | 2.81% | 0.894 |
| All features | Random Forest | Hybrid: supervised + Autoencoder | 58.1% | 4.25% | 0.954 |
| Without shortcut features | none | Isolation Forest | 0.2% | 3.59% | 0.807 |
| Without shortcut features | none | Autoencoder | 12.7% | 4.16% | 0.937 |
| Without shortcut features | XGBoost | Supervised, threshold 0.5 | 4.7% | 0.04% | 0.927 |
| Without shortcut features | XGBoost | Supervised, calibrated | 38.5% | 1.42% | 0.927 |
| Without shortcut features | XGBoost | Hybrid: supervised + Isolation Forest | 37.6% | 1.79% | 0.873 |
| Without shortcut features | XGBoost | Hybrid: supervised + Autoencoder | 37.6% | 4.02% | 0.939 |
| Without shortcut features | Random Forest | Supervised, threshold 0.5 | 36.6% | 0.08% | 0.846 |
| Without shortcut features | Random Forest | Supervised, calibrated | 42.2% | 1.36% | 0.846 |
| Without shortcut features | Random Forest | Hybrid: supervised + Isolation Forest | 37.0% | 1.84% | 0.858 |
| Without shortcut features | Random Forest | Hybrid: supervised + Autoencoder | 37.0% | 3.67% | 0.946 |

## 4. E3: leave one attack family out (zero-day protocol)

XGBoost, without shortcut features. Share of the left-out family's flows that raised an alarm.

| Family left out | Supervised, threshold 0.5 | Supervised, calibrated | Isolation Forest | Autoencoder | Hybrid: supervised + Isolation Forest | Hybrid: supervised + Autoencoder |
|---|---|---|---|---|---|---|
| Botnet | 0.6% | 2.1% | 0.4% | 0.0% | 1.3% | 1.3% |
| BruteForce | 0.5% | 2.7% | 0.0% | 0.0% | 2.3% | 2.3% |
| DDoS | 58.1% | 63.5% | 0.0% | 0.0% | 63.2% | 63.2% |
| DoS | 12.2% | 81.4% | 2.3% | 30.4% | 63.9% | 85.5% |
| Infiltration | 2.8% | 5.6% | 75.0% | 30.6% | 36.1% | 36.1% |
| PortScan | 1.1% | 3.1% | 0.1% | 0.0% | 1.1% | 1.1% |
| WebAttack | 88.6% | 92.5% | 0.0% | 0.0% | 92.1% | 92.1% |
| **Mean** | 23.4% | 35.8% | 11.1% | 8.7% | 37.1% | 40.2% |
| False alarms (mean) | 0.21% | 1.10% | 1.16% | 1.25% | 1.14% | 1.21% |

Share of unseen-family flows the model calls benign with over 99% confidence:

| Family left out | Confidently missed |
|---|---|
| Botnet | 97.6% |
| BruteForce | 74.8% |
| DDoS | 36.4% |
| DoS | 10.2% |
| Infiltration | 91.7% |
| PortScan | 75.3% |
| WebAttack | 7.4% |

## 5. E4: reliance on shortcut features (TreeSHAP, XGBoost, binary)

| Category | Shortcut share | Top features (all) | Top features (no shortcuts) |
|---|---|---|---|
| Benign | 14.6% | Init_Win_bytes_forward, Destination Port, Bwd Packet Length Min | Init_Win_bytes_forward, Bwd Packet Length Min, Init_Win_bytes_backward |
| Botnet | 8.8% | Init_Win_bytes_backward, Bwd Packet Length Min, Destination Port | Init_Win_bytes_backward, Bwd Packet Length Min, Init_Win_bytes_forward |
| BruteForce | 25.3% | Destination Port, Bwd Packet Length Min, Init_Win_bytes_backward | Init_Win_bytes_backward, Bwd Packet Length Min, Packet Length Std |
| DDoS | 11.3% | Bwd Packet Length Std, Destination Port, Init_Win_bytes_backward | Bwd Packet Length Std, Bwd Packet Length Min, Init_Win_bytes_backward |
| DoS | 11.1% | Bwd Packet Length Std, Destination Port, Init_Win_bytes_backward | Bwd Packet Length Std, Init_Win_bytes_backward, Bwd Packet Length Min |
| Infiltration | 3.3% | Idle Mean, Fwd Packet Length Max, Init_Win_bytes_forward | Idle Min, Fwd Packet Length Max, Init_Win_bytes_backward |
| PortScan | 7.8% | Packet Length Std, Bwd Packet Length Min, Destination Port | Packet Length Std, Bwd Packet Length Min, PSH Flag Count |
| WebAttack | 15.1% | Destination Port, Fwd IAT Min, Init_Win_bytes_forward | Fwd IAT Min, Init_Win_bytes_forward, Fwd Header Length |


Figures: headline.png, loao_heatmap.png, overconfidence.png, shap_shortcut_share.png
