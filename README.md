AI-Driven Fault Detection & Reliability Diagnostics for Spacecraft Using SimuPy and Random Forest
Reproducible Architecture • NASA SimuPy-Flight Integration • Random Forest Anomaly Detection • Conference-Ready Codebase

🛰️ Overview

This repository contains the complete implementation for the paper:

AI-Driven Fault Detection & Reliability Diagnostics for Spacecraft Using SimuPy and Random Forest
(Submitted to the IEEE Aerospace Conference, 2026 – Montana)

The project integrates NASA's SimuPy-Flight Vehicle Toolkit with a Random Forest–based anomaly detection pipeline to create a fully reproducible spacecraft fault-diagnostics framework.
The codebase includes:

Physics-based spacecraft simulation (attitude, rates, actuators, subsystems)

Fault injection engine (bias, drift, spikes, dropout, saturation)

Telemetry generation (nominal + faulty)

Random Forest classifier training + evaluation

Runtime anomaly detection with latency measurement

Mission-critical feedback loop

Publication-ready figures (feature importances, confusion matrix, fault probability vs. time, PR/ROC curves)

All results presented in the paper can be reproduced exactly using this repository.

📁 Repository Structure
NASA_Simupy_Spacecraft_AI_Fault_Diagnostics/
│
├── framework/
│   ├── adapters/
│   │   └── simupy_flight_adapter.py          # NASA SimuPy-Flight integration
│   ├── faults/
│   │   └── faults.py                         # Bias, drift, dropout, spike, saturation
│   ├── telemetry/
│   │   └── telemetry_generator.py            # Nominal + faulty telemetry
│   └── models/
│       └── rf_model.py                       # Random Forest model + loader
│
├── scripts/
│   ├── generate_dataset.py                   # Synthetic dataset (baseline)
│   ├── generate_simupy_dataset.py            # Real dynamics dataset via SimuPy-Flight
│   ├── train_rf.py                           # Train RF classifier
│   ├── evaluate.py                           # Classification report + confusion matrix
│   ├── stream_simupy.py                      # Runtime streaming + fault probability
│   ├── stream_simupy_log.py                  # Stream + log to CSV
│   ├── metrics_roc_pr.py                     # ROC + PR curves (AUC calculation)
│   ├── latency_eval.py                       # Fault detection latency
│   ├── false_alarm_rate.py                   # False alarm rate calculation
│   └── plot_prob.py                          # Fault probability vs. time plot
│
├── data/
│   ├── raw/                                  # CSV telemetry files
│   └── processed/                            # Merged dataset for training
│
├── results/
│   ├── metrics_summary.json
│   ├── latency_summary.json
│   ├── eval_report.json
│   ├── feature_importances_simupy.csv
│   └── confusion_matrix_*.csv
│
├── figures/                                  # Final paper-quality plots
│   ├── feature_importances.png / .pdf
│   ├── confusion_matrix.png / .pdf
│   ├── fault_prob_vs_time.png / .pdf
│   ├── pr_curve.png / .pdf
│   └── roc_curve.png / .pdf
│
├── LICENSE                                   # MIT License
└── README.md

Key Features
1. NASA SimuPy-Flight Attitude Dynamics

High-fidelity nonlinear spacecraft simulation

Exposes angular rates (p, q, r) and quaternion attitude

Extendable to actuators, thermal, power, comms

2. Fault Injection Engine

Supports all major spacecraft anomaly types:

Bias

Drift

Spike

Dropout

Saturation

Thermal imbalance (paper-discussed)

Power-bus fluctuations (paper-discussed)

3. Random Forest Fault Detection

100-tree RF classifier with probability output

Feature importance analysis

Sub-system sensitivity evaluation

4. Runtime Monitor

Live anomaly scoring

Threshold selection (target FPR ≤ 1%)

3-frame hold to suppress noise

Suitable for onboard autonomy loops

5. Performance Metrics (Paper-Aligned)

All final results match the PDF submission:

Metric	Result
ROC-AUC	1.000
PR-AUC	1.000
Detection Latency	≈ 0.70 s
False Alarm Rate	0 per hour
Confusion Matrix	Perfect classification (no FP/FN)
📊 Reproducible Figures (Included)

The following figures are generated exactly from the scripts:

Figure 2 — Feature Importances (Random Forest)

(Matches: feat_2 > feat_0 > feat_1 > quaternions)

Figure 3 — Confusion Matrix

Perfect separation of fault vs. nominal.

Figure 4 — Fault Probability vs Time

Shows RF probability rising sharply when bias fault begins.

Figure 5 — Precision-Recall Curve (AUC = 1.0)

Resilient under class imbalance.

Figure 6 — ROC Curve (AUC = 1.0)

Threshold-independent performance.

All figures are in figures/ folder as .png and .pdf (conference-ready).

🧪 How to Reproduce Results
1. Generate SimuPy-Flight telemetry
python -m scripts.generate_simupy_dataset --seconds 60 --hz 50 --out data/raw/sf_nominal.csv
python -m scripts.generate_simupy_dataset --seconds 60 --hz 50 \
    --fault bias --fault-start 10 --fault-end 20 \
    --out data/raw/sf_bias.csv

2. Merge datasets
python -m scripts.generate_dataset

3. Train Random Forest model
python -m scripts.train_rf

4. Evaluate
python -m scripts.evaluate

5. Generate PR/ROC
python -m scripts.metrics_roc_pr

6. Measure latency + false alarm rate
python -m scripts.latency_eval
python -m scripts.false_alarm_rate

📜 License

This project is released under the MIT License.

NASA SimuPy-Flight is used under its original license (NASA Open Source Agreement).

✉️ Contact

For questions related to the paper or codebase, please contact:
Vishnupriya S. Devarajulu
(GitHub: Vishnupriya-Navigator)