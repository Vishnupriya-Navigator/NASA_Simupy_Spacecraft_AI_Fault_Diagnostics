# 🚀 NASA SimuPy Spacecraft AI Fault Diagnostics  
### *AI-Driven Fault Detection & Reliability Diagnostics for Spacecraft Using NASA SimuPy-Flight and Random Forest*

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17626179.svg)](https://doi.org/10.5281/zenodo.17626179)

---

## **Abstract**

Modern spacecraft rely heavily on autonomous onboard software to ensure safe guidance, navigation, control, thermal balance, power stability, and communications.  
Small deviations in telemetry can indicate critical faults that must be detected early.

This repository contains the **complete implementation** for our paper:

### **AI-Driven Fault Detection & Reliability Diagnostics for Spacecraft Using SimuPy and Random Forest**  
*(Submitted to the IEEE Aerospace Conference, 2026 – Montana)*

The framework integrates:  
- **NASA's SimuPy-Flight Vehicle Toolkit** for high-fidelity dynamics  
- **Random Forest–based anomaly detection**  
- **Telemetry simulation + fault injection**  
- **Real-time fault-probability estimation**  
- **Mission-critical feedback loop**  
- **Reproducible metrics and figures (PR/ROC, confusion matrix, feature importances)**

All results in the paper can be reproduced exactly using this repository.

---

## 🛰 **System Architecture**

The following diagram represents the full processing pipeline used throughout the experiments:

> *(Insert your generated architecture diagram here as `figures/architecture.png`)*  
> Example:  
> ```md
> ![Architecture](figures/architecture.png)
> ```

---

## 📂 **Repository Structure**

NASA_Simupy_Spacecraft_AI_Fault_Diagnostics/
├── data/
│ ├── raw/ # SimuPy flight logs + synthetic faults
│ └── processed/ # Merged, labeled telemetry
│
├── framework/
│ ├── adapters/
│ │ └── simupy_flight_adapter.py # NASA SimuPy-Flight integration
│ ├── faults/
│ │ └── faults.py # Bias, drift, spikes, dropout, saturation
│ ├── telemetry/
│ │ ├── telemetry_generator.py # Nominal + faulty telemetry
│ │ └── telemetry_logger.py # Streaming logger (50 Hz)
│ └── models/
│ └── rf_model.py # Random Forest classifier + loader
│
├── scripts/
│ ├── generate_dataset.py # Synthetic dataset
│ ├── generate_simupy_dataset.py # Real dynamics (NASA SimuPy-Flight)
│ ├── train_rf.py # Train Random Forest
│ ├── evaluate.py # Metrics + confusion matrix
│ ├── plot_prob.py # Fault probability vs time
│ ├── metrics_roc_pr.py # PR/ROC curves
│ ├── latency_eval.py # Fault-detection latency
│ └── stream_simupy.py # Real-time stream + RF inference
│
├── results/ # JSON and CSV outputs
├── figures/ # Final publication-ready plots
├── models/ # Trained joblib models
├── requirements.txt
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
