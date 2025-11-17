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

*(Insert your architecture diagram in `figures/architecture.png`)*

---

##  **Pipeline Structure**

(ASCII diagram omitted for brevity—user can insert as needed)

---

## **Repository Structure**

```plaintext
Spacecraft_AI_Fault_Diagnostics/
├── data/
│   └── simupyflight/
├── framework/
│   ├── adapters/
│   ├── telemetry_generator.py
│   ├── faults.py
│   ├── dataset_builder.py
│   ├── random_forest_model.py
│   └── runtime_monitor.py
├── scripts/
│   ├── generate_dataset.py
│   ├── train_rf.py
│   └── probe_sf_stream.py
├── figures/
├── requirements.txt
└── README.md
```

---

## **Installation**

### **1. Clone repository**
```bash
git clone https://github.com/<your-username>/Spacecraft_AI_Fault_Diagnostics
cd Spacecraft_AI_Fault_Diagnostics
```

### **2. Create virtual environment**
```bash
python3 -m venv .venv
```

### **3. Activate environment**

**Mac / Linux**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

### **4. Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## **Usage**

### **1. Run NASA SimuPy-Flight Streaming Probe**
```bash
python -m scripts.probe_sf_stream --hz 10 --seconds 5
```

**With fault injection**
```bash
python -m scripts.probe_sf_stream --hz 10 --seconds 5 --fault bias
```

---

### **2. Generate Dataset (SimuPy + Fault Injection)**
```bash
python -m scripts.generate_dataset     --from-simupyflight     --cycles 30     --fault-types bias drift dropout     --output data/simupyflight/
```

---

### **3. Train Random Forest Classifier**
```bash
python -m scripts.train_rf --input data/simupyflight/
```

---

### **4. Run Real-Time Fault Monitor**
```bash
python -m framework.runtime_monitor --model rf_model.pkl
```

---

## **Experimental Results (Placeholder)**

| Metric     | Value |
|------------|-------|
| Accuracy   | TBD   |
| Precision  | TBD   |
| Recall     | TBD   |
| F1 Score   | TBD   |

---

## **How to Reproduce Results**

### **1. Generate SimuPy-Flight telemetry**
```bash
python -m scripts.generate_simupy_dataset --seconds 60 --hz 50 --out data/raw/sf_nominal.csv

python -m scripts.generate_simupy_dataset --seconds 60 --hz 50     --fault bias --fault-start 10 --fault-end 20     --out data/raw/sf_bias.csv
```

### **2. Merge datasets**
```bash
python -m scripts.generate_dataset
```

### **3. Train Random Forest model**
```bash
python -m scripts.train_rf
```

### **4. Evaluate**
```bash
python -m scripts.evaluate
```

### **5. Generate PR / ROC curves**
```bash
python -m scripts.metrics_roc_pr
```

### **6. Measure latency + false alarm rate**
```bash
python -m scripts.latency_eval
python -m scripts.false_alarm_rate
```

---

## **License**

This project is released under the **MIT License**.  
NASA SimuPy-Flight is used under its original **NASA Open Source Agreement**.

---

## **Contact**

For questions related to the paper or codebase, please contact:  
**Vishnupriya S. Devarajulu**
