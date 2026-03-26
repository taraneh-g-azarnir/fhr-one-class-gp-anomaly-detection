# Fetal Well-Being Prediction with One-Class Gaussian Process Anomaly Detection

This repository contains the implementation of the method proposed in:

**“Fetal Well-Being Prediction with One-Class Gaussian Process Anomaly Detection”**  
*EUSIPCO 2025*

---

## Overview

Fetal heart rate (FHR) monitoring is a key tool for assessing fetal well-being during pregnancy. However, detecting pathological patterns remains challenging due to the scarcity of labeled abnormal data.

This work formulates fetal risk assessment as a **one-class anomaly detection problem**, where models are trained exclusively on **healthy (CAT-1)** data and used to detect deviations corresponding to **pathological (CAT-3)** conditions.

We propose a framework based on **Gaussian Process (GP) regression** to model the conditional distribution of each feature and derive interpretable anomaly scores.

---

## Method Summary

### 1. Feature Representation
FHR signals are segmented and transformed into clinically meaningful features.

### 2. One-Class Gaussian Process Modeling
Each feature is modeled using a Gaussian Process conditioned on the remaining features.

### 3. Anomaly Scoring
Standardized residuals are used to compute anomaly scores.

### 4. Ensemble Detection
Per-feature scores are combined into a global anomaly score.

---

## Data

### Feature File
Each row corresponds to a 10-minute FHR segment.

### Label File
Contains:
cat ∈ {CAT-1, CAT-3}

Each label corresponds to 3 consecutive segments.

---

## Installation

```bash
git clone https://github.com/taraneh-g-azarnir/fhr-one-class-gp-anomaly-detection.git
cd fhr-one-class-gp-anomaly-detection
pip install -r requirements.txt
```

---

## Usage

```bash
python src/main.py   --features data/fhr_features.xlsx   --labels data/labels.xlsx   --output results/
```

---

## Output

- Anomaly scores  
- Confidence scores  
- Evaluation metrics  

---

## Citation

```bibtex
@inproceedings{ghanbari2025ocgp,
  title={Fetal Well-Being Prediction with One-Class Gaussian Process Anomaly Detection},
  author={Ghanbari Azarnir, Taraneh and others},
  booktitle={EUSIPCO},
  year={2025}
}
```

---

## Author

Taraneh Ghanbari Azarnir  
PhD Candidate, Electrical Engineering  
Stony Brook University  
