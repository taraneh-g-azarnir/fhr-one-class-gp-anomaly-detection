# Fetal Well-Being Prediction with One-Class Gaussian Process Anomaly Detection

Code for the EUSIPCO 2025 paper on one-class Gaussian Process–based anomaly detection for fetal heart rate (FHR) analysis.

---

## Overview

Fetal heart rate (FHR) monitoring is essential for assessing fetal well-being, but detecting pathological patterns is challenging due to limited labeled abnormal data.

This repository implements a one-class learning framework based on Gaussian Processes to detect abnormal FHR patterns using only healthy training data.

---

## Paper

This repository corresponds to the following paper:

Fetal Well-Being Prediction with One-Class Gaussian Process Anomaly Detection  
EUSIPCO 2025

---

## Repository Structure
```text
fhr-one-class-gp-anomaly-detection/
├── src/
├── results/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Data

This repository expects a feature table extracted from fetal heart rate signals.

Feature extraction is provided in the companion repository:
https://github.com/taraneh-g-azarnir/fhr-feature-extraction

The dataset used in this work is based on:
https://preana-fo.ece.stonybrook.edu/database.html

---

## Installation
```bash
git clone https://github.com/taraneh-g-azarnir/fhr-one-class-gp-anomaly-detection.git
cd fhr-one-class-gp-anomaly-detection
pip install -r requirements.txt
```
---

## Usage

Run the main pipeline:

```bash
python src/your_main_script.py
```
---

## Method Summary

The pipeline includes:

- preprocessing of FHR feature data  
- training one-class Gaussian Process models on healthy samples  
- computing anomaly scores  
- evaluating detection performance on abnormal cases  

---

## Output

The pipeline produces:

- anomaly scores for each sample  
- classification decisions  
- evaluation metrics (e.g., AUROC, precision-recall)  

---

## Author

Taraneh Ghanbari Azarnir  
PhD Candidate, Electrical Engineering  
Stony Brook University  
