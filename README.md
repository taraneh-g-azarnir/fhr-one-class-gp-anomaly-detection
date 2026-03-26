# Fetal Well-Being Prediction with One-Class Gaussian Process Anomaly Detection

This repository contains the implementation of the method proposed in:

**“Fetal Well-Being Prediction with One-Class Gaussian Process Anomaly Detection”**  
*EUSIPCO 2025*

---

## Overview

Fetal heart rate (FHR) monitoring is a key tool for assessing fetal well-being during pregnancy. However, detecting pathological patterns remains challenging due to lack of ground truth and imbalanced classes.

This work formulates fetal risk assessment as a **one-class anomaly detection problem**, where the model is trained exclusively on **healthy (CAT-1)** data and used to detect deviations corresponding to **pathological (CAT-3)** conditions.

We propose a **One-Class Gaussian Process (OCGP)** model trained on interpretable features extracted from healthy fetal heart rate (FHR) segments. The model learns the distribution of healthy FHR patterns and identifies deviations through predictive variance–based anomaly detection. We further introduce a **Health Confidence Score (HCS)**, a continuous measure of fetal well-being derived from the model uncertainty.

---

## Method Summary

### 1. Feature Representation
FHR signals are segmented and transformed into clinically meaningful features.

### 2. One-Class Gaussian Process Modeling

The extracted interpretable features from healthy FHR segments are used as inputs to a **One-Class Gaussian Process (OCGP)** model. All training outputs are set to 1 to represent the healthy class, and an **ARD kernel** is used to improve interpretability through feature-wise weighting.

### 3. Anomaly Detection

Anomaly detection is performed using the **predictive variance** of the One-Class Gaussian Process (OCGP) model. Since the model is trained only on healthy data, segments that deviate from the learned distribution exhibit higher predictive uncertainty and are identified as potential pathological cases.

### 4. Outlier Threshold

Anomaly detection is based on the **predictive variance** of the One-Class Gaussian Process (OCGP) model. Segments with predictive variance above a threshold are classified as unhealthy.

The threshold is determined using **5-fold cross-validation on the healthy (CAT-1) training data**. Specifically, the predictive variance is computed for validation samples, and the threshold is set as the **95th percentile of these variances**. 

This ensures that the model captures the distribution of healthy patterns while controlling the false alarm rate.

### 5. Health Confidence Score (HCS)

A continuous **Health Confidence Score (HCS)** is computed from the predictive variance of the One-Class Gaussian Process (OCGP) model to quantify fetal well-being.

Let \( \sigma^2(x) \) denote the predictive variance for a given sample \( x \), and let \( \tau \) be a reference scale (e.g., a threshold derived from training data such as the 95th percentile of training variances). The HCS is defined as:

\[
\mathrm{HCS}(x) = \exp\left(-\frac{\sigma^2(x)}{\tau}\right)
\]

Since the model is trained only on healthy data:
- **Low predictive variance** (\( \sigma^2(x) \)) results in \( \mathrm{HCS}(x) \approx 1 \), indicating strong similarity to healthy patterns  
- **High predictive variance** results in \( \mathrm{HCS}(x) \approx 0 \), indicating potential pathological conditions  

This provides a continuous, interpretable measure of fetal well-being between 0 and 1.

---

## Data

### Feature File
Each row corresponds to a 10-minute FHR segment.

### Label File
Expert label:

cat ∈ {CAT-1, CAT-3}

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
