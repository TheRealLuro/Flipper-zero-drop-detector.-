# Model Training & Optimization Module

This folder contains only the machine learning components for the Motion Intelligence Detector system.

It is responsible for training, evaluating, optimizing, and exporting motion classification models used by the Flipper Zero firmware.

This module runs entirely off-device and produces lightweight models designed for embedded inference.

---

## Purpose

This module focuses strictly on model development tasks:

* Training motion classification models from IMU feature datasets
* Cleaning and normalizing datasets
* Evaluating different model approaches
* Compressing models for embedded constraints
* Fine-tuning models using user-collected data
* Exporting models into deployable format

---

## Core Components

### 1. Model Training

Trains the primary motion classification model using labeled feature windows.

* Decision tree training (primary model)
* Dataset splitting and validation
* Performance evaluation across motion classes

---

### 2. Data Cleaning & Normalization

Prepares datasets for training.

* Removes invalid or noisy samples
* Normalizes feature values
* Standardizes feature vectors
* Balances class distribution when needed

---

### 3. Model Fine-Tuning

Supports updating models using new user-collected data.

* Imports new dataset from device
* Merges with existing training data
* Retrains or updates model parameters
* Outputs personalized model variant

---

### 4. Model Compression

Reduces model size for embedded deployment.

* Decision tree pruning (depth reduction)
* Feature reduction and selection
* Quantization of thresholds
* Memory footprint optimization

---

### 5. Alternative Model Experiments

Used for comparison and benchmarking.

* Simple rule-based classifiers
* Shallow decision trees
* Lightweight hybrid variants

---

### 6. Model Export

Converts trained models into a deployable format.

* Decision tree → C-compatible logic
* Hardcoded inference rules
* Embedded-friendly structure generation

---

## Outputs

This module produces:

* Trained motion classification model
* Compressed/pruned model version
* Quantized feature thresholds
* Exported C inference logic
* Fine-tuned user-specific models

---

## Workflow

```
Dataset (IMU Features)
        ↓
Data Cleaning & Normalization
        ↓
Model Training
        ↓
Evaluation
        ↓
Compression
        ↓
Model Export
```

---

## Summary

This folder contains only the machine learning pipeline for motion classification. It handles dataset preparation, model training, optimization, and export into a lightweight format suitable for embedded execution on the Flipper Zero.
