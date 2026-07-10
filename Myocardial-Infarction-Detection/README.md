# Integrating Manual ECG Feature Extraction with Ensemble Learning for Myocardial Infarction Detection

## Overview

This project presents an automated system for detecting **Myocardial Infarction (MI)** from electrocardiogram (ECG) data using manually engineered ECG features and ensemble machine learning techniques.

Unlike deep learning approaches that rely on end-to-end feature learning, this work focuses on extracting clinically meaningful ECG features and combining multiple machine learning models through ensemble learning to achieve accurate and interpretable predictions.

An additional image-based ECG classification framework was also developed to demonstrate the applicability of the proposed methodology on ECG images.

---

## Features

- Automated ECG signal preprocessing
- Manual extraction of clinically relevant ECG features
- Feature scaling and dataset balancing using SMOTE
- Ensemble learning using Random Forest, SVM, and XGBoost
- Explainable AI using SHAP
- Image-based ECG classification framework
- Model serialization for deployment

---

## Methodology

### ECG Signal-Based Framework

1. ECG signal preprocessing
2. Noise removal and signal validation
3. Manual extraction of ECG features
4. Feature standardization
5. Dataset balancing using SMOTE
6. Training individual machine learning models
7. Weighted ensemble prediction
8. Model evaluation using standard classification metrics

---

### Machine Learning Models

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The final prediction is obtained using a weighted probability voting ensemble of the three classifiers.

---

## Datasets

### PTB-XL Dataset

Used for the primary ECG signal-based myocardial infarction detection model.

Approximately **5,682** ECG recordings were used after preprocessing.

---

### Mendeley ECG Image Dataset

Used for the additional image-based myocardial infarction detection framework.

The dataset consists of **601 twelve-lead ECG images**, including Myocardial Infarction and Normal ECG classes.

---

## Technologies Used

- Python
- NumPy
- Pandas
- NeuroKit2
- Scikit-learn
- XGBoost
- OpenCV
- SHAP
- Matplotlib
- Seaborn
- Imbalanced-Learn (SMOTE)
- Pickle

---

## Performance

### ECG Signal-Based Ensemble Model

| Metric | Value |
|---------|-------|
| Accuracy | **92.04%** |
| Precision | **92.08%** |
| Recall | **92.04%** |
| F1 Score | **92.03%** |
| ROC-AUC | **0.9713** |

---

### Individual Model Accuracy

| Model | Accuracy |
|---------|----------|
| Random Forest | 91.00% |
| SVM | 87.20% |
| XGBoost | 91.00% |
| Ensemble Model | **92.04%** |

---

## Repository Structure

```
Myocardial-Infarction-Detection/
│
├── Final_MI_Detection_Improved_92.ipynb
├── final_ensemble_model.pkl
├── scaler.pkl
├── selector.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
└── images/
```

---


## Results

The proposed ensemble learning framework demonstrates that manually engineered ECG features can effectively identify myocardial infarction while maintaining model interpretability.

Compared to individual classifiers, the ensemble model achieved improved classification performance and excellent discrimination capability with a ROC-AUC of **0.9713**.

The additional image-based framework further validates the effectiveness of manual feature extraction combined with ensemble learning for ECG image classification.

---

## Future Work

- Deployment as a web-based clinical decision support system
- Integration with real-time ECG acquisition hardware
- Extension to multi-class cardiovascular disease classification
- Validation on larger and more diverse clinical datasets

---

## Acknowledgements

- PTB-XL ECG Dataset
- Mendeley ECG Image Dataset
- NeuroKit2
- Scikit-learn
- XGBoost
