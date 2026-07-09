# Anemia Detection Using Eye and Nail Images

## Overview

This project presents a deep learning-based approach for detecting anemia using images of the **eye conjunctiva** and **fingernails**. The objective is to provide a non-invasive and accessible method for anemia screening by analyzing visual features extracted from these regions.

The system employs a Convolutional Neural Network (CNN) for feature extraction, followed by an XGBoost classifier for binary classification into **Anemic** and **Non-Anemic** categories. The project also incorporates SHAP (SHapley Additive exPlanations) to improve model interpretability and a Gradio interface for interactive predictions.

---

## Features

- Non-invasive anemia detection using medical images
- Supports both eye conjunctiva and fingernail images
- CNN-based feature extraction
- XGBoost classifier for final prediction
- Image preprocessing and augmentation
- SHAP explainability for model interpretation
- Gradio web interface for predictions
- Performance evaluation using standard classification metrics

---

## Technologies Used

- Python
- TensorFlow / Keras
- XGBoost
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- SHAP
- Gradio
- Matplotlib

---

## Dataset

The project utilizes two image datasets:

- Eye conjunctiva images
- Fingernail images

Each dataset contains images belonging to:

- Anemic
- Non-Anemic

**Note:** The datasets are not included in this repository due to their size. Please download the datasets separately and update the dataset paths before running the notebook.

---

## Methodology

1. Load eye and nail image datasets.
2. Perform image preprocessing and normalization.
3. Split the datasets into training and testing sets.
4. Train a CNN to extract discriminative image features.
5. Train an XGBoost classifier using the extracted features.
6. Evaluate model performance using classification metrics.
7. Generate SHAP explanations for model predictions.
8. Deploy the trained model using a Gradio interface.

---

## Results

Performance obtained on the test dataset:

| Metric | Value |
|---------|------:|
| Accuracy | **93.5%** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|----------:|-------:|---------:|
| Non-Anemic | 1.00 | 0.90 | 0.95 |
| Anemic | 0.85 | 1.00 | 0.92 |

---

## Repository Structure

```
Anemia-Detection/
│
├── anemia_detection.ipynb
├── README.md
├── requirements.txt
├── images/
│   ├── architecture.png
│   ├── sample_predictions.png
│   └── confusion_matrix.png
└── model/
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```
anemia_detection.ipynb
```

Update the dataset paths according to your local system and execute all cells.

---

## Future Improvements

- Increase dataset diversity with images from different populations.
- Improve performance using modern CNN architectures such as EfficientNet or Vision Transformers.
- Develop a mobile application for on-device anemia screening.
- Extend the system to estimate anemia severity rather than binary classification.
- Validate the model on larger clinical datasets.

---

## Author

**Saana Shijo**

B.E. Electronics and Communication Engineering  
RV COllege of Engineering

---

## Disclaimer

This project was developed for academic and research purposes. It is not intended to replace professional medical diagnosis or clinical testing.
