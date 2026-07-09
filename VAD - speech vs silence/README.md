# Real-Time Voice Activity Detection (Machine Learning)

A Machine Learning-based Voice Activity Detection (VAD) system that classifies audio frames as **Speech** or **Non-Speech (Silence/Background Noise)** using time-domain and frequency-domain audio features.

This repository contains the **ML pipeline** used for the VAD project. The embedded STM32 firmware is maintained separately.

---

## Project Overview

The objective of this project is to improve bandwidth utilization in voice communication systems by detecting speech activity and suppressing silence and background noise.

The workflow consists of:

1. Audio dataset preparation
2. Feature extraction
3. Machine Learning model training
4. Real-time speech classification
5. GUI visualization

---

## Dataset

The model was trained using the **Google Speech Commands Dataset**.

- Speech recordings
- Background noise recordings
- Sampling Rate: **8 kHz**
- Frame Length: **20 ms (160 samples)**

Speech frames were labeled as:

- **1 → Speech**
- **0 → Non-Speech (Background Noise)**

---

## Feature Extraction

For every 20 ms frame, the following features are extracted:

### Time-Domain Features

- Energy
- Root Mean Square (RMS)
- Zero Crossing Rate (ZCR)
- Peak Amplitude
- Variance

### Frequency-Domain Features

- Spectral Centroid
- Spectral Flatness
- Spectral Entropy

These features are used as inputs to the Machine Learning classifier.

---

## Machine Learning Models Evaluated

The following classifiers were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

The best-performing model is automatically saved as:

```
models/vad_model.pkl
```

---

## Model Performance

| Model | Test Accuracy | 5-Fold Cross Validation |
|--------|--------------:|------------------------:|
| Logistic Regression | 80.82% | 80.65% |
| Decision Tree | 91.26% | 90.71% |
| Random Forest | 93.85% | 94.00% |
| SVM | 75.54% | 73.94% |
| **XGBoost** | **94.15%** | **93.92%** |

**Best Model:** XGBoost

---

## Repository Structure

```
VAD - speech vs silence/
│
├── images/
│   ├── output - speech.jpeg
│   └── output - silence.jpeg
│
├── feature_extraction.py
├── train_model.py
├── realtime_vad.py
├── model_results.csv
└── README.md
```

---

## Workflow

```
Google Speech Commands Dataset
            │
            ▼
      Frame Generation
            │
            ▼
    Feature Extraction
            │
            ▼
    Machine Learning Training
            │
            ▼
    Best Model Selection
            │
            ▼
   Real-Time Audio Classification
            │
            ▼
 Speech / Non-Speech Detection
```

---

## Technologies Used

- Python 3
- NumPy
- Pandas
- Librosa
- Scikit-learn
- XGBoost
- Joblib
- PySerial
- Tkinter

---

## Results

The developed system successfully classifies incoming audio frames as **Speech** or **Non-Speech** in real time.

The graphical interface displays:

- Live waveform
- Speech detection status
- Energy
- RMS
- Zero Crossing Rate (ZCR)
- Estimated bandwidth savings

Example outputs are available in the **images** folder.

---

## Future Improvements

- Deploy the trained model directly on the STM32 microcontroller.
- Replace placeholder spectral features with FFT-based computation on embedded hardware.
- Improve robustness under highly noisy environments.
- Explore lightweight deep learning models for embedded deployment.

---

## Author

**Saana Shijo**

B.E. Electronics and Communication Engineering  
RV College of Engineering
