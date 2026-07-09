import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

# -------------------------------
# LOAD DATASET
# -------------------------------

print("Loading dataset...")

df = pd.read_csv("vad_dataset.csv")

print(df.shape)

# -------------------------------
# BALANCE DATASET
# -------------------------------

speech = df[df["Label"] == 1]
noise = df[df["Label"] == 0]

samples = min(len(speech), len(noise), 150000)

speech = speech.sample(samples, random_state=42)
noise = noise.sample(samples, random_state=42)

df = pd.concat([speech, noise])

df = df.sample(frac=1, random_state=42)

print("\nBalanced Dataset:", df.shape)

# -------------------------------
# FEATURES
# -------------------------------

X = df.drop("Label", axis=1)
y = df["Label"]

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# MODELS
# -------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "SVM":
        SVC(kernel="rbf"),

    "XGBoost":
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

}

results = []

best_model = None
best_accuracy = 0

# -------------------------------
# TRAIN
# -------------------------------

for name, model in models.items():

    print("\n==============================")
    print(name)
    print("==============================")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("Accuracy:", acc)

    cv = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print("5-Fold Accuracy:", cv.mean())

    print(classification_report(y_test, pred))

    results.append([

        name,

        acc,

        cv.mean()

    ])

    if acc > best_accuracy:

        best_accuracy = acc
        best_model = model

# -------------------------------
# SAVE BEST MODEL
# -------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/vad_model.pkl")

print("\nBest Model Saved!")

# -------------------------------
# RESULTS TABLE
# -------------------------------

results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "CV Accuracy"
    ]
)

print(results)

results.to_csv("model_results.csv", index=False)

print("\nResults saved.")