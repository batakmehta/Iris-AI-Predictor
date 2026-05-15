#pip install pandas numpy scikit-learn streamlit matplotlib
import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATASET
# ==========================================
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("\n========== DATA LOADED ==========")
print(df.head())


# ==========================================
# 2. FEATURES & TARGET
# ==========================================
X = df[iris.feature_names]
y = df["target"]


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ==========================================
# 4. MODEL PIPELINES
# ==========================================
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=300))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear", probability=True))
    ])
}


# ==========================================
# 5. TRAIN + EVALUATE
# ==========================================
best_model = None
best_score = 0
best_name = ""

print("\n========== TRAINING MODELS ==========")

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation (more realistic)
    cv_score = cross_val_score(model, X, y, cv=5).mean()

    print(f"Accuracy      : {acc:.4f}")
    print(f"CV Score      : {cv_score:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Track best model
    if cv_score > best_score:
        best_score = cv_score
        best_model = model
        best_name = name


# ==========================================
# 6. BEST MODEL
# ==========================================
print("\n========== BEST MODEL ==========")
print("Model :", best_name)
print("Score :", round(best_score, 4))


# ==========================================
# 7. SAVE MODEL
# ==========================================
with open("all_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("\nModel saved as 'best_iris_model.pkl'")


# ==========================================
# 8. TEST PREDICTION (NO WARNING VERSION)
# ==========================================
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)

prediction = best_model.predict(sample)[0]
probabilities = best_model.predict_proba(sample)[0]

print("\n========== SAMPLE PREDICTION ==========")
print("Input:", sample.values.tolist())
print("Predicted class:", iris.target_names[prediction])
print("Probabilities:", probabilities)