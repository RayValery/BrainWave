import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# === Load features ===
features_path = Path("outputs/features.csv")
df = pd.read_csv(features_path)

# === Drop unknown labels (optional) ===
df = df[df["label"] != "unknown"]

sns.pairplot(df, hue="label", vars=["delta", "theta", "alpha", "beta"])

# === Prepare X (features) and y (labels) ===
X = df[["delta", "theta", "alpha", "beta"]]
y = df["label"]

# === Standardize X ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split into train and test ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# === Train a simple model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ F1-score:", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# === Plot confusion matrix ===
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
