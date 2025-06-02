import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("outputs/features_entropy.csv")

# Drop non-feature columns
df = df.drop(columns=["subject", "run"])

# Split into features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1_macro")
accuracy_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

# Print cross-validated metrics
print("✅ Cross-validated results:")
print(f"Mean Accuracy: {accuracy_scores.mean():.2f}")
print(f"Mean F1-score: {f1_scores.mean():.2f}")

# Train on full dataset for analysis
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=["rest", "motor"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["rest", "motor"], yticklabels=["rest", "motor"])
plt.xlabel("Predicted")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Random Forest on full data)")
plt.tight_layout()
plt.show()

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

# Plot top features
plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x="Importance", y="Feature")
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
