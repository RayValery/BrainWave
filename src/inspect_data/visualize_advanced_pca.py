import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Load advanced features ===
df = pd.read_csv(Path("outputs/features_advanced.csv"))
df = df[df["label"] != "unknown"]

# === Select only numerical features ===
X = df.drop(columns=["subject", "run", "label"])
y = df["label"]

# === Standardize the data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Run PCA ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === Create a DataFrame for plotting ===
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["label"] = y.values

# === Plot the PCA projection ===
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", palette="Set1")
plt.title("PCA Projection of Advanced EEG Features")
plt.tight_layout()
plt.show()
