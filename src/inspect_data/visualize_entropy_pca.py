import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load feature data
df = pd.read_csv("outputs/features_entropy.csv")

# Drop non-numeric columns
df = df.drop(columns=["subject", "run"])  # 👈 ключова зміна!

# Split into features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Prepare DataFrame for plotting
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["label"] = y

# Plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="label", palette="Set2")
plt.title("PCA Projection (entropy + Hjorth + ratios)")
plt.tight_layout()
plt.show()
