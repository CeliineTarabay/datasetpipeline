import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP

# 2.1 Read Data
df = pd.read_csv("dataset2.csv")

# 2.2 Quick Peek
print(df.head())
print(df.info())
print(df.describe())

# Clean and preprocess
# Select relevant columns
selected_columns = [
    
    "Temperature (°C)",
    "Relative Humidity (%)",
    "Absolute Humidity (g/m³)",
    "Barometric Pressure (mmHg)",
    "Dew Point (°C)",
    "Wind Chill (°C)",
    "Humidex (°C)",
    "Altitude (m)",
    "Speed (m/s)",
    "UV Index",
    "Illuminance (lx)",
    "Solar Irradiance (W/m²)",
    "Solar PAR (μmol/m²/s)",
    "Wind Direction (°)",
    "Magnetic Heading (°)",
    "True Heading (°)",
    "Temperature (°C)_Calibrata",
    "Relative Humidity (%)_Calibrata",
    "distance_to_start",
    "distance_to_end",
    "Distance2Path",
    "Wind Speed (km/hr) Run 1",
    "DayOfYear",
    "Year",
    "SecondsOfDay",
    "DayOfWeek",
    "DaysFromJuly15",
    "Skin Conductance (microS)",
    "Phasic Skin Conductance (a.u.)",
    "Tonic Skin Conductance (a.u.)",
    "Skin Conductance Phasic Driver (a.u.)",
    "Heart Rate (bpm)",
    "Emotional Index (a.u.)",
    "Sympathovagal Balance (a.u.)",
    
]

df_cluster = df[selected_columns].copy()
print(df_cluster.head())


# Optional: outlier clipping at 1st and 99th percentile
for col in df_cluster.columns:
    lower, upper = df_cluster[col].quantile([0.01, 0.99])
    df_cluster[col] = df_cluster[col].clip(lower, upper)
    print(f"{col}: Lower={lower}, Upper={upper}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# X_scaled is now a NumPy array
print("Scaled data shape:", X_scaled.shape)

# Dimentionality reduction
pca = PCA(n_components=2)  # for visualization
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_.sum())

df_cluster['PC1'] = X_pca[:, 0]  # First principal component
df_cluster['PC2'] = X_pca[:, 1]  # Second principal component

# Assuming you've already fitted PCA:
pca_components = pd.DataFrame(pca.components_, columns=df_cluster.columns[:-2], index=["PC1", "PC2"])

# Plot the contribution of variables for PC1 and PC2
for pc in ["PC1", "PC2"]:
    pca_sorted = pca_components.loc[pc].abs().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(pca_sorted.index, pca_sorted.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Variable Contribution to {pc}")
    plt.ylabel("Contribution Magnitude")
    plt.xlabel("Variables")
    plt.tight_layout()
    plt.show()


# Choose and Apply a Clustering Algorithm
# K-Means (as a Starting Point)


# Decide number of clusters k
# Elbow method (inertia plot)

inertias = []
K_range = range(2, 10)  # for example, k = 2..9
for k in K_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    kmeans_test.fit(X_scaled)
    inertias.append(kmeans_test.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia (sum of squared distances)')
plt.title('Elbow Method for K-Means')
plt.show()

# Compute the Calinski-Harabasz Index
# Assuming X_scaled is your scaled data and labels are from k-means

kmeans = KMeans(n_clusters=4, random_state=42)  
labels = kmeans.fit_predict(X_scaled)

ch_score = calinski_harabasz_score(X_scaled, labels)
print(f"Calinski-Harabasz Index: {ch_score}")

ch_scores = []
k_values = range(2, 10)  # Test k from 2 to 9

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
    labels = kmeans.labels_
    score = calinski_harabasz_score(X_scaled, labels)
    ch_scores.append(score)
    print(f"k={k}, Calinski-Harabasz Index: {score}")

# Plot CH scores
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(k_values, ch_scores, marker='o')
plt.title('Calinski-Harabasz Index vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Calinski-Harabasz Index')
plt.grid(True)
plt.show()




# Train Final K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

# Define cluster label dictionary
cluster_label_dict = {
    0: "Warm & Dry / Moderate Stress",
    1: "Warm & Dry / Elevated Stress",
    2: "Cool & Humid / Low Stress",
    3: "Warm & Humid / High Stress"
}

# Map cluster labels to descriptive names
df_cluster["Cluster Label"] = df_cluster["Cluster"].map(cluster_label_dict)

# Check if all columns are numeric, and handle non-numeric issues
print("Checking data types before grouping:")
print(df_cluster.dtypes)

# Ensure all columns except 'Cluster' are numeric
for col in df_cluster.columns:
    if col != 'Cluster':  # Assuming 'Cluster' is the column you group by
        df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')

# Verify the 'Cluster' column exists and is valid
if 'Cluster' not in df_cluster.columns:
    raise KeyError("The 'Cluster' column is missing from the DataFrame.")

# Group by 'Cluster' and compute the mean
cluster_summary = df_cluster.groupby('Cluster').mean()
print(cluster_summary)


# Explore and Interpret the Clusters
cluster_summary = df_cluster.groupby('Cluster').mean()

# Numerical summaries
cluster_summary = df_cluster.groupby('Cluster').mean()
print(cluster_summary)

# Possibly also .describe() or .median()
cluster_median = df_cluster.groupby('Cluster').median()


# Add the descriptive labels to your dataframe
df_cluster["Cluster Label"] = df_cluster["Cluster"].map(cluster_label_dict)

# Create a PCA scatter plot with annotations
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Cluster Label",
    data=df_cluster,
    palette="viridis",
    alpha=0.7
)

# Add annotations for cluster names (centroids)
for cluster_id, label in cluster_label_dict.items():
    cluster_data = df_cluster[df_cluster["Cluster"] == cluster_id]
    centroid = cluster_data[["PC1", "PC2"]].mean()
    plt.text(
        centroid["PC1"],
        centroid["PC2"],
        label,
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )

plt.title("Clusters in PCA Space with Domain-Friendly Labels")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# Group by clusters and compute averages
cluster_summary = df_cluster.groupby("Cluster Label").mean()

# Visualize in PCA space
pca_components = pd.DataFrame(pca.components_, columns=selected_columns, index=["PC1", "PC2"])
print(pca_components.T.sort_values(by="PC1", ascending=False))  # Top variables for PC1
print(pca_components.T.sort_values(by="PC2", ascending=False))  # Top variables for PC2

# Select only numeric columns for the grouping
numeric_columns = df_cluster.select_dtypes(include=[np.number]).columns

# Group by clusters and compute the mean for numeric columns only
cluster_summary = df_cluster[numeric_columns].groupby(df_cluster["Cluster"]).mean()

# Save the cluster summary to CSV
cluster_summary.to_csv("cluster_summary.csv")

# Print the summary to the console
print("Cluster Summary (Mean Values):")
print(cluster_summary)

#Visualize over time
df_cluster["Time"] = df["Time since start (s)"]  # attach the time from original df

plt.figure(figsize=(12, 5))
sns.lineplot(x="Time", y="Cluster", data=df_cluster, marker="o")
plt.title("Cluster Transitions Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Cluster")
plt.grid(True)
plt.show()


# Perform 3D PCA
pca = PCA(n_components=3)
X_pca_3d = pca.fit_transform(X_scaled)

# Plot in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=df_cluster['Cluster'],
    cmap='viridis',
    alpha=0.7
)
plt.colorbar(scatter, label="Cluster")
ax.set_title("Clusters in 3D PCA Space")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()

#umap
umap = UMAP(n_neighbors=15, random_state=42)
X_umap = umap.fit_transform(X_scaled)

sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=df_cluster['Cluster'], palette='viridis', alpha=0.7)
plt.title("UMAP Clustering")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.show()



tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df_cluster['Cluster'], palette='viridis', alpha=0.7)
plt.title("t-SNE Clustering")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.show()

