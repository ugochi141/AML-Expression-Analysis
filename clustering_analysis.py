import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

print("Running Clustering Analysis to Identify AML Subtypes...")

# Load expression data
print("Loading data...")
expr_data = pd.read_csv("aml_expression_matrix.csv", index_col=0, nrows=5000)
expr_data = expr_data.apply(pd.to_numeric, errors='coerce')

# Log transform if needed
if expr_data.max().max() > 100:
    expr_data = np.log2(expr_data.clip(lower=0.1))

# Select top variable genes
gene_var = expr_data.var(axis=1)
top_genes = gene_var.nlargest(500).index
expr_subset = expr_data.loc[top_genes]

# Transpose for clustering samples
X = expr_subset.T
print(f"Clustering {X.shape[0]} samples using {X.shape[1]} genes")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-means clustering
print("\n1. Performing K-means clustering...")
# Determine optimal number of clusters
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

# Use k=4 for final clustering (adjust based on elbow)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# 2. Hierarchical clustering
print("\n2. Performing hierarchical clustering...")
# Calculate linkage
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram (simplified)
plt.subplot(1, 2, 2)
dendrogram(linkage_matrix, 
           no_labels=True,
           color_threshold=0.7*max(linkage_matrix[:,2]))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample')
plt.ylabel('Distance')

plt.tight_layout()
plt.savefig('plots/clustering_analysis.png', dpi=150)
plt.close()

# 3. Visualize clusters with PCA
print("\n3. Visualizing clusters...")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cluster_labels, 
                     cmap='Set1', 
                     s=50, 
                     alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title(f'AML Sample Clusters (k={optimal_k})')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('plots/cluster_pca_visualization.png', dpi=150)
plt.close()

# 4. Characterize clusters
print("\n4. Characterizing clusters...")
cluster_df = pd.DataFrame({
    'sample': X.index,
    'cluster': cluster_labels
})
cluster_df.to_csv('results/sample_clusters.csv', index=False)

# Find genes that distinguish clusters
cluster_profiles = []
for i in range(optimal_k):
    cluster_samples = X.index[cluster_labels == i]
    cluster_mean = expr_subset[cluster_samples].mean(axis=1)
    cluster_profiles.append(cluster_mean)

cluster_profiles_df = pd.DataFrame(cluster_profiles).T
cluster_profiles_df.columns = [f'Cluster_{i}' for i in range(optimal_k)]

# Find top genes for each cluster
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i in range(optimal_k):
    # Get genes most highly expressed in this cluster
    cluster_specific = cluster_profiles_df[f'Cluster_{i}'] - cluster_profiles_df.drop(f'Cluster_{i}', axis=1).mean(axis=1)
    top_cluster_genes = cluster_specific.nlargest(10)
    
    axes[i].barh(range(len(top_cluster_genes)), top_cluster_genes.values)
    axes[i].set_yticks(range(len(top_cluster_genes)))
    axes[i].set_yticklabels([str(g)[:20] + '...' if len(str(g)) > 20 else str(g) 
                            for g in top_cluster_genes.index], fontsize=8)
    axes[i].set_xlabel('Relative Expression')
    axes[i].set_title(f'Top Genes for Cluster {i}')
    axes[i].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/cluster_specific_genes.png', dpi=150)
plt.close()

# Save cluster profiles
cluster_profiles_df.to_csv('results/cluster_expression_profiles.csv')

# Summary report
summary = f"""
Clustering Analysis Summary
==========================
Samples analyzed: {len(X)}
Genes used: {len(top_genes)}
Optimal clusters: {optimal_k}

Cluster Distribution:
"""
for i in range(optimal_k):
    count = sum(cluster_labels == i)
    summary += f"\nCluster {i}: {count} samples ({count/len(X)*100:.1f}%)"

summary += """

Files Generated:
- plots/clustering_analysis.png: Elbow plot and dendrogram
- plots/cluster_pca_visualization.png: PCA with cluster colors
- plots/cluster_specific_genes.png: Top genes per cluster
- results/sample_clusters.csv: Cluster assignments
- results/cluster_expression_profiles.csv: Mean expression per cluster

These clusters may represent different AML subtypes with distinct molecular signatures.
"""

print(summary)

with open('results/clustering_summary.txt', 'w') as f:
    f.write(summary)
