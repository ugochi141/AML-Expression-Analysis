import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

print("Running PCA Analysis on AML Data...")

# Load the expression data (using subset for efficiency)
print("Loading expression data...")
expr_data = pd.read_csv("aml_expression_matrix.csv", index_col=0, nrows=5000)
expr_data = expr_data.apply(pd.to_numeric, errors='coerce')

# Log transform if needed
if expr_data.max().max() > 100:
    expr_data = np.log2(expr_data.clip(lower=0.1))

# Remove genes with low variance
gene_var = expr_data.var(axis=1)
top_var_genes = gene_var.nlargest(1000).index
expr_subset = expr_data.loc[top_var_genes]

print(f"Using top {len(top_var_genes)} variable genes for PCA")

# Transpose for PCA (samples as rows)
expr_t = expr_subset.T

# Standardize
scaler = StandardScaler()
expr_scaled = scaler.fit_transform(expr_t)

# Perform PCA
pca = PCA(n_components=10)
pca_result = pca.fit_transform(expr_scaled)

# Create PCA plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# PC1 vs PC2
ax1.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.set_title('PCA: PC1 vs PC2')
ax1.grid(True, alpha=0.3)

# PC2 vs PC3
ax2.scatter(pca_result[:, 1], pca_result[:, 2], alpha=0.6)
ax2.set_xlabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.set_ylabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
ax2.set_title('PCA: PC2 vs PC3')
ax2.grid(True, alpha=0.3)

# Scree plot
ax3.bar(range(1, 11), pca.explained_variance_ratio_[:10])
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Variance Explained')
ax3.set_title('Scree Plot')
ax3.grid(True, alpha=0.3)

# Cumulative variance
ax4.plot(range(1, 11), np.cumsum(pca.explained_variance_ratio_[:10]), 'bo-')
ax4.set_xlabel('Number of Components')
ax4.set_ylabel('Cumulative Variance Explained')
ax4.set_title('Cumulative Variance')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
ax4.legend()

plt.tight_layout()
plt.savefig('plots/pca_analysis.png', dpi=150)
plt.close()

# Save PCA results
pca_df = pd.DataFrame(
    pca_result[:, :3],
    columns=['PC1', 'PC2', 'PC3'],
    index=expr_t.index
)
pca_df.to_csv('results/pca_coordinates.csv')

# Save loadings (top genes contributing to each PC)
loadings = pd.DataFrame(
    pca.components_[:3].T,
    columns=['PC1', 'PC2', 'PC3'],
    index=expr_subset.index
)

# Get top contributing genes for PC1
top_pc1_genes = loadings['PC1'].abs().nlargest(20)
top_pc1_genes.to_csv('results/top_pc1_genes.csv')

print(f"\nPCA Complete!")
print(f"Total variance explained by first 3 PCs: {sum(pca.explained_variance_ratio_[:3]):.1%}")
print(f"Saved plots to plots/pca_analysis.png")
print(f"Saved PCA coordinates to results/pca_coordinates.csv")
