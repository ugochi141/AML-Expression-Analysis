import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

print("Starting AML Expression Analysis...")
print("=" * 50)

# Create output directories
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Step 1: Load and check the data
print("\n1. Loading expression data...")
# Read just the first row to get dimensions
first_row = pd.read_csv("aml_expression_matrix.csv", nrows=1)
n_samples = len(first_row.columns) - 1  # -1 for index column
print(f"   Number of samples: {n_samples}")

# Count genes without loading full dataset
with open("aml_expression_matrix.csv", 'r') as f:
    n_genes = sum(1 for line in f) - 1
print(f"   Number of genes: {n_genes}")

# Step 2: Calculate gene statistics in chunks
print("\n2. Calculating gene statistics...")
chunk_size = 5000
gene_means = []
gene_vars = []
gene_names = []

for chunk in pd.read_csv("aml_expression_matrix.csv", index_col=0, chunksize=chunk_size):
    # Convert to numeric
    chunk = chunk.apply(pd.to_numeric, errors='coerce')
    
    # Log transform if needed (check first chunk only)
    if len(gene_means) == 0 and chunk.max().max() > 100:
        print("   Applying log2 transformation...")
        chunk = np.log2(chunk + 1)
    elif len(gene_means) > 0:  # Apply same transformation as first chunk
        chunk = np.log2(chunk + 1)
    
    # Calculate statistics
    gene_means.extend(chunk.mean(axis=1).tolist())
    gene_vars.extend(chunk.var(axis=1).tolist())
    gene_names.extend(chunk.index.tolist())
    
    print(f"   Processed {len(gene_means)} genes...", end='\r')

print(f"\n   Total genes processed: {len(gene_means)}")

# Create statistics dataframe
stats_df = pd.DataFrame({
    'gene': gene_names,
    'mean': gene_means,
    'variance': gene_vars
})
stats_df.set_index('gene', inplace=True)

# Step 3: Find top variable genes
print("\n3. Identifying top variable genes...")
top_var_genes = stats_df.nlargest(100, 'variance')
top_var_genes.to_csv("results/top_100_variable_genes.csv")
print(f"   Saved top 100 variable genes")

# Step 4: Create visualizations
print("\n4. Creating visualizations...")

# 4.1 Gene statistics plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Mean distribution
axes[0,0].hist(stats_df['mean'], bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0,0].set_xlabel('Mean Log2 Expression')
axes[0,0].set_ylabel('Number of Genes')
axes[0,0].set_title('Distribution of Mean Expression')
axes[0,0].grid(True, alpha=0.3)

# Variance distribution
axes[0,1].hist(stats_df['variance'], bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0,1].set_xlabel('Variance')
axes[0,1].set_ylabel('Number of Genes')
axes[0,1].set_title('Distribution of Variance')
axes[0,1].grid(True, alpha=0.3)

# Mean vs Variance
axes[1,0].scatter(stats_df['mean'], stats_df['variance'], alpha=0.5, s=1)
axes[1,0].set_xlabel('Mean Expression')
axes[1,0].set_ylabel('Variance')
axes[1,0].set_title('Mean-Variance Relationship')
axes[1,0].grid(True, alpha=0.3)

# Top 20 variable genes
top20_var = stats_df.nlargest(20, 'variance')
axes[1,1].barh(range(len(top20_var)), top20_var['variance'])
axes[1,1].set_yticks(range(len(top20_var)))
axes[1,1].set_yticklabels([str(idx)[:15] + '...' if len(str(idx)) > 15 else str(idx) 
                           for idx in top20_var.index])
axes[1,1].set_xlabel('Variance')
axes[1,1].set_title('Top 20 Most Variable Genes')
axes[1,1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/gene_statistics_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Created gene statistics overview")

# 4.2 Expression level categories
expr_categories = pd.cut(stats_df['mean'], 
                        bins=[-np.inf, 2, 5, 8, np.inf],
                        labels=['Low', 'Medium', 'High', 'Very High'])
expr_counts = expr_categories.value_counts()

plt.figure(figsize=(8, 6))
expr_counts.plot(kind='bar', color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
plt.xlabel('Expression Level')
plt.ylabel('Number of Genes')
plt.title('Gene Expression Level Categories')
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/expression_categories.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Created expression categories plot")

# Step 5: Load and analyze sample information
print("\n5. Analyzing sample information...")
sample_info = pd.read_csv("aml_sample_info.csv", index_col=0)
print(f"   Loaded info for {len(sample_info)} samples")

# Save first few rows as example
sample_info.head(10).to_csv("results/sample_info_example.csv")

# Step 6: Create summary report
print("\n6. Creating summary report...")
summary = f"""AML Expression Analysis Summary Report
=====================================
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Overview:
-----------------
- Total Samples: {n_samples}
- Total Genes/Probes: {n_genes}
- Data points: ~{n_samples * n_genes:,}

Expression Statistics:
---------------------
- Mean expression range: {stats_df['mean'].min():.2f} to {stats_df['mean'].max():.2f}
- Variance range: {stats_df['variance'].min():.4f} to {stats_df['variance'].max():.2f}
- Genes with low expression (<2): {(stats_df['mean'] < 2).sum():,}
- Genes with high expression (>8): {(stats_df['mean'] > 8).sum():,}
- Highly variable genes (variance >1): {(stats_df['variance'] > 1).sum():,}

Top 5 Most Variable Genes:
-------------------------"""

for i, (gene, row) in enumerate(top_var_genes.head().iterrows(), 1):
    summary += f"\n{i}. {gene}: variance = {row['variance']:.3f}, mean = {row['mean']:.3f}"

summary += """

Output Files Generated:
----------------------
1. plots/gene_statistics_overview.png - Comprehensive gene statistics
2. plots/expression_categories.png - Expression level distribution
3. results/top_100_variable_genes.csv - Most variable genes list
4. results/sample_info_example.csv - Sample information preview

Next Steps:
-----------
1. Perform PCA on top variable genes
2. Cluster samples to identify AML subtypes
3. Compare with known AML gene signatures
4. Pathway enrichment analysis
"""

with open("results/analysis_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("\n" + "="*50)
print("Analysis complete! Check the 'plots' and 'results' directories.")
