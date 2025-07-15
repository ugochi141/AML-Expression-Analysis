import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("Starting simple AML analysis...")

# Create directories
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load sample info
print("Loading sample information...")
samples = pd.read_csv("aml_sample_info.csv", index_col=0)
print(f"Found {len(samples)} samples")

# Process first 5000 genes only for quick analysis
print("\nProcessing expression data (first 5000 genes)...")
expr_data = pd.read_csv("aml_expression_matrix.csv", index_col=0, nrows=5000)
print(f"Loaded {expr_data.shape[0]} genes x {expr_data.shape[1]} samples")

# Convert to numeric and log transform
expr_data = expr_data.apply(pd.to_numeric, errors='coerce')
if expr_data.max().max() > 100:
    print("Applying log2 transformation...")
    expr_data = np.log2(expr_data.clip(lower=0.1))

# Calculate basic statistics
print("\nCalculating statistics...")
gene_means = expr_data.mean(axis=1)
gene_vars = expr_data.var(axis=1)

# Simple plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
gene_means.hist(bins=50)
plt.title('Gene Expression Distribution')
plt.xlabel('Mean Log2 Expression')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
top_var = gene_vars.nlargest(20)
plt.barh(range(len(top_var)), top_var.values)
plt.yticks(range(len(top_var)), [str(g)[:20] for g in top_var.index])
plt.xlabel('Variance')
plt.title('Top 20 Variable Genes')

plt.tight_layout()
plt.savefig('plots/basic_analysis.png')
plt.close()

# Save results
top_var.to_csv('results/top_variable_genes.csv', header=['variance'])
print("\nAnalysis complete!")
print("Check plots/basic_analysis.png and results/top_variable_genes.csv")
