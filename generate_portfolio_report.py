import os
import glob
from datetime import datetime
import pandas as pd

print("Generating Final Portfolio Report...")

# Create a comprehensive HTML report
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AML Expression Analysis - Ugochi Ndubuisi</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin: 10px 0;
        }}
        .skills {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        .skill-tag {{
            background-color: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
        }}
        .code-block {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AML Gene Expression Analysis</h1>
        <h2>Bioinformatics Portfolio Project</h2>
        <p>Ugochi Ndubuisi | {datetime.now().strftime('%B %Y')}</p>
    </div>

    <div class="section">
        <h2>🎯 Project Overview</h2>
        <p>
            This comprehensive bioinformatics project analyzes gene expression patterns in 
            Acute Myeloid Leukemia (AML) patients, leveraging my background in hematology 
            and laboratory operations to identify potential biomarkers and molecular subtypes.
        </p>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">537</div>
                <div class="metric-label">AML Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">54,675</div>
                <div class="metric-label">Genes Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">329 MB</div>
                <div class="metric-label">Data Processed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">4</div>
                <div class="metric-label">AML Subtypes</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🔬 Background & Motivation</h2>
        <p>
            With over 6 years of experience in clinical laboratory operations and specialized 
            expertise in hematology diagnostics, I recognized the potential to apply computational 
            methods to improve AML diagnosis and treatment. This project bridges traditional 
            laboratory techniques with modern bioinformatics approaches.
        </p>
        
        <h3>Clinical Relevance</h3>
        <ul>
            <li>Identification of molecular signatures for AML subtypes</li>
            <li>Discovery of potential diagnostic biomarkers</li>
            <li>Insights into therapeutic targets</li>
            <li>Understanding of disease heterogeneity</li>
        </ul>
    </div>

    <div class="section">
        <h2>🛠️ Technical Implementation</h2>
        
        <h3>Technologies & Skills</h3>
        <div class="skills">
            <span class="skill-tag">Python</span>
            <span class="skill-tag">Pandas</span>
            <span class="skill-tag">NumPy</span>
            <span class="skill-tag">Scikit-learn</span>
            <span class="skill-tag">Matplotlib</span>
            <span class="skill-tag">Streamlit</span>
            <span class="skill-tag">GEO Database</span>
            <span class="skill-tag">Git/GitHub</span>
            <span class="skill-tag">Data Visualization</span>
            <span class="skill-tag">Machine Learning</span>
        </div>

        <h3>Key Analyses Performed</h3>
        <ol>
            <li><strong>Data Acquisition:</strong> Retrieved dataset from GEO using GEOparse</li>
            <li><strong>Preprocessing:</strong> Log2 transformation, quality filtering</li>
            <li><strong>Exploratory Analysis:</strong> Distribution analysis, variance calculation</li>
            <li><strong>Dimensionality Reduction:</strong> PCA to identify major variation sources</li>
            <li><strong>Clustering:</strong> K-means and hierarchical clustering for subtype discovery</li>
            <li><strong>Visualization:</strong> Interactive dashboard with Streamlit</li>
        </ol>
    </div>

    <div class="section">
        <h2>📊 Key Results</h2>
        
        <h3>1. Gene Expression Patterns</h3>
        <p>Identified 100 highly variable genes that distinguish AML samples, with the top genes showing 
        potential as diagnostic markers.</p>
        
        <h3>2. AML Subtypes</h3>
        <p>Clustering analysis revealed 4 distinct molecular subtypes within the AML samples, each with 
        unique expression signatures.</p>
        
        <h3>3. Principal Components</h3>
        <p>First 3 principal components explain significant variance, revealing the major sources of 
        heterogeneity in AML.</p>
    </div>

    <div class="section">
        <h2>💻 Code Example</h2>
        <div class="code-block">
# Load and process AML expression data
expr_data = pd.read_csv("aml_expression_matrix.csv", index_col=0)
expr_data = np.log2(expr_data.clip(lower=0.1))

# Identify top variable genes
gene_variance = expr_data.var(axis=1)
top_genes = gene_variance.nlargest(100)

# Perform PCA
pca = PCA(n_components=10)
pca_result = pca.fit_transform(expr_data.T)
        </div>
    </div>

    <div class="section">
        <h2>📁 Project Deliverables</h2>
        <ul>
            <li>✅ Complete analysis pipeline in Python</li>
            <li>✅ Interactive Streamlit dashboard</li>
            <li>✅ Comprehensive documentation</li>
            <li>✅ GitHub repository with version control</li>
            <li>✅ Publication-ready visualizations</li>
            <li>✅ Identified AML molecular subtypes</li>
        </ul>
    </div>

    <div class="section">
        <h2>🚀 Future Directions</h2>
        <ul>
            <li>Integration with clinical outcome data</li>
            <li>Machine learning models for prognosis prediction</li>
            <li>Pathway enrichment analysis</li>
            <li>Validation with independent cohorts</li>
            <li>Development of diagnostic gene panels</li>
        </ul>
    </div>

    <div class="section">
        <h2>👤 About Me</h2>
        <p>
            <strong>Ugochi Ndubuisi</strong><br>
            Laboratory Operation Manager | Bioinformatics Enthusiast<br>
            📍 Maryland, United States<br>
            📧 u.l.ndubuisi@gmail.com<br>
            💼 <a href="https://linkedin.com/in/ugochindubuisi">LinkedIn</a> | 
            🌐 <a href="https://github.com/YOUR_USERNAME">GitHub</a>
        </p>
    </div>
</body>
</html>
"""

# Save HTML report
with open('portfolio_report.html', 'w') as f:
    f.write(html_content)

print("Generated portfolio_report.html")
print("\nPortfolio report generation complete!")
print("Open portfolio_report.html in your browser to view the report.")
