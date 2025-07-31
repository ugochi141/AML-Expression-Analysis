# AML Gene Expression Analysis Portfolio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aml-expression-analysis-3bkve4lz3l4xmwfhjyvz7r.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/ugochi141/AML-Expression-Analysis)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![GUI](https://img.shields.io/badge/GUI-Tkinter-green.svg)](https://docs.python.org/3/library/tkinter.html)

**Live Dashboard:** https://aml-expression-analysis-3bkve4lz3l4xmwfhjyvz7r.streamlit.app/

# AML Gene Expression Analysis Portfolio

## 🚀 Quick Start

### Option 1: Desktop GUI Application (Recommended)

**🎯 User-friendly interface - No coding required!**

```bash
# Clone the repository
git clone https://github.com/ugochi141/AML-Expression-Analysis.git
cd AML-Expression-Analysis

# Install dependencies
pip install -r requirements.txt

# Launch GUI application
python aml_gui_app.py
```

**Or double-click these files:**
- Windows: `launch_aml.bat`
- macOS/Linux: `launch_aml.sh`
- Python: `launch_aml_gui.py`

### Option 2: Streamlit Web Dashboard

```bash
# Launch web dashboard
streamlit run aml_dashboard.py
```

### Option 3: Command Line Analysis

```bash
# Run complete analysis
python aml_analysis_complete.py

# Run specific components
python pca_analysis.py
python clustering_analysis.py
```

## 🧬 Features

### GUI Application
- **Interactive Data Loading**: Easy file selection and data preview
- **Comprehensive Analysis**: Basic statistics, PCA, clustering, and more
- **Real-time Visualization**: Interactive plots and charts
- **Report Generation**: Automated HTML and PDF reports
- **Export Capabilities**: Save results in multiple formats

### Analysis Capabilities
- **Gene Expression Profiling**: Statistical analysis of expression patterns
- **Principal Component Analysis**: Dimensionality reduction and visualization
- **Clustering Analysis**: Sample and gene clustering
- **Differential Expression**: Identify significantly expressed genes
- **Biomarker Discovery**: Find potential diagnostic markers

### Visualization Tools
- **Expression Heatmaps**: Gene and sample expression patterns
- **PCA Plots**: Sample relationships and clustering
- **Statistical Plots**: Distribution analysis and comparisons
- **Interactive Charts**: Plotly-based interactive visualizations

## 📊 Dataset Information

- **Samples**: 537 AML patient samples
- **Genes**: 54,675 gene probes
- **Platform**: Affymetrix HG-U133 Plus 2.0
- **Data Size**: ~329 MB
- **Source**: Gene Expression Omnibus (GEO)

## 🔧 System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space
- Windows, macOS, or Linux

## 📁 Project Structure

```
AML-Expression-Analysis/
├── aml_gui_app.py              # Main GUI application
├── aml_dashboard.py            # Streamlit web dashboard
├── aml_analysis_complete.py    # Complete analysis script
├── pca_analysis.py            # PCA analysis
├── clustering_analysis.py     # Clustering analysis
├── launch_aml_gui.py          # GUI launcher
├── launch_aml.bat/.sh         # Platform launchers
├── requirements.txt           # Dependencies
├── data/                      # Data files
├── plots/                     # Generated plots
├── results/                   # Analysis results
└── reports/                   # Generated reports
```

## 🎯 Use Cases

### Research Applications
- **Cancer Biology**: Study AML molecular mechanisms
- **Biomarker Discovery**: Identify diagnostic/prognostic markers
- **Drug Target Discovery**: Find therapeutic targets
- **Patient Stratification**: Classify AML subtypes

### Educational Applications
- **Bioinformatics Training**: Learn gene expression analysis
- **Data Science Education**: Practice with real biological data
- **Statistical Analysis**: Understand genomic statistics
- **Visualization Techniques**: Create publication-quality plots

# My Analysis Portfolio
