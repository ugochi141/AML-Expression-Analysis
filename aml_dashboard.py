import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob

st.set_page_config(
    page_title="AML Expression Analysis Dashboard",
    page_icon="🧬",
    layout="wide"
)

# Title and introduction
st.title("🧬 AML Gene Expression Analysis Dashboard")
st.markdown("""
**Author:** Ugochi Ndubuisi | **Dataset:** 537 AML Samples | **Genes:** 54,675
""")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Gene Expression", "PCA Analysis", "Top Genes", "About"])

# Load data with caching
@st.cache_data
def load_gene_stats():
    if os.path.exists('results/top_variable_genes.csv'):
        return pd.read_csv('results/top_variable_genes.csv', index_col=0)
    return None

@st.cache_data
def load_sample_info():
    if os.path.exists('aml_sample_info.csv'):
        return pd.read_csv('aml_sample_info.csv', index_col=0)
    return None

# Main content
if page == "Overview":
    st.header("Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", "537")
    with col2:
        st.metric("Genes Analyzed", "54,675")
    with col3:
        st.metric("Data Size", "329 MB")
    with col4:
        st.metric("Platform", "HG-U133 Plus 2.0")
    
    st.markdown("""
    ### Key Findings
    - Identified highly variable genes in AML samples
    - Discovered molecular heterogeneity through PCA
    - Generated expression profiles for potential biomarkers
    
    ### Clinical Relevance
    This analysis provides insights into:
    - Potential diagnostic biomarkers for AML
    - Molecular subtypes within AML patients
    - Therapeutic target identification
    """)
    
    # Show available plots
    st.subheader("Generated Visualizations")
    plots = glob.glob("plots/*.png")
    if plots:
        selected_plot = st.selectbox("Select a plot to view:", plots)
        st.image(selected_plot)

elif page == "Gene Expression":
    st.header("Gene Expression Analysis")
    
    # Try to load and display gene statistics
    gene_stats = load_gene_stats()
    if gene_stats is not None:
        st.subheader("Top Variable Genes")
        
        # Interactive bar chart
        top_20 = gene_stats.head(20)
        fig = px.bar(
            top_20, 
            y=top_20.index, 
            x='variance' if 'variance' in top_20.columns else top_20.values,
            orientation='h',
            title="Top 20 Most Variable Genes",
            labels={'x': 'Variance', 'y': 'Gene ID'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("Gene Statistics Table")
        st.dataframe(gene_stats.head(50))
        
        # Download button
        csv = gene_stats.to_csv()
        st.download_button(
            label="Download full gene list as CSV",
            data=csv,
            file_name="aml_variable_genes.csv",
            mime="text/csv"
        )

elif page == "PCA Analysis":
    st.header("Principal Component Analysis")
    
    st.markdown("""
    PCA reveals the major sources of variation in the gene expression data,
    helping identify potential AML subtypes and outlier samples.
    """)
    
    # Display PCA plots if they exist
    pca_plots = glob.glob("plots/*pca*.png")
    if pca_plots:
        for plot in pca_plots:
            st.image(plot)
    else:
        st.info("Run PCA analysis to generate plots")

elif page == "Top Genes":
    st.header("Top Expressed Genes in AML")
    
    # Create a searchable interface
    st.subheader("Gene Search")
    gene_search = st.text_input("Search for a gene (enter gene ID):")
    
    if gene_search:
        st.info(f"Searching for {gene_search}...")
        # Here you would search through your data
    
    # Show expression patterns
    st.subheader("Expression Patterns")
    st.markdown("""
    The following genes show the highest expression levels across AML samples:
    - These may serve as diagnostic markers
    - Could be therapeutic targets
    - May indicate AML-specific pathways
    """)

elif page == "About":
    st.header("About This Project")
    
    st.markdown("""
    ### Background
    This project leverages my 6+ years of experience in clinical laboratory operations 
    and hematology to analyze gene expression patterns in Acute Myeloid Leukemia (AML).
    
    ### Technical Stack
    - **Languages:** Python
    - **Libraries:** pandas, numpy, matplotlib, scikit-learn, streamlit
    - **Data Source:** Gene Expression Omnibus (GEO)
    
    ### Author
    **Ugochi Ndubuisi**
    - 🔬 Laboratory Operation Manager at Kaiser Permanente
    - 🎓 Doctorate in Health Sciences
    - 📧 u.l.ndubuisi@gmail.com
    - 💼 [LinkedIn](https://linkedin.com/in/ugochindubuisi)
    
    ### Skills Demonstrated
    - Bioinformatics analysis
    - Large-scale data processing
    - Statistical analysis
    - Data visualization
    - Interactive dashboard development
    """)

# Footer
st.markdown("---")
st.markdown("Created by Ugochi Ndubuisi | AML Bioinformatics Portfolio Project")
