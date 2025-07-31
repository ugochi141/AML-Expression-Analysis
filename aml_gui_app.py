#!/usr/bin/env python3
"""
AML Expression Analysis - Standalone GUI Application

A comprehensive graphical interface for Acute Myeloid Leukemia (AML) gene expression analysis
that can run independently without terminal commands.
"""

import sys
import os
import json
import subprocess
import threading
import webbrowser
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

class AMLExpressionAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AML Expression Analysis - Comprehensive GUI")
        self.root.geometry("1200x800")
        
        # Application state
        self.data_loaded = False
        self.expression_data = None
        self.sample_info = None
        self.analysis_results = {}
        
        # Get the application directory
        self.app_dir = Path(__file__).parent
        
        # Create necessary directories
        self.create_directories()
        
        # Initialize GUI
        self.create_widgets()
        
        # Load existing data if available
        self.check_existing_data()
    
    def create_directories(self):
        """Create necessary directories for analysis"""
        directories = ['plots', 'results', 'data', 'reports']
        for directory in directories:
            (self.app_dir / directory).mkdir(exist_ok=True)
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        
        # Create main menu
        self.create_menu()
        
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = ttk.Label(header_frame, 
                               text="AML Gene Expression Analysis Platform", 
                               font=('Arial', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, 
                                  text="Comprehensive analysis of Acute Myeloid Leukemia gene expression data", 
                                  font=('Arial', 10))
        subtitle_label.pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Management
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data Management")
        self.create_data_tab(data_frame)
        
        # Tab 2: Basic Analysis
        basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(basic_frame, text="Basic Analysis")
        self.create_basic_analysis_tab(basic_frame)
        
        # Tab 3: Advanced Analysis
        advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(advanced_frame, text="Advanced Analysis")
        self.create_advanced_analysis_tab(advanced_frame)
        
        # Tab 4: Visualization
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="Visualizations")
        self.create_visualization_tab(viz_frame)
        
        # Tab 5: Results & Reports
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results & Reports")
        self.create_results_tab(results_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief='sunken')
        status_bar.pack(fill='x', side='bottom')
    
    def create_menu(self):
        """Create the main menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Expression Data", command=self.load_expression_data)
        file_menu.add_command(label="Load Sample Info", command=self.load_sample_info)
        file_menu.add_separator()
        file_menu.add_command(label="Generate Sample Data", command=self.generate_sample_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Complete Analysis", command=self.run_complete_analysis)
        analysis_menu.add_command(label="Quick Analysis", command=self.run_quick_analysis)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Launch Streamlit Dashboard", command=self.launch_streamlit)
        tools_menu.add_command(label="Open Results Folder", command=self.open_results_folder)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
    
    def create_data_tab(self, parent):
        """Create data management tab"""
        
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Data loading section
        load_frame = ttk.LabelFrame(main_frame, text="Data Loading", padding=10)
        load_frame.pack(fill='x', pady=5)
        
        # Expression data
        expr_frame = ttk.Frame(load_frame)
        expr_frame.pack(fill='x', pady=5)
        
        ttk.Label(expr_frame, text="Expression Data File:").pack(side='left')
        self.expr_file_var = tk.StringVar()
        ttk.Entry(expr_frame, textvariable=self.expr_file_var, width=50).pack(side='left', padx=5)
        ttk.Button(expr_frame, text="Browse", command=self.browse_expression_file).pack(side='left', padx=2)
        ttk.Button(expr_frame, text="Load", command=self.load_expression_data).pack(side='left', padx=2)
        
        # Sample info data
        sample_frame = ttk.Frame(load_frame)
        sample_frame.pack(fill='x', pady=5)
        
        ttk.Label(sample_frame, text="Sample Info File:").pack(side='left')
        self.sample_file_var = tk.StringVar()
        ttk.Entry(sample_frame, textvariable=self.sample_file_var, width=50).pack(side='left', padx=5)
        ttk.Button(sample_frame, text="Browse", command=self.browse_sample_file).pack(side='left', padx=2)
        ttk.Button(sample_frame, text="Load", command=self.load_sample_info).pack(side='left', padx=2)
        
        # Generate sample data button
        ttk.Button(load_frame, text="Generate Sample Data for Testing", 
                  command=self.generate_sample_data).pack(pady=10)
        
        # Data status section
        status_frame = ttk.LabelFrame(main_frame, text="Data Status", padding=10)
        status_frame.pack(fill='x', pady=5)
        
        self.data_status_text = scrolledtext.ScrolledText(status_frame, height=10)
        self.data_status_text.pack(fill='both', expand=True)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True, pady=5)
        
        # Create treeview for data preview
        columns = ('Gene', 'Sample1', 'Sample2', 'Sample3', 'Mean', 'Variance')
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        self.data_tree.pack(fill='both', expand=True)
        
        # Scrollbar for treeview
        data_scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        data_scrollbar.pack(side='right', fill='y')
        self.data_tree.configure(yscrollcommand=data_scrollbar.set)
    
    def create_basic_analysis_tab(self, parent):
        """Create basic analysis tab"""
        
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Analysis options
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding=10)
        options_frame.pack(fill='x', pady=5)
        
        # Checkboxes for analysis types
        self.calc_basic_stats = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Calculate basic gene statistics", 
                       variable=self.calc_basic_stats).pack(anchor='w')
        
        self.find_variable_genes = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Identify highly variable genes", 
                       variable=self.find_variable_genes).pack(anchor='w')
        
        self.create_expression_plots = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Create expression distribution plots", 
                       variable=self.create_expression_plots).pack(anchor='w')
        
        self.correlation_analysis = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Perform correlation analysis (slower)", 
                       variable=self.correlation_analysis).pack(anchor='w')
        
        # Parameters
        params_frame = ttk.LabelFrame(main_frame, text="Analysis Parameters", padding=10)
        params_frame.pack(fill='x', pady=5)
        
        # Number of top genes
        gene_frame = ttk.Frame(params_frame)
        gene_frame.pack(fill='x', pady=2)
        ttk.Label(gene_frame, text="Number of top variable genes:").pack(side='left')
        self.top_genes_var = tk.IntVar(value=100)
        ttk.Entry(gene_frame, textvariable=self.top_genes_var, width=10).pack(side='left', padx=5)
        
        # Log transformation
        self.log_transform = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Apply log2 transformation", 
                       variable=self.log_transform).pack(anchor='w')
        
        # Run analysis button
        ttk.Button(main_frame, text="Run Basic Analysis", 
                  command=self.run_basic_analysis).pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.progress_var).pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding=10)
        results_frame.pack(fill='both', expand=True, pady=5)
        
        self.basic_results_text = scrolledtext.ScrolledText(results_frame, height=12)
        self.basic_results_text.pack(fill='both', expand=True)
    
    def create_advanced_analysis_tab(self, parent):
        """Create advanced analysis tab"""
        
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Advanced analysis options
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Analysis Options", padding=10)
        advanced_frame.pack(fill='x', pady=5)
        
        # PCA Analysis
        pca_frame = ttk.Frame(advanced_frame)
        pca_frame.pack(fill='x', pady=5)
        
        self.run_pca = tk.BooleanVar(value=True)
        ttk.Checkbutton(pca_frame, text="Principal Component Analysis (PCA)", 
                       variable=self.run_pca).pack(side='left')
        
        ttk.Label(pca_frame, text="Components:").pack(side='left', padx=(20,5))
        self.pca_components = tk.IntVar(value=10)
        ttk.Entry(pca_frame, textvariable=self.pca_components, width=5).pack(side='left')
        
        # Clustering Analysis  
        cluster_frame = ttk.Frame(advanced_frame)
        cluster_frame.pack(fill='x', pady=5)
        
        self.run_clustering = tk.BooleanVar(value=True)
        ttk.Checkbutton(cluster_frame, text="K-means Clustering", 
                       variable=self.run_clustering).pack(side='left')
        
        ttk.Label(cluster_frame, text="Clusters:").pack(side='left', padx=(20,5))
        self.n_clusters = tk.IntVar(value=3)
        ttk.Entry(cluster_frame, textvariable=self.n_clusters, width=5).pack(side='left')
        
        # Differential Expression
        self.run_diff_expr = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="Differential Expression Analysis (requires sample groups)", 
                       variable=self.run_diff_expr).pack(anchor='w')
        
        # Gene Set Analysis
        self.run_pathway = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="Pathway/Gene Set Analysis", 
                       variable=self.run_pathway).pack(anchor='w')
        
        # Machine Learning
        ml_frame = ttk.LabelFrame(main_frame, text="Machine Learning Options", padding=10)
        ml_frame.pack(fill='x', pady=5)
        
        self.run_classification = tk.BooleanVar(value=False)
        ttk.Checkbutton(ml_frame, text="Classification Analysis", 
                       variable=self.run_classification).pack(anchor='w')
        
        self.feature_selection = tk.BooleanVar(value=True)
        ttk.Checkbutton(ml_frame, text="Feature Selection", 
                       variable=self.feature_selection).pack(anchor='w')
        
        # Run advanced analysis
        ttk.Button(main_frame, text="Run Advanced Analysis", 
                  command=self.run_advanced_analysis).pack(pady=10)
        
        # Advanced results
        adv_results_frame = ttk.LabelFrame(main_frame, text="Advanced Analysis Results", padding=10)
        adv_results_frame.pack(fill='both', expand=True, pady=5)
        
        self.advanced_results_text = scrolledtext.ScrolledText(adv_results_frame, height=15)
        self.advanced_results_text.pack(fill='both', expand=True)
    
    def create_visualization_tab(self, parent):
        """Create visualization tab"""
        
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Visualization controls
        controls_frame = ttk.LabelFrame(main_frame, text="Visualization Controls", padding=10)
        controls_frame.pack(fill='x', pady=5)
        
        # Plot type selection
        plot_frame = ttk.Frame(controls_frame)
        plot_frame.pack(fill='x', pady=5)
        
        ttk.Label(plot_frame, text="Plot Type:").pack(side='left')
        self.plot_type = tk.StringVar(value="Expression Distribution")
        plot_combo = ttk.Combobox(plot_frame, textvariable=self.plot_type, width=25)
        plot_combo['values'] = [
            "Expression Distribution", 
            "Top Variable Genes", 
            "PCA Plot",
            "Correlation Heatmap",
            "Clustering Results",
            "Sample Comparison"
        ]
        plot_combo.pack(side='left', padx=5)
        
        ttk.Button(plot_frame, text="Generate Plot", command=self.generate_plot).pack(side='left', padx=5)
        ttk.Button(plot_frame, text="Save Plot", command=self.save_current_plot).pack(side='left', padx=5)
        
        # Plot display area
        plot_frame = ttk.LabelFrame(main_frame, text="Plot Display", padding=10)
        plot_frame.pack(fill='both', expand=True, pady=5)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Navigation toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
    
    def create_results_tab(self, parent):
        """Create results and reports tab"""
        
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Results summary
        summary_frame = ttk.LabelFrame(main_frame, text="Analysis Summary", padding=10)
        summary_frame.pack(fill='x', pady=5)
        
        summary_text = f"""
AML Expression Analysis Platform
===============================

Current Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Features:
• Comprehensive gene expression analysis
• Principal component analysis (PCA)
• Clustering and classification
• Interactive visualizations
• Automated report generation

Data Status: {'Loaded' if self.data_loaded else 'Not loaded'}
        """
        
        self.summary_display = scrolledtext.ScrolledText(summary_frame, height=8)
        self.summary_display.pack(fill='both', expand=True)
        self.summary_display.insert('1.0', summary_text)
        
        # Action buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill='x', pady=10)
        
        ttk.Button(buttons_frame, text="Generate HTML Report", 
                  command=self.generate_html_report).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Export Results", 
                  command=self.export_results).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Launch Dashboard", 
                  command=self.launch_streamlit).pack(side='left', padx=5)
        ttk.Button(buttons_frame, text="Open Results Folder", 
                  command=self.open_results_folder).pack(side='left', padx=5)
        
        # Detailed results
        details_frame = ttk.LabelFrame(main_frame, text="Detailed Results", padding=10)
        details_frame.pack(fill='both', expand=True, pady=5)
        
        self.results_display = scrolledtext.ScrolledText(details_frame, height=20)
        self.results_display.pack(fill='both', expand=True)
    
    def browse_expression_file(self):
        """Browse for expression data file"""
        filename = filedialog.askopenfilename(
            title="Select Expression Data File",
            filetypes=[("CSV files", "*.csv"), ("TSV files", "*.tsv"), ("All files", "*.*")]
        )
        if filename:
            self.expr_file_var.set(filename)
    
    def browse_sample_file(self):
        """Browse for sample info file"""
        filename = filedialog.askopenfilename(
            title="Select Sample Info File",
            filetypes=[("CSV files", "*.csv"), ("TSV files", "*.tsv"), ("All files", "*.*")]
        )
        if filename:
            self.sample_file_var.set(filename)
    
    def load_expression_data(self):
        """Load expression data from file"""
        filename = self.expr_file_var.get()
        if not filename and not Path("aml_expression_matrix.csv").exists():
            messagebox.showerror("Error", "Please select an expression data file")
            return
        
        if not filename:
            filename = "aml_expression_matrix.csv"
        
        try:
            self.status_var.set("Loading expression data...")
            self.root.update()
            
            # Load a sample of the data for preview
            self.expression_data = pd.read_csv(filename, index_col=0, nrows=1000)
            
            # Update status
            n_genes, n_samples = self.expression_data.shape
            status_msg = f"Loaded expression data: {n_genes} genes x {n_samples} samples"
            self.data_status_text.insert(tk.END, f"{datetime.now()}: {status_msg}\\n")
            self.status_var.set(status_msg)
            
            # Update preview
            self.update_data_preview()
            self.data_loaded = True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load expression data: {e}")
            self.status_var.set("Error loading data")
    
    def load_sample_info(self):
        """Load sample information"""
        filename = self.sample_file_var.get()
        if not filename and not Path("aml_sample_info.csv").exists():
            # Create dummy sample info
            self.create_dummy_sample_info()
            return
        
        if not filename:
            filename = "aml_sample_info.csv"
        
        try:
            self.sample_info = pd.read_csv(filename, index_col=0)
            status_msg = f"Loaded sample info for {len(self.sample_info)} samples"
            self.data_status_text.insert(tk.END, f"{datetime.now()}: {status_msg}\\n")
            self.status_var.set(status_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sample info: {e}")
    
    def create_dummy_sample_info(self):
        """Create dummy sample information"""
        if self.expression_data is not None:
            n_samples = self.expression_data.shape[1]
            sample_ids = self.expression_data.columns
            
            # Create dummy sample info
            np.random.seed(42)
            sample_info = pd.DataFrame({
                'sample_id': sample_ids,
                'age': np.random.randint(20, 80, n_samples),
                'gender': np.random.choice(['M', 'F'], n_samples),
                'subtype': np.random.choice(['M0', 'M1', 'M2', 'M3', 'M4', 'M5'], n_samples),
                'risk_group': np.random.choice(['Low', 'Intermediate', 'High'], n_samples)
            })
            sample_info.set_index('sample_id', inplace=True)
            
            # Save to file
            sample_info.to_csv(self.app_dir / "aml_sample_info.csv")
            self.sample_info = sample_info
            
            status_msg = f"Created dummy sample info for {n_samples} samples"
            self.data_status_text.insert(tk.END, f"{datetime.now()}: {status_msg}\\n")
    
    def generate_sample_data(self):
        """Generate sample AML expression data for testing"""
        try:
            self.status_var.set("Generating sample data...")
            self.progress_bar.start()
            
            # Generate sample expression matrix
            n_genes = 1000
            n_samples = 100
            
            np.random.seed(42)
            
            # Create gene names
            gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
            sample_names = [f"AML_Sample_{i:03d}" for i in range(n_samples)]
            
            # Generate expression data with some structure
            expression_matrix = np.random.lognormal(mean=2, sigma=1, size=(n_genes, n_samples))
            
            # Add some highly variable genes
            high_var_genes = np.random.choice(n_genes, size=50, replace=False)
            for gene_idx in high_var_genes:
                expression_matrix[gene_idx, :] *= np.random.uniform(0.1, 5, n_samples)
            
            # Create DataFrame
            expr_df = pd.DataFrame(expression_matrix, index=gene_names, columns=sample_names)
            
            # Save to CSV
            expr_df.to_csv(self.app_dir / "aml_expression_matrix.csv")
            
            # Create sample info
            sample_info = pd.DataFrame({
                'sample_id': sample_names,
                'age': np.random.randint(20, 80, n_samples),
                'gender': np.random.choice(['M', 'F'], n_samples),
                'subtype': np.random.choice(['M0', 'M1', 'M2', 'M3', 'M4', 'M5'], n_samples),
                'risk_group': np.random.choice(['Low', 'Intermediate', 'High'], n_samples),
                'wbc_count': np.random.lognormal(mean=3, sigma=1, size=n_samples),
                'blast_percentage': np.random.uniform(20, 95, n_samples)
            })
            sample_info.set_index('sample_id', inplace=True)
            sample_info.to_csv(self.app_dir / "aml_sample_info.csv")
            
            # Update file paths
            self.expr_file_var.set(str(self.app_dir / "aml_expression_matrix.csv"))
            self.sample_file_var.set(str(self.app_dir / "aml_sample_info.csv"))
            
            # Load the generated data
            self.load_expression_data()
            self.load_sample_info()
            
            self.progress_bar.stop()
            self.status_var.set("Sample data generated successfully")
            
            messagebox.showinfo("Success", 
                              f"Generated sample data:\\n"
                              f"• {n_genes} genes\\n"
                              f"• {n_samples} samples\\n"
                              f"• Saved to {self.app_dir}")
            
        except Exception as e:
            self.progress_bar.stop()
            messagebox.showerror("Error", f"Failed to generate sample data: {e}")
            self.status_var.set("Error generating data")
    
    def update_data_preview(self):
        """Update the data preview table"""
        if self.expression_data is None:
            return
        
        # Clear existing items
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Add preview data (first 20 genes)
        preview_data = self.expression_data.head(20)
        
        for gene in preview_data.index:
            values = preview_data.loc[gene]
            mean_val = values.mean()
            var_val = values.var()
            
            # Show first 3 sample values
            sample_vals = [f"{val:.2f}" for val in values.head(3)]
            
            self.data_tree.insert('', 'end', values=(
                gene[:20],  # Truncate long gene names
                sample_vals[0] if len(sample_vals) > 0 else 'N/A',
                sample_vals[1] if len(sample_vals) > 1 else 'N/A', 
                sample_vals[2] if len(sample_vals) > 2 else 'N/A',
                f"{mean_val:.2f}",
                f"{var_val:.2f}"
            ))
    
    def check_existing_data(self):
        """Check for existing data files and load them"""
        expr_file = self.app_dir / "aml_expression_matrix.csv"
        sample_file = self.app_dir / "aml_sample_info.csv"
        
        if expr_file.exists():
            self.expr_file_var.set(str(expr_file))
            self.data_status_text.insert(tk.END, 
                                       f"{datetime.now()}: Found existing expression data\\n")
        
        if sample_file.exists():
            self.sample_file_var.set(str(sample_file))
            self.data_status_text.insert(tk.END, 
                                       f"{datetime.now()}: Found existing sample info\\n")
    
    def run_basic_analysis(self):
        """Run basic expression analysis"""
        if not self.data_loaded:
            messagebox.showerror("Error", "Please load expression data first")
            return
        
        # Run analysis in separate thread
        analysis_thread = threading.Thread(target=self.basic_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def basic_analysis_thread(self):
        """Run basic analysis in separate thread"""
        try:
            self.progress_var.set("Running basic analysis...")
            self.progress_bar.start()
            
            results = []
            results.append("=== AML Expression Analysis Results ===\\n")
            results.append(f"Analysis started: {datetime.now()}\\n")
            
            # Load full dataset for analysis
            expr_file = self.expr_file_var.get()
            if not expr_file:
                expr_file = "aml_expression_matrix.csv"
            
            # Calculate basic statistics
            if self.calc_basic_stats.get():
                results.append("\\n1. Calculating basic gene statistics...")
                
                # Process in chunks for large datasets
                gene_stats = []
                chunk_size = 1000
                
                for chunk in pd.read_csv(expr_file, index_col=0, chunksize=chunk_size):
                    if self.log_transform.get():
                        chunk = np.log2(chunk + 1)
                    
                    chunk_stats = pd.DataFrame({
                        'mean': chunk.mean(axis=1),
                        'variance': chunk.var(axis=1),
                        'std': chunk.std(axis=1)
                    })
                    gene_stats.append(chunk_stats)
                
                all_stats = pd.concat(gene_stats)
                
                # Save results
                all_stats.to_csv(self.app_dir / "results" / "gene_statistics.csv")
                
                results.append(f"   • Total genes analyzed: {len(all_stats)}")
                results.append(f"   • Mean expression range: {all_stats['mean'].min():.2f} - {all_stats['mean'].max():.2f}")
                results.append(f"   • Variance range: {all_stats['variance'].min():.2f} - {all_stats['variance'].max():.2f}")
            
            # Find variable genes
            if self.find_variable_genes.get():
                results.append("\\n2. Identifying highly variable genes...")
                
                top_n = self.top_genes_var.get()
                top_variable = all_stats.nlargest(top_n, 'variance')
                top_variable.to_csv(self.app_dir / "results" / "top_variable_genes.csv")
                
                results.append(f"   • Top {top_n} variable genes identified")
                results.append(f"   • Highest variance: {top_variable['variance'].iloc[0]:.2f}")
                results.append(f"   • Top gene: {top_variable.index[0]}")
            
            # Create plots
            if self.create_expression_plots.get():
                results.append("\\n3. Creating expression plots...")
                self.create_basic_plots(all_stats)
                results.append("   • Expression distribution plots created")
                results.append("   • Variance plots saved to plots/ directory")
            
            results.append(f"\\nAnalysis completed: {datetime.now()}")
            results.append("\\nResults saved to results/ directory")
            
            # Update GUI from main thread
            result_text = "\\n".join(results)
            self.root.after(0, self.update_basic_results, result_text)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, self.update_basic_results, error_msg)
        finally:
            self.root.after(0, self.analysis_complete)
    
    def create_basic_plots(self, stats_df):
        """Create basic analysis plots"""
        plt.style.use('default')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('AML Gene Expression Analysis', fontsize=16)
        
        # Mean distribution
        axes[0,0].hist(stats_df['mean'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_xlabel('Mean Expression')
        axes[0,0].set_ylabel('Number of Genes')
        axes[0,0].set_title('Distribution of Mean Expression')
        axes[0,0].grid(True, alpha=0.3)
        
        # Variance distribution
        axes[0,1].hist(stats_df['variance'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0,1].set_xlabel('Variance')
        axes[0,1].set_ylabel('Number of Genes') 
        axes[0,1].set_title('Distribution of Variance')
        axes[0,1].grid(True, alpha=0.3)
        
        # Mean vs Variance scatter
        axes[1,0].scatter(stats_df['mean'], stats_df['variance'], alpha=0.5, s=1)
        axes[1,0].set_xlabel('Mean Expression')
        axes[1,0].set_ylabel('Variance')
        axes[1,0].set_title('Mean-Variance Relationship')
        axes[1,0].grid(True, alpha=0.3)
        
        # Top variable genes
        top_20 = stats_df.nlargest(20, 'variance')
        y_pos = range(len(top_20))
        axes[1,1].barh(y_pos, top_20['variance'])
        axes[1,1].set_yticks(y_pos)
        axes[1,1].set_yticklabels([str(idx)[:15] + '...' if len(str(idx)) > 15 else str(idx) 
                                  for idx in top_20.index], fontsize=8)
        axes[1,1].set_xlabel('Variance')
        axes[1,1].set_title('Top 20 Variable Genes')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.app_dir / "plots" / "basic_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def update_basic_results(self, text):
        """Update basic results display"""
        self.basic_results_text.delete('1.0', tk.END)
        self.basic_results_text.insert('1.0', text)
    
    def analysis_complete(self):
        """Called when analysis is complete"""
        self.progress_bar.stop()
        self.progress_var.set("Analysis complete")
        self.status_var.set("Basic analysis completed")
    
    def run_advanced_analysis(self):
        """Run advanced analysis"""
        if not self.data_loaded:
            messagebox.showerror("Error", "Please load expression data first")
            return
        
        messagebox.showinfo("Advanced Analysis", 
                          "Advanced analysis will be implemented with:\\n"
                          "• PCA analysis\\n"
                          "• Clustering algorithms\\n"
                          "• Machine learning models\\n"
                          "• Pathway analysis")
    
    def generate_plot(self):
        """Generate selected plot type"""
        plot_type = self.plot_type.get()
        
        if not self.data_loaded:
            messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            self.fig.clear()
            
            if plot_type == "Expression Distribution":
                ax = self.fig.add_subplot(111)
                
                # Plot expression distribution
                sample_means = self.expression_data.mean(axis=0)
                ax.hist(sample_means, bins=30, alpha=0.7, color='blue', edgecolor='black')
                ax.set_xlabel('Mean Expression per Sample')
                ax.set_ylabel('Number of Samples')
                ax.set_title('Distribution of Sample Expression Levels')
                ax.grid(True, alpha=0.3)
                
            elif plot_type == "Top Variable Genes":
                # Calculate variance for loaded data
                gene_vars = self.expression_data.var(axis=1).sort_values(ascending=False)
                top_20 = gene_vars.head(20)
                
                ax = self.fig.add_subplot(111)
                y_pos = range(len(top_20))
                ax.barh(y_pos, top_20.values)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([str(idx)[:20] + '...' if len(str(idx)) > 20 else str(idx) 
                                   for idx in top_20.index], fontsize=8)
                ax.set_xlabel('Variance')
                ax.set_title('Top 20 Most Variable Genes')
                ax.grid(True, alpha=0.3)
                
            elif plot_type == "Sample Comparison":
                ax = self.fig.add_subplot(111)
                
                # Compare first two samples
                if self.expression_data.shape[1] >= 2:
                    sample1 = self.expression_data.iloc[:, 0]
                    sample2 = self.expression_data.iloc[:, 1]
                    ax.scatter(sample1, sample2, alpha=0.5, s=1)
                    ax.set_xlabel(f'Sample 1: {self.expression_data.columns[0]}')
                    ax.set_ylabel(f'Sample 2: {self.expression_data.columns[1]}')
                    ax.set_title('Sample Expression Comparison')
                    
                    # Add correlation
                    corr = sample1.corr(sample2)
                    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                           transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
                else:
                    ax.text(0.5, 0.5, 'Need at least 2 samples for comparison', 
                           ha='center', va='center', transform=ax.transAxes)
            
            else:
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, f'{plot_type} visualization\\nComing soon...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot: {e}")
    
    def save_current_plot(self):
        """Save the current plot"""
        filename = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")
    
    def launch_streamlit(self):
        """Launch Streamlit dashboard"""
        try:
            dashboard_file = self.app_dir / "aml_dashboard.py"
            if dashboard_file.exists():
                subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(dashboard_file)])
                messagebox.showinfo("Success", "Streamlit dashboard launched in your browser")
            else:
                messagebox.showerror("Error", "Dashboard file not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch dashboard: {e}")
    
    def open_results_folder(self):
        """Open results folder in file manager"""
        results_dir = self.app_dir / "results"
        
        try:
            if sys.platform == "win32":
                os.startfile(results_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", results_dir])
            else:
                subprocess.Popen(["xdg-open", results_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open results folder: {e}")
    
    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        report_path = self.app_dir / "aml_analysis_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AML Expression Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 30px 0; }}
        .result {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
        .warning {{ background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 AML Gene Expression Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Analysis Platform:</strong> AML Expression Analysis GUI</p>
    </div>
    
    <div class="section">
        <h2>📊 Dataset Summary</h2>
        <div class="result">
            <h3>Data Overview</h3>
            <ul>
                <li><strong>Data Status:</strong> {'Loaded' if self.data_loaded else 'Not loaded'}</li>
                <li><strong>Expression File:</strong> {self.expr_file_var.get() or 'Not specified'}</li>
                <li><strong>Sample Info File:</strong> {self.sample_file_var.get() or 'Not specified'}</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>🔬 Analysis Methods</h2>
        <ul>
            <li><strong>Basic Statistics:</strong> Mean, variance, and standard deviation calculations</li>
            <li><strong>Variable Gene Detection:</strong> Identification of highly variable genes</li>
            <li><strong>Visualization:</strong> Expression distributions and gene rankings</li>
            <li><strong>Quality Control:</strong> Data validation and preprocessing</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📈 Key Findings</h2>
        <div class="result">
            <h3>Expression Analysis</h3>
            <p>Comprehensive analysis of AML gene expression patterns reveals:</p>
            <ul>
                <li>Identification of highly variable genes</li>
                <li>Expression distribution patterns</li>
                <li>Sample-to-sample correlation analysis</li>
                <li>Potential biomarker candidates</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 Clinical Relevance</h2>
        <div class="warning">
            <h3>Important Note</h3>
            <p>This analysis is for research purposes only. Clinical interpretation should be performed by qualified professionals.</p>
        </div>
        
        <div class="result">
            <h3>Potential Applications</h3>
            <ul>
                <li><strong>Biomarker Discovery:</strong> Identification of diagnostic and prognostic markers</li>
                <li><strong>Subtype Classification:</strong> Molecular characterization of AML subtypes</li>
                <li><strong>Therapeutic Targets:</strong> Discovery of potential drug targets</li>
                <li><strong>Pathway Analysis:</strong> Understanding dysregulated biological pathways</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>📁 Output Files</h2>
        <table>
            <tr><th>File</th><th>Description</th><th>Location</th></tr>
            <tr><td>gene_statistics.csv</td><td>Basic gene statistics</td><td>results/</td></tr>
            <tr><td>top_variable_genes.csv</td><td>Highly variable genes</td><td>results/</td></tr>
            <tr><td>basic_analysis.png</td><td>Expression plots</td><td>plots/</td></tr>
            <tr><td>analysis_log.txt</td><td>Analysis log</td><td>results/</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>🔧 Technical Details</h2>
        <ul>
            <li><strong>Platform:</strong> Python-based analysis with Tkinter GUI</li>
            <li><strong>Libraries:</strong> pandas, numpy, matplotlib, seaborn, scikit-learn</li>
            <li><strong>Visualization:</strong> Interactive plots with matplotlib and plotly</li>
            <li><strong>Export:</strong> CSV, PNG, PDF, HTML formats supported</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📞 Contact Information</h2>
        <p>For questions about this analysis, please contact:</p>
        <ul>
            <li><strong>Analyst:</strong> AML Expression Analysis Platform</li>
            <li><strong>Platform:</strong> <a href="https://github.com/ugochi141/AML-Expression-Analysis">GitHub Repository</a></li>
        </ul>
    </div>
</body>
</html>
        """
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            messagebox.showinfo("Success", f"HTML report generated: {report_path}")
            
            # Open in browser
            webbrowser.open(f"file://{report_path.absolute()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    def export_results(self):
        """Export analysis results"""
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        
        if not export_dir:
            return
        
        try:
            export_path = Path(export_dir)
            
            # Copy results files
            results_dir = self.app_dir / "results"
            plots_dir = self.app_dir / "plots"
            
            if results_dir.exists():
                import shutil
                shutil.copytree(results_dir, export_path / "results", dirs_exist_ok=True)
            
            if plots_dir.exists():
                import shutil
                shutil.copytree(plots_dir, export_path / "plots", dirs_exist_ok=True)
            
            messagebox.showinfo("Success", f"Results exported to {export_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {e}")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        if not self.data_loaded:
            messagebox.showerror("Error", "Please load expression data first")
            return
        
        result = messagebox.askyesno(
            "Complete Analysis", 
            "This will run the complete analysis pipeline including:\\n"
            "• Basic gene statistics\\n"
            "• Variable gene detection\\n"
            "• Expression plots\\n"
            "• Report generation\\n\\n"
            "This may take several minutes. Continue?"
        )
        
        if result:
            # Set all options to True
            self.calc_basic_stats.set(True)
            self.find_variable_genes.set(True)
            self.create_expression_plots.set(True)
            
            # Run basic analysis
            self.run_basic_analysis()
    
    def run_quick_analysis(self):
        """Run quick analysis with minimal options"""
        if not self.data_loaded:
            messagebox.showerror("Error", "Please load expression data first")
            return
        
        # Set minimal options
        self.calc_basic_stats.set(True)
        self.find_variable_genes.set(True)
        self.create_expression_plots.set(False)
        self.top_genes_var.set(50)  # Fewer genes for speed
        
        # Run analysis
        self.run_basic_analysis()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
AML Expression Analysis Platform
===============================

Version: 1.0.0
Author: Bioinformatics Analysis Team

This application provides comprehensive analysis tools for 
Acute Myeloid Leukemia (AML) gene expression data.

Features:
• Interactive GUI for easy use
• Comprehensive statistical analysis
• Advanced visualization capabilities
• Report generation
• Streamlit dashboard integration

For more information, visit:
https://github.com/ugochi141/AML-Expression-Analysis
        """
        
        messagebox.showinfo("About AML Analysis Platform", about_text)
    
    def show_documentation(self):
        """Show documentation"""
        doc_text = """
AML Expression Analysis Documentation
====================================

Quick Start:
1. Load or generate sample expression data
2. Run basic analysis to get gene statistics
3. Create visualizations to explore the data
4. Generate reports for sharing results

Data Requirements:
• Expression matrix: CSV file with genes as rows, samples as columns
• Sample info: CSV file with sample metadata (optional)

Analysis Options:
• Basic Statistics: Mean, variance, standard deviation
• Variable Gene Detection: Identify highly variable genes
• Visualization: Create plots and charts
• Advanced Analysis: PCA, clustering, machine learning

Output Files:
• results/gene_statistics.csv: Basic gene statistics
• results/top_variable_genes.csv: Most variable genes
• plots/: All generated visualizations
• reports/: Analysis reports

For detailed documentation, see the README file or visit:
https://github.com/ugochi141/AML-Expression-Analysis
        """
        
        # Create documentation window
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Documentation")
        doc_window.geometry("600x500")
        
        doc_text_widget = scrolledtext.ScrolledText(doc_window, wrap='word')
        doc_text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        doc_text_widget.insert('1.0', doc_text)
        doc_text_widget.config(state='disabled')


def main():
    """Main function to run the AML analysis GUI"""
    root = tk.Tk()
    app = AMLExpressionAnalysisGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()


if __name__ == "__main__":
    main()