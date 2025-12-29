#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON Files Visualizer for CAIL Dataset

This script visualizes JSON files in the specified directory, providing insights into:
1. Data structure and statistics
2. Label distributions
3. Text length analysis
4. Feature correlations

All output and visualizations are generated in English.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib settings for English labels and proper rendering
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class JSONVisualizer:
    def __init__(self, json_dir, output_dir='visualizations/json_visualizations'):
        """
        Initialize the JSON Visualizer
        
        Args:
            json_dir (str): Directory containing JSON files
            output_dir (str): Directory to save visualizations
        """
        self.json_dir = json_dir
        self.output_dir = output_dir
        self.json_files = []
        self.data_dict = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def discover_json_files(self):
        """
        Discover all JSON files in the specified directory
        """
        print(f"Discovering JSON files in: {self.json_dir}")
        self.json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        
        if not self.json_files:
            print(f"No JSON files found in directory: {self.json_dir}")
            return False
        
        print(f"Found {len(self.json_files)} JSON file(s):")
        for file in self.json_files:
            print(f"  - {file}")
        
        return True
    
    def load_json_file(self, file_path):
        """
        Load a JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            list/dict: Loaded JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            return None
    
    def load_all_json_files(self):
        """
        Load all discovered JSON files
        """
        print("\nLoading JSON files...")
        for file in self.json_files:
            file_path = os.path.join(self.json_dir, file)
            data = self.load_json_file(file_path)
            if data:
                self.data_dict[file] = data
                print(f"Loaded {file} successfully")
        
        if not self.data_dict:
            print("Failed to load any JSON files")
            return False
        
        return True
    
    def analyze_data_structure(self, file_name, data):
        """
        Analyze the structure of loaded JSON data
        
        Args:
            file_name (str): Name of the JSON file
            data (list/dict): Loaded JSON data
        """
        print(f"\nAnalyzing structure of {file_name}:")
        
        if isinstance(data, list):
            print(f"  Data type: List")
            print(f"  Number of entries: {len(data)}")
            
            if data:
                first_entry = data[0]
                print(f"  First entry type: {type(first_entry).__name__}")
                
                if isinstance(first_entry, dict):
                    keys = list(first_entry.keys())
                    print(f"  Keys in entries: {keys}")
                    
                    # Analyze data types for each key
                    print("  Data types for each key:")
                    for key in keys[:10]:  # Show first 10 keys
                        sample_values = [entry[key] for entry in data[:100] if key in entry]
                        if sample_values:
                            data_type = type(sample_values[0]).__name__
                            print(f"    - {key}: {data_type}")
        
        elif isinstance(data, dict):
            print(f"  Data type: Dictionary")
            print(f"  Number of keys: {len(data.keys())}")
            print(f"  Keys: {list(data.keys())}")
        
        else:
            print(f"  Data type: {type(data).__name__}")
    
    def analyze_all_files(self):
        """
        Analyze structure of all loaded JSON files
        """
        for file_name, data in self.data_dict.items():
            self.analyze_data_structure(file_name, data)
    
    def create_dataframe(self, data):
        """
        Convert JSON data to pandas DataFrame
        
        Args:
            data (list): List of dictionaries
            
        Returns:
            DataFrame: pandas DataFrame
        """
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)
        return None
    
    def visualize_label_distribution(self, df, file_name, label_key='label'):
        """
        Visualize distribution of labels
        
        Args:
            df (DataFrame): pandas DataFrame
            file_name (str): Name of the JSON file
            label_key (str): Column name for labels
        """
        if label_key not in df.columns:
            print(f"  Label key '{label_key}' not found in {file_name}")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Count label frequencies
        label_counts = df[label_key].value_counts()
        
        # Create bar plot
        sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
        
        plt.title(f'Label Distribution in {file_name}', fontsize=16)
        plt.xlabel('Label', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Rotate x-axis labels if there are many
        if len(label_counts) > 10:
            plt.xticks(rotation=90)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, f'{os.path.splitext(file_name)[0]}_label_distribution.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"  Created label distribution plot: {save_path}")
    
    def visualize_text_length(self, df, file_name, text_keys=None):
        """
        Visualize text length distributions
        
        Args:
            df (DataFrame): pandas DataFrame
            file_name (str): Name of the JSON file
            text_keys (list): List of column names containing text
        """
        if text_keys is None:
            # Auto-detect text columns
            text_keys = [col for col in df.columns if df[col].dtype == 'object']
        
        for key in text_keys:
            if key in df.columns:
                # Calculate text lengths
                df[f'{key}_length'] = df[key].apply(lambda x: len(str(x)))
                
                plt.figure(figsize=(12, 6))
                
                # Create histogram
                sns.histplot(df[f'{key}_length'], bins=50, kde=True, color='skyblue')
                
                plt.title(f'Text Length Distribution ({key}) in {file_name}', fontsize=16)
                plt.xlabel(f'Length of {key}', fontsize=14)
                plt.ylabel('Frequency', fontsize=14)
                
                plt.tight_layout()
                
                # Save plot
                save_path = os.path.join(self.output_dir, f'{os.path.splitext(file_name)[0]}_{key}_length_distribution.png')
                plt.savefig(save_path, dpi=300)
                plt.close()
                
                print(f"  Created text length distribution plot for {key}: {save_path}")
    
    def visualize_dataset_comparison(self):
        """
        Compare dataset sizes
        """
        file_names = []
        dataset_sizes = []
        
        for file_name, data in self.data_dict.items():
            if isinstance(data, list):
                file_names.append(os.path.splitext(file_name)[0])
                dataset_sizes.append(len(data))
        
        if not file_names:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        sns.barplot(x=file_names, y=dataset_sizes, palette='bright')
        
        plt.title('Dataset Size Comparison', fontsize=16)
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel('Number of Entries', fontsize=14)
        
        # Add value labels on top of bars
        for i, v in enumerate(dataset_sizes):
            plt.text(i, v + max(dataset_sizes)*0.01, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, 'dataset_size_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"\nCreated dataset size comparison plot: {save_path}")
    
    def visualize_numeric_features(self, df, file_name, numeric_keys=None):
        """
        Visualize numeric features
        
        Args:
            df (DataFrame): pandas DataFrame
            file_name (str): Name of the JSON file
            numeric_keys (list): List of column names containing numeric data
        """
        if numeric_keys is None:
            # Auto-detect numeric columns
            numeric_keys = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_keys:
            print(f"  No numeric features found in {file_name}")
            return
        
        print(f"  Found {len(numeric_keys)} numeric feature(s) in {file_name}")
        
        # Create correlation matrix if there are multiple numeric features
        if len(numeric_keys) > 1:
            plt.figure(figsize=(10, 8))
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_keys].corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                       linewidths=0.5, fmt='.2f')
            
            plt.title(f'Correlation Matrix of Numeric Features in {file_name}', fontsize=16)
            
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.output_dir, f'{os.path.splitext(file_name)[0]}_correlation_matrix.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            print(f"  Created correlation matrix plot: {save_path}")
        
        # Create distribution plots for each numeric feature
        for key in numeric_keys:
            plt.figure(figsize=(12, 6))
            
            # Create histogram and boxplot side by side
            plt.subplot(1, 2, 1)
            sns.histplot(df[key], bins=30, kde=True, color='green')
            plt.title(f'Distribution of {key}', fontsize=14)
            plt.xlabel(key, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[key], color='green')
            plt.title(f'Boxplot of {key}', fontsize=14)
            plt.xlabel(key, fontsize=12)
            
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.output_dir, f'{os.path.splitext(file_name)[0]}_{key}_distribution.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            print(f"  Created distribution plot for {key}: {save_path}")
    
    def visualize_all(self):
        """
        Create all visualizations
        """
        print("\nCreating visualizations...")
        
        # Compare dataset sizes
        self.visualize_dataset_comparison()
        
        # Visualize each dataset
        for file_name, data in self.data_dict.items():
            print(f"\nVisualizing {file_name}:")
            
            df = self.create_dataframe(data)
            if df is None:
                print(f"  Cannot convert {file_name} to DataFrame")
                continue
            
            # Basic information about the DataFrame
            print(f"  Number of entries: {len(df)}")
            print(f"  Number of features: {len(df.columns)}")
            
            # Visualize label distribution
            self.visualize_label_distribution(df, file_name)
            
            # Visualize text lengths
            self.visualize_text_length(df, file_name)
            
            # Visualize numeric features
            self.visualize_numeric_features(df, file_name)
    
    def run(self):
        """
        Run the complete visualization pipeline
        """
        print("JSON File Visualizer Started")
        print("=" * 50)
        
        # Step 1: Discover JSON files
        if not self.discover_json_files():
            return False
        
        # Step 2: Load JSON files
        if not self.load_all_json_files():
            return False
        
        # Step 3: Analyze data structure
        self.analyze_all_files()
        
        # Step 4: Create visualizations
        self.visualize_all()
        
        print("\n" + "=" * 50)
        print("JSON File Visualizer Completed Successfully!")
        print(f"Visualizations saved to: {self.output_dir}")
        return True


def main():
    # Define the directory containing JSON files
    json_dir = r"d:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\SCM\datasets\cail"
    
    # Create visualizer instance
    visualizer = JSONVisualizer(json_dir)
    
    # Run the visualization pipeline
    visualizer.run()


if __name__ == "__main__":
    main()
