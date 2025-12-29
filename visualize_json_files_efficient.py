#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Efficient JSON Files Visualizer for CAIL Dataset

This script efficiently visualizes large JSON files, providing insights into:
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
import jsonlines
warnings.filterwarnings('ignore')

# Set matplotlib settings for English labels and proper rendering
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class EfficientJSONVisualizer:
    def __init__(self, json_dir, output_dir='visualizations/json_visualizations', sample_size=10000):
        """
        Initialize the Efficient JSON Visualizer
        
        Args:
            json_dir (str): Directory containing JSON files
            output_dir (str): Directory to save visualizations
            sample_size (int): Maximum number of samples to use for visualization
        """
        self.json_dir = json_dir
        self.output_dir = output_dir
        self.json_files = []
        self.sample_size = sample_size
        
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
            file_path = os.path.join(self.json_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.1f} MB)")
        
        return True
    
    def sample_json_file(self, file_path):
        """
        Sample data from a large JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            list: Sampled JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                total_entries = len(data)
                if total_entries <= self.sample_size:
                    return data, total_entries
                else:
                    # Randomly sample entries
                    sample_indices = np.random.choice(total_entries, self.sample_size, replace=False)
                    sampled_data = [data[i] for i in sample_indices]
                    return sampled_data, total_entries
            else:
                return [data], 1
                
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            return None, 0
    
    def analyze_file_structure(self, file_path, sample_data, total_entries):
        """
        Analyze the structure of a JSON file
        
        Args:
            file_path (str): Path to JSON file
            sample_data (list): Sampled JSON data
            total_entries (int): Total number of entries in the file
        """
        file_name = os.path.basename(file_path)
        print(f"\nAnalyzing {file_name}:")
        print(f"  Total entries in file: {total_entries:,}")
        print(f"  Using {len(sample_data):,} entries for analysis")
        
        if not sample_data:
            print(f"  No data to analyze in {file_name}")
            return None
        
        first_entry = sample_data[0]
        print(f"  Entry type: {type(first_entry).__name__}")
        
        if isinstance(first_entry, dict):
            keys = list(first_entry.keys())
            print(f"  Keys in entries: {keys}")
            
            # Analyze data types for each key
            print("  Data types for each key:")
            for key in keys[:10]:  # Show first 10 keys
                sample_values = [entry[key] for entry in sample_data[:100] if key in entry]
                if sample_values:
                    data_type = type(sample_values[0]).__name__
                    print(f"    - {key}: {data_type}")
                    
            return keys
        
        return None
    
    def create_dataframe(self, sample_data):
        """
        Convert sampled JSON data to pandas DataFrame
        
        Args:
            sample_data (list): List of dictionaries
            
        Returns:
            DataFrame: pandas DataFrame
        """
        if isinstance(sample_data, list) and all(isinstance(item, dict) for item in sample_data):
            return pd.DataFrame(sample_data)
        return None
    
    def visualize_label_distribution(self, df, file_name, label_key='label', total_entries=0):
        """
        Visualize distribution of labels
        
        Args:
            df (DataFrame): pandas DataFrame
            file_name (str): Name of the JSON file
            label_key (str): Column name for labels
            total_entries (int): Total number of entries in the file
        """
        if label_key not in df.columns:
            # Try common label key variations
            common_label_keys = ['labels', 'target', 'y', 'category', 'law', 'charge']
            for key in common_label_keys:
                if key in df.columns:
                    label_key = key
                    break
            else:
                print(f"  No label column found in {file_name}")
                return
        
        print(f"  Visualizing label distribution ({label_key})...")
        plt.figure(figsize=(12, 6))
        
        # Count label frequencies
        label_counts = df[label_key].value_counts()
        
        # Create bar plot
        ax = sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, palette='viridis')
        
        title = f'Label Distribution in {os.path.splitext(file_name)[0]}'
        if total_entries > len(df):
            title += f' (Sample of {len(df):,} from {total_entries:,} total entries)'
        plt.title(title, fontsize=16)
        
        plt.xlabel('Label', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        
        # Rotate x-axis labels if there are many
        if len(label_counts) > 10:
            plt.xticks(rotation=90)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, f'{os.path.splitext(file_name)[0]}_{label_key}_distribution.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"  Saved label distribution plot: {save_path}")
    
    def visualize_text_length(self, df, file_name, text_keys=None, total_entries=0):
        """
        Visualize text length distributions
        
        Args:
            df (DataFrame): pandas DataFrame
            file_name (str): Name of the JSON file
            text_keys (list): List of column names containing text
            total_entries (int): Total number of entries in the file
        """
        if text_keys is None:
            # Auto-detect text columns
            text_keys = [col for col in df.columns if df[col].dtype == 'object']
        
        for key in text_keys[:5]:  # Limit to first 5 text columns
            if key in df.columns:
                print(f"  Visualizing text length distribution ({key})...")
                # Calculate text lengths
                df[f'{key}_length'] = df[key].apply(lambda x: len(str(x)))
                
                plt.figure(figsize=(12, 6))
                
                # Create histogram
                sns.histplot(df[f'{key}_length'], bins=50, kde=True, color='skyblue')
                
                title = f'Text Length Distribution ({key}) in {os.path.splitext(file_name)[0]}'
                if total_entries > len(df):
                    title += f' (Sample of {len(df):,} from {total_entries:,} total entries)'
                plt.title(title, fontsize=16)
                
                plt.xlabel(f'Length of {key}', fontsize=14)
                plt.ylabel('Frequency', fontsize=14)
                
                plt.tight_layout()
                
                # Save plot
                save_path = os.path.join(self.output_dir, f'{os.path.splitext(file_name)[0]}_{key}_length_distribution.png')
                plt.savefig(save_path, dpi=300)
                plt.close()
                
                print(f"  Saved text length distribution plot: {save_path}")
    
    def visualize_numeric_features(self, df, file_name, total_entries=0):
        """
        Visualize numeric features
        
        Args:
            df (DataFrame): pandas DataFrame
            file_name (str): Name of the JSON file
            total_entries (int): Total number of entries in the file
        """
        # Auto-detect numeric columns
        numeric_keys = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_keys:
            print(f"  No numeric features found in {file_name}")
            return
        
        print(f"  Found {len(numeric_keys)} numeric feature(s), visualizing...")
        
        # Create distribution plots for each numeric feature
        for key in numeric_keys[:3]:  # Limit to first 3 numeric columns
            plt.figure(figsize=(12, 6))
            
            # Create histogram and boxplot side by side
            plt.subplot(1, 2, 1)
            sns.histplot(df[key], bins=30, kde=True, color='green')
            
            title = f'Distribution of {key}'
            if total_entries > len(df):
                title += f' (Sample)'
            plt.title(title, fontsize=14)
            
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
            
            print(f"  Saved distribution plot for {key}: {save_path}")
    
    def visualize_dataset_comparison(self, file_info):
        """
        Compare dataset sizes
        
        Args:
            file_info (dict): Dictionary containing file information
        """
        if not file_info:
            return
        
        print("\nCreating dataset size comparison...")
        file_names = []
        dataset_sizes = []
        
        for file_name, info in file_info.items():
            file_names.append(os.path.splitext(file_name)[0])
            dataset_sizes.append(info['total_entries'])
        
        plt.figure(figsize=(10, 6))
        
        # Create bar plot
        sns.barplot(x=file_names, y=dataset_sizes, palette='bright')
        
        plt.title('Dataset Size Comparison', fontsize=16)
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel('Number of Entries', fontsize=14)
        
        # Add value labels on top of bars
        for i, v in enumerate(dataset_sizes):
            plt.text(i, v + max(dataset_sizes)*0.01, f'{v:,}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.output_dir, 'json_dataset_size_comparison.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"  Saved dataset size comparison: {save_path}")
    
    def analyze_file(self, file_path):
        """
        Analyze a single JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            dict: File information
        """
        file_name = os.path.basename(file_path)
        
        # Sample the file
        print(f"\nProcessing {file_name}...")
        sample_data, total_entries = self.sample_json_file(file_path)
        
        if sample_data is None:
            return None
        
        # Analyze structure
        keys = self.analyze_file_structure(file_path, sample_data, total_entries)
        
        # Create DataFrame
        df = self.create_dataframe(sample_data)
        if df is None:
            print(f"  Cannot convert {file_name} to DataFrame")
            return None
        
        # Visualize
        self.visualize_label_distribution(df, file_name, total_entries=total_entries)
        self.visualize_text_length(df, file_name, total_entries=total_entries)
        self.visualize_numeric_features(df, file_name, total_entries=total_entries)
        
        return {'total_entries': total_entries, 'keys': keys}
    
    def run(self):
        """
        Run the complete visualization pipeline
        """
        print("Efficient JSON File Visualizer Started")
        print("=" * 60)
        
        # Step 1: Discover JSON files
        if not self.discover_json_files():
            return False
        
        # Step 2: Analyze each file and create visualizations
        file_info = {}
        for file in self.json_files:
            file_path = os.path.join(self.json_dir, file)
            info = self.analyze_file(file_path)
            if info:
                file_info[file] = info
        
        # Step 3: Compare dataset sizes
        self.visualize_dataset_comparison(file_info)
        
        print("\n" + "=" * 60)
        print("Efficient JSON File Visualizer Completed!")
        print(f"Visualizations saved to: {self.output_dir}")
        return True


def main():
    # Define the directory containing JSON files
    json_dir = r"d:\HuaweiMoveData\Users\86189\Desktop\Uni-LAP-main_desktop\SCM\datasets\cail"
    
    # Create visualizer instance with appropriate sample size
    visualizer = EfficientJSONVisualizer(json_dir, sample_size=20000)
    
    # Run the visualization pipeline
    visualizer.run()


if __name__ == "__main__":
    main()
