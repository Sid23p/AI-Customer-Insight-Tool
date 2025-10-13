"""
AI-Powered Customer Insight Tool
Customer Segmentation using RFM Analysis and K-Means Clustering

This script implements a comprehensive customer segmentation pipeline that:
1. Loads and cleans retail transaction data
2. Calculates RFM (Recency, Frequency, Monetary) metrics
3. Applies preprocessing and scaling
4. Determines optimal number of clusters using elbow method and silhouette analysis
5. Performs K-Means clustering for customer segmentation
6. Generates visualizations for portfolio submission

Author: Siddhant Patil
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def prepare_rfm_data(file_path):
    """
    Function 1: Load, clean, and calculate RFM metrics from transaction data
    
    Args:
        file_path (str): Path to the CSV file containing transaction data
        
    Returns:
        pd.DataFrame: DataFrame with RFM metrics calculated for each customer
    """
    print("Loading and processing transaction data...")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Rename columns for clarity (if needed)
    df.columns = df.columns.str.strip()
    
    # Display basic info about the dataset
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Mandatory filtering: Drop rows where CustomerID is missing
    df_clean = df.dropna(subset=['CustomerID']).copy()
    print(f"After removing missing CustomerID: {df_clean.shape}")
    
    # Filter out invalid transactions (Quantity <= 0 or UnitPrice <= 0)
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    print(f"After removing invalid transactions: {df_clean.shape}")
    
    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Calculate TotalPrice (Quantity × UnitPrice)
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    # Calculate snapshot date (maximum date in the dataset)
    snapshot_date = df_clean['InvoiceDate'].max()
    print(f"Snapshot date: {snapshot_date}")
    
    # Calculate RFM metrics by grouping by CustomerID
    rfm_df = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency (unique invoices)
        'TotalPrice': 'sum'  # Monetary
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    })
    
    print(f"RFM dataset shape: {rfm_df.shape}")
    print(f"RFM statistics:")
    print(rfm_df.describe())
    
    return rfm_df


def preprocess_and_scale(rfm_df):
    """
    Function 2: Apply outlier mitigation and standardization to RFM data
    
    Args:
        rfm_df (pd.DataFrame): DataFrame with RFM metrics
        
    Returns:
        pd.DataFrame: DataFrame with scaled RFM features
    """
    print("\nPreprocessing and scaling RFM data...")
    
    # Create a copy to avoid modifying the original
    rfm_scaled = rfm_df.copy()
    
    # Outlier mitigation: Apply logarithmic transformation to Monetary feature
    # Add 1 to avoid log(0) issues
    rfm_scaled['Monetary_log'] = np.log1p(rfm_scaled['Monetary'])
    
    print("Applied logarithmic transformation to Monetary feature")
    print(f"Monetary statistics before log transform:")
    print(f"  Mean: {rfm_df['Monetary'].mean():.2f}")
    print(f"  Std: {rfm_df['Monetary'].std():.2f}")
    print(f"  Skewness: {rfm_df['Monetary'].skew():.2f}")
    
    print(f"Monetary statistics after log transform:")
    print(f"  Mean: {rfm_scaled['Monetary_log'].mean():.2f}")
    print(f"  Std: {rfm_scaled['Monetary_log'].std():.2f}")
    print(f"  Skewness: {rfm_scaled['Monetary_log'].skew():.2f}")
    
    # Select features for scaling (R, F, and logged M)
    features_to_scale = ['Recency', 'Frequency', 'Monetary_log']
    rfm_features = rfm_scaled[features_to_scale]
    
    # Apply StandardScaler
    scaler = StandardScaler()
    rfm_scaled_features = scaler.fit_transform(rfm_features)
    
    # Create DataFrame with scaled features
    rfm_scaled_df = pd.DataFrame(
        rfm_scaled_features,
        columns=[f'{col}_scaled' for col in features_to_scale],
        index=rfm_df.index
    )
    
    print(f"Scaled features shape: {rfm_scaled_df.shape}")
    print("Standardization completed")
    
    return rfm_scaled_df


def evaluate_optimal_k(rfm_scaled_df, max_k=10):
    """
    Function 3: Determine optimal number of clusters using elbow method and silhouette analysis
    
    Args:
        rfm_scaled_df (pd.DataFrame): DataFrame with scaled RFM features
        max_k (int): Maximum number of clusters to evaluate
        
    Returns:
        int: Optimal number of clusters
    """
    print(f"\nEvaluating optimal number of clusters (k=2 to {max_k})...")
    
    # Initialize lists to store metrics
    wss_scores = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    # Calculate metrics for each k
    for k in k_values:
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(rfm_scaled_df)
        
        # Calculate WSS (inertia)
        wss_scores.append(kmeans.inertia_)
        
        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(rfm_scaled_df, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"k={k}: WSS={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.3f}")
    
    # Find optimal k based on highest silhouette score
    optimal_k = k_values[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)
    
    print(f"\nOptimal k: {optimal_k} (Silhouette Score: {max_silhouette:.3f})")
    
    # Create visualization with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow Plot (WSS vs k)
    ax1.plot(k_values, wss_scores, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Within-Cluster Sum of Squares (WSS)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Optimal k = {optimal_k}')
    ax1.legend()
    
    # Silhouette Score Bar Chart
    bars = ax2.bar(k_values, silhouette_scores, color='skyblue', alpha=0.7, edgecolor='navy')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight the optimal k
    bars[optimal_k-2].set_color('red')
    bars[optimal_k-2].set_alpha(1.0)
    
    # Add value labels on bars
    for i, (k, score) in enumerate(zip(k_values, silhouette_scores)):
        ax2.text(k, score + 0.005, f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('k_evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved k_evaluation_metrics.png")
    plt.show()
    
    return optimal_k


def perform_final_clustering(rfm_scaled_df, rfm_df, n_clusters):
    """
    Function 4: Perform final K-Means clustering and profile customer segments
    
    Args:
        rfm_scaled_df (pd.DataFrame): DataFrame with scaled RFM features
        rfm_df (pd.DataFrame): Original RFM DataFrame
        n_clusters (int): Number of clusters for final clustering
        
    Returns:
        pd.DataFrame: RFM DataFrame with cluster labels and profiling
    """
    print(f"\nPerforming final clustering with k={n_clusters}...")
    
    # Perform final K-Means clustering
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(rfm_scaled_df)
    
    # Add cluster labels to original RFM DataFrame
    rfm_final = rfm_df.copy()
    rfm_final['Cluster'] = cluster_labels
    
    # Calculate cluster profiles (mean R, F, M for each cluster)
    cluster_profiles = rfm_final.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)
    
    # Add cluster sizes
    cluster_sizes = rfm_final['Cluster'].value_counts().sort_index()
    cluster_profiles['Size'] = cluster_sizes
    cluster_profiles['Percentage'] = (cluster_sizes / len(rfm_final) * 100).round(1)
    
    print("\nCluster Profiles:")
    print(cluster_profiles)
    
    # Calculate final silhouette score
    final_silhouette = silhouette_score(rfm_scaled_df, cluster_labels)
    print(f"\nFinal Silhouette Score: {final_silhouette:.3f}")
    
    return rfm_final


def visualize_clusters(rfm_final, n_clusters):
    """
    Function 5: Create visualization showing cluster profiles
    
    Args:
        rfm_final (pd.DataFrame): RFM DataFrame with cluster labels
        n_clusters (int): Number of clusters
    """
    print(f"\nCreating cluster profile visualization...")
    
    # Create figure with subplots for each RFM metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors for clusters
    colors = sns.color_palette("husl", n_clusters)
    
    # Plot Recency
    sns.boxplot(data=rfm_final, x='Cluster', y='Recency', ax=axes[0], 
                palette=colors, showfliers=False)
    axes[0].set_title('Recency Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Recency (Days)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Frequency
    sns.boxplot(data=rfm_final, x='Cluster', y='Frequency', ax=axes[1], 
                palette=colors, showfliers=False)
    axes[1].set_title('Frequency Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('Frequency (Unique Invoices)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Plot Monetary
    sns.boxplot(data=rfm_final, x='Cluster', y='Monetary', ax=axes[2], 
                palette=colors, showfliers=False)
    axes[2].set_title('Monetary Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Cluster', fontsize=12)
    axes[2].set_ylabel('Monetary Value', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Add cluster size annotations
    cluster_sizes = rfm_final['Cluster'].value_counts().sort_index()
    for i, size in enumerate(cluster_sizes):
        percentage = (size / len(rfm_final)) * 100
        for ax in axes:
            ax.text(i, ax.get_ylim()[1] * 0.95, f'n={size}\n({percentage:.1f}%)', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('cluster_profile_box_plot.png', dpi=300, bbox_inches='tight')
    print("Saved cluster_profile_box_plot.png")
    plt.show()
    
    # Print detailed cluster analysis
    print("\nDetailed Cluster Analysis:")
    cluster_analysis = rfm_final.groupby('Cluster').agg({
        'Recency': ['mean', 'std'],
        'Frequency': ['mean', 'std'],
        'Monetary': ['mean', 'std']
    }).round(2)
    
    print(cluster_analysis)
    
    # Provide business interpretation
    print("\nBusiness Interpretation:")
    for cluster in range(n_clusters):
        cluster_data = rfm_final[rfm_final['Cluster'] == cluster]
        avg_recency = cluster_data['Recency'].mean()
        avg_frequency = cluster_data['Frequency'].mean()
        avg_monetary = cluster_data['Monetary'].mean()
        size = len(cluster_data)
        percentage = (size / len(rfm_final)) * 100
        
        print(f"\nCluster {cluster} ({size} customers, {percentage:.1f}%):")
        print(f"  Recency: {avg_recency:.1f} days (avg days since last purchase)")
        print(f"  Frequency: {avg_frequency:.1f} (avg unique invoices)")
        print(f"  Monetary: ${avg_monetary:.2f} (avg total spending)")


if __name__ == "__main__":
    """
    Main execution block: Sequential execution of the AI pipeline
    """
    print("=" * 60)
    print("AI-POWERED CUSTOMER INSIGHT TOOL")
    print("Customer Segmentation using RFM Analysis and K-Means Clustering")
    print("=" * 60)
    
    # Set the data file path
    DATA_FILE = "online_retail.csv"
    
    try:
        # Step 1: Prepare RFM data
        print("\nSTEP 1: Data Preparation and RFM Calculation")
        rfm_df = prepare_rfm_data(DATA_FILE)
        
        # Step 2: Preprocess and scale data
        print("\nSTEP 2: Data Preprocessing and Scaling")
        rfm_scaled_df = preprocess_and_scale(rfm_df)
        
        # Step 3: Evaluate optimal number of clusters
        print("\nSTEP 3: Optimal Cluster Evaluation")
        optimal_k = evaluate_optimal_k(rfm_scaled_df, max_k=10)
        
        # Step 4: Perform final clustering
        print("\nSTEP 4: Final Clustering and Profiling")
        rfm_final = perform_final_clustering(rfm_scaled_df, rfm_df, optimal_k)
        
        # Step 5: Visualize clusters
        print("\nSTEP 5: Cluster Visualization")
        visualize_clusters(rfm_final, optimal_k)
        
        print("\n" + "=" * 60)
        print("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY!")
        print("Generated files:")
        print("- k_evaluation_metrics.png")
        print("- cluster_profile_box_plot.png")
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DATA_FILE}'")
        print("Please ensure the file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your data and try again.")
