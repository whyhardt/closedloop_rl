# Improved clustering analysis with clearer visualizations and explanations
# This code analyzes participant embeddings to find behavioral clusters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from scipy.stats import pearsonr
import os

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})

output_dir = "/Users/martynaplomecka/closedloop_rl/analysis/plots/new_plots/clustering_embeddings"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('AAAAsindy_analysis_with_metrics.csv')

behavioral_metrics = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward']
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]

complete_df = df.dropna(subset=behavioral_metrics + embedding_cols).copy()
complete_df = complete_df.reset_index(drop=True)

print(f"Dataset: {len(complete_df)} participants, {len(embedding_cols)} embedding dimensions")

# Extract and standardize embeddings
X_embeddings = complete_df[embedding_cols].values
participant_ids = complete_df['participant_id'].values
behaviors = complete_df[behavioral_metrics].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_embeddings)

#COR ANALYSIS
print("Calculating correlations between embeddings and behavioral metrics...")
correlations = []
for metric in behavioral_metrics:
    for col in embedding_cols:
        r, p = pearsonr(complete_df[col], complete_df[metric])
        correlations.append({
            'embedding': col,
            'behavioral_metric': metric,
            'correlation': r,
            'p_value': p
        })

corr_df = pd.DataFrame(correlations)
corr_df['abs_corr'] = corr_df['correlation'].abs()
corr_df = corr_df.sort_values('abs_corr', ascending=False)

print("\nTop 10 correlations between embeddings and behavioral metrics:")
print(corr_df.head(10))

# DIM RED

#  t-SNE
perplexity = min(30, max(5, len(complete_df)//4))
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
           learning_rate=200, n_iter=1000)
tsne_result = tsne.fit_transform(X_scaled)

#  UMAP
n_neighbors = min(15, max(3, len(complete_df)//5))
umap_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                        min_dist=0.1, random_state=42)
umap_result = umap_reducer.fit_transform(X_scaled)

# clustering

from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Test different numbers of clusters
max_clusters = min(10, len(complete_df)//10)  # Don't test more than we can reasonably interpret?
cluster_range = range(2, max_clusters + 1)

# Method 1: Elbow method (within-cluster sum of squares)
wcss = []
for k in cluster_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)

# Method 2: Silhouette analysis
silhouette_scores = []
for k in cluster_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_labels_temp = kmeans_temp.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels_temp)
    silhouette_scores.append(silhouette_avg)

# Find optimal k using silhouette score (higher is better)
optimal_k_silhouette = cluster_range[np.argmax(silhouette_scores)]

# Calculate elbow point (look for the "knee" in the WCSS curve)
# Use second derivative to find the elbow
wcss_array = np.array(wcss)
if len(wcss_array) >= 3:
    # Calculate second derivative
    second_derivatives = np.diff(wcss_array, 2)
    # Find the point where curvature changes most (elbow)
    elbow_k = cluster_range[np.argmax(second_derivatives) + 2]  # +2 because diff reduces array size
else:
    elbow_k = 3  # Default fallback

print(f"Cluster optimization results:")
print(f"  Silhouette method suggests: {optimal_k_silhouette} clusters (score: {max(silhouette_scores):.3f})")
print(f"  Elbow method suggests: {elbow_k} clusters")
print(f"  Silhouette scores: {dict(zip(cluster_range, [f'{s:.3f}' for s in silhouette_scores]))}")

# Choose the number of clusters (prioritize silhouette score but consider interpretability)
if max(silhouette_scores) > 0.3:  # Good silhouette score
    n_clusters = optimal_k_silhouette
    selection_method = "silhouette analysis"
elif max(silhouette_scores) > 0.2:  # Moderate silhouette score
    # Choose between silhouette and elbow, prefer smaller number for interpretability
    n_clusters = min(optimal_k_silhouette, elbow_k)
    selection_method = "combined silhouette + elbow"
else:  # Poor silhouette scores, use elbow or default
    n_clusters = min(elbow_k, 4)  # Cap at 4 for interpretability
    selection_method = "elbow method (poor silhouette scores)"

print(f"\nSelected {n_clusters} clusters using {selection_method}")

# Final clustering with optimal number
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)

# Visualize cluster selection process
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot elbow curve
ax1.plot(cluster_range, wcss, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Elbow at k={elbow_k}')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Within-Cluster Sum of Squares')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot silhouette scores
ax2.plot(cluster_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k_silhouette, color='red', linestyle='--', alpha=0.7, 
           label=f'Best silhouette at k={optimal_k_silhouette}')
ax2.axhline(y=0.3, color='orange', linestyle=':', alpha=0.7, label='Good threshold (0.3)')
ax2.axhline(y=0.2, color='yellow', linestyle=':', alpha=0.7, label='Fair threshold (0.2)')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Average Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_optimization.png'), 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Create results dataframe
results_df = pd.DataFrame({
    'participant_id': participant_ids,
    'tsne_x': tsne_result[:, 0],
    'tsne_y': tsne_result[:, 1],
    'umap_x': umap_result[:, 0],
    'umap_y': umap_result[:, 1],
    'cluster': cluster_labels
})

# Add behavioral metrics
for metric in behavioral_metrics:
    results_df[metric] = complete_df[metric].values

# === ANALYZE CLUSTERS ===
print("\nAnalyzing cluster characteristics...")

cluster_stats = []
for cluster in range(n_clusters):
    cluster_data = results_df[results_df['cluster'] == cluster]
    stats = {'cluster': cluster, 'n_participants': len(cluster_data)}
    
    for metric in behavioral_metrics:
        stats[f'{metric}_mean'] = cluster_data[metric].mean()
        stats[f'{metric}_std'] = cluster_data[metric].std()
    
    cluster_stats.append(stats)

cluster_stats_df = pd.DataFrame(cluster_stats)
print("\nCluster Summary:")
print(cluster_stats_df.round(3))

# === IMPROVED VISUALIZATIONS ===

# 1. Main clustering visualization (larger, clearer)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# t-SNE clusters
scatter1 = axes[0].scatter(results_df['tsne_x'], results_df['tsne_y'], 
                         c=results_df['cluster'], cmap='tab10', 
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[0].set_title('t-SNE: Participant Clusters\n(Similar participants are close together)', 
                 fontsize=14, pad=20)
axes[0].set_xlabel('t-SNE Dimension 1')
axes[0].set_ylabel('t-SNE Dimension 2')
axes[0].grid(True, alpha=0.3)

# Add cluster labels
for cluster in range(n_clusters):
    cluster_data = results_df[results_df['cluster'] == cluster]
    center_x = cluster_data['tsne_x'].mean()
    center_y = cluster_data['tsne_y'].mean()
    axes[0].annotate(f'Cluster {cluster}', (center_x, center_y), 
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# UMAP clusters
scatter2 = axes[1].scatter(results_df['umap_x'], results_df['umap_y'], 
                         c=results_df['cluster'], cmap='tab10', 
                         s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1].set_title('UMAP: Participant Clusters\n(Alternative view of the same clusters)', 
                 fontsize=14, pad=20)
axes[1].set_xlabel('UMAP Dimension 1')
axes[1].set_ylabel('UMAP Dimension 2')
axes[1].grid(True, alpha=0.3)

# Add cluster labels for UMAP
for cluster in range(n_clusters):
    cluster_data = results_df[results_df['cluster'] == cluster]
    center_x = cluster_data['umap_x'].mean()
    center_y = cluster_data['umap_y'].mean()
    axes[1].annotate(f'Cluster {cluster}', (center_x, center_y), 
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Add colorbar
cbar = plt.colorbar(scatter1, ax=axes, orientation='horizontal', 
                   fraction=0.05, pad=0.1, shrink=0.8)
cbar.set_label('Cluster ID', fontsize=12)
cbar.set_ticks(range(n_clusters))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'clear_clustering_overview.png'), 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 2. Behavioral differences between clusters (clearer visualization)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

# Create better behavioral metric names for display
metric_names = {
    'switch_rate': 'Switch Rate',
    'stay_after_reward': 'Stay After Reward',
    'perseveration': 'Perseveration',
    'avg_reward': 'Average Reward'
}

colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

for i, metric in enumerate(behavioral_metrics):
    ax = axes[i]
    
    # Create violin plot for better distribution visualization
    parts = ax.violinplot([results_df[results_df['cluster'] == c][metric].values 
                          for c in range(n_clusters)], 
                         positions=range(n_clusters), showmeans=True, showmedians=True)
    
    # Color the violin plots
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Add individual points
    for cluster in range(n_clusters):
        cluster_data = results_df[results_df['cluster'] == cluster][metric]
        x_pos = [cluster] * len(cluster_data)
        ax.scatter(x_pos, cluster_data, alpha=0.4, s=20, color='black')
    
    ax.set_title(metric_names[metric], fontsize=12, pad=15)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Value')
    ax.set_xticks(range(n_clusters))
    ax.grid(True, alpha=0.3)
    
    # Add statistical annotations
    for cluster in range(n_clusters):
        cluster_data = results_df[results_df['cluster'] == cluster][metric]
        mean_val = cluster_data.mean()
        ax.text(cluster, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.2f}', 
               ha='center', fontsize=10, fontweight='bold')

plt.suptitle('How Do Clusters Differ in Behavior?\n', 
            fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Fix tight_layout warning
plt.savefig(os.path.join(output_dir, 'clear_behavioral_differences.png'), 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 3. Show behavioral patterns in embedding space
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(behavioral_metrics):
    ax = axes[i]
    
    # Use t-SNE coordinates colored by behavioral metric
    scatter = ax.scatter(results_df['tsne_x'], results_df['tsne_y'], 
                        c=results_df[metric], cmap='RdYlBu_r', 
                        s=50, alpha=0.8, edgecolors='black', linewidth=0.3)
    
    ax.set_title(f'{metric_names[metric]}\n(Color shows metric value)', fontsize=12)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(metric, fontsize=10)

plt.suptitle('Where Do Different Behaviors Appear in Embedding Space?\n', 
            fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Fix tight_layout warning
plt.savefig(os.path.join(output_dir, 'behavioral_patterns_in_space.png'), 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# === CORRELATION HEATMAP (from your original code) ===
print("Creating correlation heatmap...")
plt.figure(figsize=(14, 8))

# Pivot the correlation dataframe for heatmap
pivot_df = corr_df.pivot(index='behavioral_metric', columns='embedding', values='correlation')

# Select top correlations for readability
top_embeddings = corr_df.sort_values('abs_corr', ascending=False)['embedding'].unique()[:min(15, len(embedding_cols))]
pivot_subset = pivot_df[top_embeddings]

# Create heatmap with better formatting
sns.heatmap(pivot_subset, cmap='RdBu_r', annot=True, fmt=".3f", center=0, 
           cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
           linewidths=0.5, square=False)

plt.title('Correlation Between Top Embedding Dimensions and Behavioral Metrics\n', 
         fontsize=14, pad=20)
plt.xlabel('Embedding Dimensions', fontsize=12)
plt.ylabel('Behavioral Metrics', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), 
           dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# === DETAILED CORRELATION ANALYSIS ===
print("\n" + "="*60)
print("CORRELATION ANALYSIS RESULTS")
print("="*60)

# Show strongest correlations for each behavioral metric
for metric in behavioral_metrics:
    metric_corrs = corr_df[corr_df['behavioral_metric'] == metric].sort_values('abs_corr', ascending=False)
    strongest = metric_corrs.iloc[0]
    print(f"\n{metric.upper()}:")
    print(f"  Strongest correlation: {strongest['embedding']} (r = {strongest['correlation']:.3f}, p = {strongest['p_value']:.3f})")
    
    # Show top 3 correlations for this metric
    print(f"  Top 3 embedding correlations:")
    for i in range(min(3, len(metric_corrs))):
        row = metric_corrs.iloc[i]
        print(f"    {i+1}. {row['embedding']}: r = {row['correlation']:.3f} (p = {row['p_value']:.3f})")

# Significant correlations summary
significant_corrs = corr_df[corr_df['p_value'] < 0.05]
print(f"\nSIGNIFICANT CORRELATIONS (p < 0.05): {len(significant_corrs)} out of {len(corr_df)}")
strong_corrs = significant_corrs[significant_corrs['abs_corr'] > 0.3]
print(f"STRONG CORRELATIONS (|r| > 0.3): {len(strong_corrs)}")

if len(strong_corrs) > 0:
    print("\nStrongest significant correlations:")
    for i, (_, row) in enumerate(strong_corrs.head(5).iterrows()):
        print(f"  {i+1}. {row['embedding']} ↔ {row['behavioral_metric']}: r = {row['correlation']:.3f} (p = {row['p_value']:.3f})")

# === SUMMARY STATISTICS ===
print("\n" + "="*60)
print("CLUSTERING ANALYSIS SUMMARY")
print("="*60)
print(f"• Analyzed {len(complete_df)} participants")
print(f"• Found {n_clusters} distinct behavioral clusters")
print(f"• Used {len(embedding_cols)} embedding dimensions")

print(f"\nCluster Sizes:")
for cluster in range(n_clusters):
    count = len(results_df[results_df['cluster'] == cluster])
    percentage = (count / len(results_df)) * 100
    print(f"  Cluster {cluster}: {count} participants ({percentage:.1f}%)")

print("• Clusters represent groups of participants with similar neural embedding patterns")
print("• Each cluster shows different behavioral tendencies")
print("• This suggests that neural patterns predict behavioral strategies")

# Calculate the most distinctive behavioral differences between clusters
print(f"\nMost Distinctive Behavioral Differences Between Clusters:")
for metric in behavioral_metrics:
    cluster_means = [results_df[results_df['cluster'] == c][metric].mean() 
                    for c in range(n_clusters)]
    max_diff = max(cluster_means) - min(cluster_means)
    best_cluster = cluster_means.index(max(cluster_means))
    worst_cluster = cluster_means.index(min(cluster_means))
    print(f"  {metric}: {max_diff:.3f} difference (Cluster {worst_cluster}: {min(cluster_means):.3f} → Cluster {best_cluster}: {max(cluster_means):.3f})")






# Overall analysis interpretation
print(f"\nOVERALL INTERPRETATION:")
print(f"• Strongest embedding-behavior correlation: r = {corr_df.iloc[0]['correlation']:.3f}")
print(f"  ({corr_df.iloc[0]['embedding']} ↔ {corr_df.iloc[0]['behavioral_metric']})")
print(f"• Number of significant correlations: {len(significant_corrs)}/{len(corr_df)} ({len(significant_corrs)/len(corr_df)*100:.1f}%)")

if len(strong_corrs) > 0:
    print(f"• Strong correlations found: Neural embeddings DO predict behavior")
else:
    print(f"• Few strong correlations: Neural embeddings may not strongly predict behavior")

