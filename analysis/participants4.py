import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
import os
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

output_dir = '/Users/martynaplomecka/closedloop_rl/analysis/plots/clustering_plots'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('AAAAsindy_analysis_with_metrics.csv')
df = df.rename(columns={'slcn_age - years': 'age'})

df = df[df['age'] <= 45].copy()
print(f"Number of participants after age filtering (≤45): {len(df)}")

# embedding features
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
print(f"Found {len(embedding_cols)} embedding dimensions")

# Define behavioral metrics
behavioral_metrics = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward', 'n_trials']

complete_data = df.dropna(subset=embedding_cols + ['age'] + behavioral_metrics)
print(f"Number of participants with complete data: {len(complete_data)}")


# PCA for general structure visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(complete_data[embedding_cols])
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"PCA total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

# t-SNE for cluster visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(complete_data)-1))
tsne_result = tsne.fit_transform(complete_data[embedding_cols])

# optimal number of clusters
max_clusters = min(10, len(complete_data) - 1)
silhouette_scores = []
db_scores = []
ch_scores = []

for n_clusters in range(2, max_clusters + 1):
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(complete_data[embedding_cols])
    
    silhouette = silhouette_score(complete_data[embedding_cols], cluster_labels)
    db = davies_bouldin_score(complete_data[embedding_cols], cluster_labels)
    ch = calinski_harabasz_score(complete_data[embedding_cols], cluster_labels)
    
    silhouette_scores.append(silhouette)
    db_scores.append(db)
    ch_scores.append(ch)
    
    print(f"Clusters: {n_clusters}, Silhouette: {silhouette:.3f}, Davies-Bouldin: {db:.3f}, Calinski-Harabasz: {ch:.3f}")

# Plot clustering metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score (higher is better)')
axes[0].grid(True)

axes[1].plot(range(2, max_clusters + 1), db_scores, 'o-')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Davies-Bouldin Score')
axes[1].set_title('Davies-Bouldin Score (lower is better)')
axes[1].grid(True)

axes[2].plot(range(2, max_clusters + 1), ch_scores, 'o-')
axes[2].set_xlabel('Number of Clusters')
axes[2].set_ylabel('Calinski-Harabasz Score')
axes[2].set_title('Calinski-Harabasz Score (higher is better)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig(f'{output_dir}/cluster_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# optimal number of clusters based on metrics
optimal_n_silhouette = np.argmax(silhouette_scores) + 2
optimal_n_db = np.argmin(db_scores) + 2
optimal_n_ch = np.argmax(ch_scores) + 2

print(f"Optimal number of clusters (Silhouette): {optimal_n_silhouette}")
print(f"Optimal number of clusters (Davies-Bouldin): {optimal_n_db}")
print(f"Optimal number of clusters (Calinski-Harabasz): {optimal_n_ch}")

# Choose the optimal number based on majority voting
optimal_n_clusters = int(np.median([optimal_n_silhouette, optimal_n_db, optimal_n_ch]))
print(f"Selected optimal number of clusters: {optimal_n_clusters}")

# Apply KMeans with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(complete_data[embedding_cols])
complete_data = complete_data.copy()
complete_data['cluster'] = cluster_labels







#PLOTS
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA Plot
scatter1 = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', s=80, alpha=0.8)
axes[0].set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
axes[0].set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
axes[0].set_title(f'PCA Projection with {optimal_n_clusters} Clusters')
axes[0].grid(True, alpha=0.3)

# t-SNE Plot
scatter2 = axes[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis', s=80, alpha=0.8)
axes[1].set_xlabel('t-SNE Component 1')
axes[1].set_ylabel('t-SNE Component 2')
axes[1].set_title(f't-SNE Projection with {optimal_n_clusters} Clusters')
axes[1].grid(True, alpha=0.3)

# Add colorbar
plt.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cbar = fig.colorbar(scatter1, cax=cbar_ax)
cbar.set_label('Cluster', rotation=270, labelpad=15)

plt.savefig(f'{output_dir}/embedding_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# Step 6: Behavioral measures by cluster
n_metrics = len(behavioral_metrics) + 1  # +1 for age
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
fig.suptitle(f'Behavioral Measures by Cluster (Age ≤ 45)', fontsize=16)

# Flatten axes for easier indexing
if n_rows == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# Plot behavioral metrics + age
for i, metric in enumerate(behavioral_metrics + ['age']):
    if i < len(axes):
        ax = axes[i]
        sns.boxplot(x='cluster', y=metric, data=complete_data, palette='viridis', ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(metric.replace("_", " ").title())
        
        # Statistical test for differences between clusters
        groups = [complete_data[complete_data['cluster'] == c][metric].dropna() for c in range(optimal_n_clusters)]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) > 1:
            try:
                f_stat, p_value = f_oneway(*groups)
                test_name = "ANOVA"
            except:
                try:
                    h_stat, p_value = kruskal(*groups)
                    test_name = "Kruskal-Wallis"
                except:
                    test_name = "Test failed"
                    p_value = 1.0
            
            if p_value < 0.001:
                p_text = f"{test_name}: p < 0.001"
            elif p_value < 0.01:
                p_text = f"{test_name}: p < 0.01"
            elif p_value < 0.05:
                p_text = f"{test_name}: p < 0.05"
            else:
                p_text = f"{test_name}: p = {p_value:.3f}"
                
            ax.text(0.05, 0.95, p_text, transform=ax.transAxes, fontsize=9, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

for i in range(n_metrics, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f'{output_dir}/behavioral_by_cluster.png', dpi=300, bbox_inches='tight')
plt.close()

cluster_summary = complete_data.groupby('cluster').agg({
    'age': ['mean', 'std', 'count'],
    'switch_rate': ['mean', 'std'],
    'stay_after_reward': ['mean', 'std'],
    'perseveration': ['mean', 'std'],
    'avg_reward': ['mean', 'std'],
    'n_trials': ['mean', 'std']
})

cluster_summary.to_csv(f'{output_dir}/cluster_summary_statistics.csv')

# : Feature correlation analysis
corr_data = []

for feature in behavioral_metrics + ['age']:
    if feature in complete_data.columns:
        try:
            rho, p = spearmanr(complete_data['cluster'], complete_data[feature])
            corr_data.append({
                'Feature': feature,
                'Correlation': rho,
                'p-value': p
            })
        except:
            print(f"Correlation calculation failed for {feature}")

if corr_data:
    corr_df = pd.DataFrame(corr_data)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    
    # Create correlation barplot
    plt.figure(figsize=(12, 8))
    colors = ['blue' if x >= 0 else 'red' for x in corr_df['Correlation']]
    bars = plt.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
    
    # Add significance stars
    for i, p in enumerate(corr_df['p-value']):
        if p < 0.05:
            x_pos = corr_df['Correlation'].iloc[i] + (0.05 if corr_df['Correlation'].iloc[i] >= 0 else -0.05)
            plt.text(x_pos, i, '*', ha='center', va='center', fontsize=12)
    
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
    plt.xlabel('Spearman Correlation with Cluster')
    plt.title('Feature Importance for Cluster Differentiation')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    corr_df.to_csv(f'{output_dir}/cluster_feature_correlations.csv', index=False)

# Radar chart for cluster profiles
radar_metrics = behavioral_metrics + ['age']
cluster_radar_data = []

for cluster_id in range(optimal_n_clusters):
    cluster_data = complete_data[complete_data['cluster'] == cluster_id]
    if len(cluster_data) > 0:
        cluster_values = []
        for metric in radar_metrics:
            if metric in cluster_data.columns:
                metric_mean = cluster_data[metric].mean()
                metric_std = complete_data[metric].std()
                metric_mean_overall = complete_data[metric].mean()
                
                if metric_std > 0:
                    z_score = (metric_mean - metric_mean_overall) / metric_std
                else:
                    z_score = 0
                    
                cluster_values.append(z_score)
            else:
                cluster_values.append(0)
                
        cluster_radar_data.append(cluster_values)

# Create radar chart
if cluster_radar_data:
    angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for i, values in enumerate(cluster_radar_data):
        values = values + values[:1]
        ax.plot(angles, values, linewidth=2, label=f'Cluster {i} (n={len(complete_data[complete_data["cluster"] == i])})')
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in radar_metrics])
    ax.set_title('Cluster Profiles (Z-scores relative to population mean)', size=15, pad=20)
    ax.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cluster_profiles_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

# : PCA overlay plots with behavioral metrics
n_metrics = len(behavioral_metrics) + 1
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 15))
fig.suptitle('Behavioral Metrics Mapped onto RNN Embedding Space (Age ≤ 45)', fontsize=16)

if n_rows == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for i, metric in enumerate(behavioral_metrics + ['age']):
    if i < len(axes) and metric in complete_data.columns:
        ax = axes[i]
        
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                           c=complete_data[metric], cmap='coolwarm', s=80, alpha=0.8)
        
        # Add cluster centers
        for cluster_idx in range(optimal_n_clusters):
            cluster_points = pca_result[complete_data['cluster'] == cluster_idx]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                ax.text(center[0], center[1], str(cluster_idx), 
                       fontsize=16, ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))
        
        ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title(f'PCA Projection Colored by {metric.replace("_", " ").title()}')
        
        # Add colorbar for each subplot
        plt.colorbar(scatter, ax=ax, label=metric.replace("_", " ").title(), shrink=0.8)
        ax.grid(True, alpha=0.3)

# Hide unused subplots
for i in range(n_metrics, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f'{output_dir}/pca_with_metrics_overlay.png', dpi=300, bbox_inches='tight')
plt.close()

#  Cluster centers analysis
cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=embedding_cols)
cluster_centers_df['cluster'] = range(optimal_n_clusters)

cluster_centers_melted = pd.melt(cluster_centers_df, id_vars=['cluster'], 
                                value_vars=embedding_cols, 
                                var_name='Embedding_Dimension', 
                                value_name='Value')

plt.figure(figsize=(14, 8))
sns.lineplot(x='Embedding_Dimension', y='Value', hue='cluster', data=cluster_centers_melted, 
            palette='viridis', marker='o')
plt.title('Cluster Centers across Embedding Dimensions')
plt.xticks(rotation=90)
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/cluster_centers_by_dimension.png', dpi=300, bbox_inches='tight')
plt.close()

# Hierarchical clustering dendrogram
if len(complete_data) > 1:
    Z = linkage(complete_data[embedding_cols], method='ward')
    
    plt.figure(figsize=(15, 8))
    dendrogram(
        Z,
        truncate_mode='lastp',
        p=optimal_n_clusters * 2,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        color_threshold=0.7 * max(Z[:, 2]) if len(Z) > 0 else None
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or (Cluster Size)')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hierarchical_clustering.png', dpi=300, bbox_inches='tight')
    plt.close()

