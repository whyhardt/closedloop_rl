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

os.makedirs('analysis/plots', exist_ok=True)

df = pd.read_csv('AAAAsindy_analysis_with_metrics.csv')
df = df.rename(columns={'slcn_age - years': 'age'})

# Step 1: Extract embedding features
embedding_cols = [col for col in df.columns if col.startswith('embedding_')]

# 
behavioral_metrics = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward', 'n_trials']

# Filter to only include rows with embedding data, behavioral metrics, and age
complete_data = df.dropna(subset=embedding_cols + ['age'] + behavioral_metrics)
print(f"Number of participants with complete data: {len(complete_data)}")

# Step 2: Dimensionality reduction for visualization
# PCA for general structure visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(complete_data[embedding_cols])
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"PCA total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

# t-SNE for cluster visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(complete_data)-1))
tsne_result = tsne.fit_transform(complete_data[embedding_cols])

# Step 3: Determine optimal number of clusters
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




plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(2, max_clusters + 1), silhouette_scores, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score (higher is better)')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(2, max_clusters + 1), db_scores, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Score')
plt.title('Davies-Bouldin Score (lower is better)')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(range(2, max_clusters + 1), ch_scores, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Score')
plt.title('Calinski-Harabasz Score (higher is better)')
plt.grid(True)

plt.tight_layout()
plt.savefig('analysis/plots/cluster_metrics.png', dpi=300)
plt.close()



# Step 4: Choose optimal number of clusters based on metrics
# Silhouette score (higher is better)
optimal_n_silhouette = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters

# Davies-Bouldin (lower is better)
optimal_n_db = np.argmin(db_scores) + 2

# Calinski-Harabasz (higher is better)
optimal_n_ch = np.argmax(ch_scores) + 2

print(f"Optimal number of clusters (Silhouette): {optimal_n_silhouette}")
print(f"Optimal number of clusters (Davies-Bouldin): {optimal_n_db}")
print(f"Optimal number of clusters (Calinski-Harabasz): {optimal_n_ch}")

# Choose the optimal number based on majority voting or pick one of the metrics
optimal_n_clusters = int(np.median([optimal_n_silhouette, optimal_n_db, optimal_n_ch]))
print(f"Selected optimal number of clusters: {optimal_n_clusters}")

# Apply KMeans with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(complete_data[embedding_cols])
complete_data['cluster'] = cluster_labels

# Step 5: Create visualizations of the embeddings with clusters
plt.figure(figsize=(20, 10))

# PCA Plot
plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis', s=80, alpha=0.8)
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title(f'PCA Projection with {optimal_n_clusters} Clusters')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# t-SNE Plot
plt.subplot(1, 2, 2)
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis', s=80, alpha=0.8)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title(f't-SNE Projection with {optimal_n_clusters} Clusters')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/plots/embedding_clusters.png', dpi=300)
plt.close()

# relationship between clusters and behavioral metrics
fig = plt.figure(figsize=(20, 15))
fig.suptitle(f'Behavioral Measures by Cluster', fontsize=16)

n_metrics = len(behavioral_metrics) + 1  # +1 for age
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

# Behavioral metrics + age
for i, metric in enumerate(behavioral_metrics + ['age']):
    ax = plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='cluster', y=metric, data=complete_data, palette='viridis')
    plt.title(f'{metric} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(metric)
    
    # Statistical test for differences between clusters
    groups = [complete_data[complete_data['cluster'] == c][metric].dropna() for c in range(optimal_n_clusters)]
    groups = [g for g in groups if len(g) > 0]  #
    
    if len(groups) > 1:  # Need at least 2 groups for comparison
        try:
            f_stat, p_value = f_oneway(*groups)
            test_name = "ANOVA"
        except:
            # If ANOVA fails, try non-parametric Kruskal-Wallis?
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
            
        plt.text(0.05, 0.95, p_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])  
plt.savefig('analysis/plots/behavioral_by_cluster.png', dpi=300)
plt.close()

cluster_summary = complete_data.groupby('cluster').agg({
    'age': ['mean', 'std', 'count'],
    'switch_rate': ['mean', 'std'],
    'stay_after_reward': ['mean', 'std'],
    'perseveration': ['mean', 'std'],
    'avg_reward': ['mean', 'std'],
    'n_trials': ['mean', 'std']
})

cluster_summary.to_csv('analysis/plots/cluster_summary_statistics.csv')

corr_data = []

# Check which features are most correlated with cluster labels
for feature in behavioral_metrics + ['age']:
    if feature in complete_data.columns:
        # Use Spearman correlation for ordinal data like clusters
        try:
            rho, p = spearmanr(complete_data['cluster'], complete_data[feature])
            corr_data.append({
                'Feature': feature,
                'Correlation': rho,
                'p-value': p
            })
        except:
            print(f"fail for {feature}")

corr_df = pd.DataFrame(corr_data)
corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)

# correlation barplot
plt.figure(figsize=(12, 8))
colors = ['blue' if x >= 0 else 'red' for x in corr_df['Correlation']]
bars = plt.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
for i, p in enumerate(corr_df['p-value']):
    if p < 0.05:
        plt.text(corr_df['Correlation'].iloc[i] + (0.05 if corr_df['Correlation'].iloc[i] >= 0 else -0.05), 
                i, '*', ha='center', va='center', fontsize=12)

plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
plt.xlabel('Spearman Correlation with Cluster')
plt.title('Feature Importance for Cluster Differentiation')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/cluster_feature_importance.png', dpi=300)
plt.close()

corr_df.to_csv('analysis/plots/cluster_feature_correlations.csv', index=False)



# Step 9: Create radar chart to visualize cluster profiles
plt.figure(figsize=(12, 10))
cluster_radar_data = []

# Select the metrics to compare (all behavioral metrics + age)
radar_metrics = behavioral_metrics + ['age']

# Get the mean values for each cluster and metric
for cluster_id in range(optimal_n_clusters):
    cluster_data = complete_data[complete_data['cluster'] == cluster_id]
    if len(cluster_data) > 0:
        # Get mean values and normalize 
        cluster_values = []
        for metric in radar_metrics:
            if metric in cluster_data.columns:
                # Calculate z-score for this metric within this cluster
                metric_mean = cluster_data[metric].mean()
                metric_std = complete_data[metric].std()  # Use overall std for normalization
                metric_mean_overall = complete_data[metric].mean()  # Use overall mean as reference
                
                if metric_std > 0:  # Avoid division by zero
                    z_score = (metric_mean - metric_mean_overall) / metric_std
                else:
                    z_score = 0
                    
                cluster_values.append(z_score)
            else:
                cluster_values.append(0)  # Default if metric not available
                
        cluster_radar_data.append(cluster_values)

# Spider plot for cluster profiles
angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, values in enumerate(cluster_radar_data):
    values = values + values[:1]  # Close the loop
    ax.plot(angles, values, linewidth=2, label=f'Cluster {i} (n={len(complete_data[complete_data["cluster"] == i])})')
    ax.fill(angles, values, alpha=0.1)

# Add metric labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_title('Cluster Profiles (Z-scores relative to population mean)', size=15)
ax.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig('analysis/plots/cluster_profiles_radar.png', dpi=300)
plt.close()

#  overlay plots showing how behavioral metrics map onto embedding space
plt.figure(figsize=(18, 15))
plt.suptitle('Behavioral Metrics Mapped onto RNN Embedding Space', fontsize=16)

# subplot for each behavioral metric + age
n_metrics = len(behavioral_metrics) + 1  # +1 for age
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

for i, metric in enumerate(behavioral_metrics + ['age']):
    if metric in complete_data.columns:
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # scatter plot of embeddings colored by the metric
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=complete_data[metric], cmap='coolwarm', s=80, alpha=0.8)
        
        # cluster boundaries or centers
        for cluster_idx in range(optimal_n_clusters):
            cluster_points = pca_result[complete_data['cluster'] == cluster_idx]
            if len(cluster_points) > 0:
                center = cluster_points.mean(axis=0)
                plt.text(center[0], center[1], str(cluster_idx), 
                        fontsize=16, ha='center', va='center',
                        bbox=dict(boxstyle='circle', facecolor='white', alpha=0.7))
        
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'PCA Projection Colored by {metric}')
        plt.colorbar(scatter, label=metric)
        plt.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for suptitle
plt.savefig('analysis/plots/pca_with_metrics_overlay.png', dpi=300)
plt.close()




# cluster centers across embedding dimensions to understand what each dimension encodes
plt.figure(figsize=(14, 8))
cluster_centers = kmeans.cluster_centers_
cluster_centers_df = pd.DataFrame(cluster_centers, columns=embedding_cols)

cluster_centers_df['cluster'] = range(optimal_n_clusters)

cluster_centers_melted = pd.melt(cluster_centers_df, id_vars=['cluster'], 
                                value_vars=embedding_cols, 
                                var_name='Embedding Dimension', 
                                value_name='Value')

# Plot cluster centers by dimension
sns.lineplot(x='Embedding Dimension', y='Value', hue='cluster', data=cluster_centers_melted, 
            palette='viridis', marker='o')
plt.title('Cluster Centers across Embedding Dimensions')
plt.xticks(rotation=90)
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('analysis/plots/cluster_centers_by_dimension.png', dpi=300)
plt.close()



Z = linkage(complete_data[embedding_cols], method='ward')

# dendrogram
plt.figure(figsize=(15, 8))
dendrogram(
    Z,
    truncate_mode='lastp',  # Show only the last p merged clusters
    p=optimal_n_clusters * 2,  # Show twice the optimal number for context
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    color_threshold=0.7 * max(Z[:, 2])  # Color threshold
)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('analysis/plots/hierarchical_clustering.png', dpi=300)
plt.close()

print(f"Analysis complete. Used {optimal_n_clusters} clusters based on mathematical criteria.")
print("All outputs saved to analysis/plots/ directory.")