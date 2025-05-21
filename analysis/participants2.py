import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

df = pd.read_csv('AAAAsindy_analysis_with_metrics.csv')

df = df.rename(columns={'slcn_age - years': 'age'})

behavioral_measures = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward']
output_metrics = ['bic_spice', 'aic_spice'] #whatever, updtae here

fig = plt.figure(figsize=(15, 12))
outer_grid = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

#  colormap for age
cmap = plt.cm.viridis
norm = Normalize(vmin=df['age'].min(), vmax=df['age'].max())

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Age (years)', fontsize=12)

# Function to calculate correlation and p-value
def calculate_correlation(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if sum(mask) < 3:  # Need at least 3 data points for correlation
        return "Insufficient data"
    
    corr = np.corrcoef(x[mask], y[mask])[0, 1]
    return f"r = {corr:.3f}"

# through output metrics
for i, metric in enumerate(output_metrics):
    inner_grid = gridspec.GridSpecFromSubplotSpec(1, len(behavioral_measures), 
                                             subplot_spec=outer_grid[i], wspace=0.3)
    
    for j, behavior in enumerate(behavioral_measures):
        ax = plt.Subplot(fig, inner_grid[j])
        
        # Filter out NaN values for this specific pair
        valid_data = df.dropna(subset=[behavior, metric, 'age'])
        
        # Only proceed if we have enough data points
        if len(valid_data) > 2:
            # Create scatter plot
            scatter = ax.scatter(valid_data[behavior], valid_data[metric], 
                       c=valid_data['age'], cmap=cmap, norm=norm, 
                       alpha=0.7, edgecolors='w', linewidth=0.5)
            
            corr_text = calculate_correlation(valid_data[behavior], valid_data[metric])
            
            #  regression line
            sns.regplot(x=behavior, y=metric, data=valid_data, 
                       scatter=False, ci=None, line_kws={'color': 'red'}, ax=ax)
            
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, 
                   fontsize=12, ha='center')
        
        ax.set_title(f"{behavior.replace('_', ' ').title()}", fontsize=12)
        ax.set_xlabel(behavior.replace('_', ' ').title(), fontsize=10)
        
        if j == 0:  # Only add y-label to the leftmost subplot
            if metric == 'bic_spice':
                ax.set_ylabel('BIC (Spice Model)', fontsize=10)
            else:
                ax.set_ylabel('AIC (Spice Model)', fontsize=10)
        
        fig.add_subplot(ax)
    
    row_title = "Behavioral Measures vs BIC (Spice Model)" if metric == 'bic_spice' else "Behavioral Measures vs AIC (Spice Model)"
    fig.text(0.5, 0.98 - i*0.48, row_title, ha='center', fontsize=16, fontweight='bold')

import os
os.makedirs('analysis/plots', exist_ok=True)
plt.savefig('analysis/plots/behavioral_vs_bic_aic_scatter.png', dpi=300, bbox_inches='tight')
plt.close()



# second plot for correlation matrix
plt.figure(figsize=(10, 8))

# Select columns for correlation
selected_columns = behavioral_measures + output_metrics + ['age']
corr_df = df[selected_columns].dropna()

corr_matrix = corr_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Behavioral Measures, Model Metrics, and Age', fontsize=14)
plt.tight_layout()
plt.savefig('analysis/plots/correlation_matrix.png', dpi=300)
plt.close()

print("\nsig correlations with BIC and AIC:")
for behavior in behavioral_measures:
    for metric in output_metrics:
        valid_data = df.dropna(subset=[behavior, metric])
        if len(valid_data) > 2:
            corr = np.corrcoef(valid_data[behavior], valid_data[metric])[0, 1]
            if abs(corr) > 0.3:  # Only show correlations stronger than 0.3
                print(f"{behavior} vs {metric}: r = {corr:.3f}")