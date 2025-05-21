import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

os.makedirs('analysis/plots', exist_ok=True)

df = pd.read_csv('AAAAsindy_analysis_with_metrics.csv')

df = df.rename(columns={'slcn_age - years': 'age'})

#  model metrics to analyze
model_metrics = [
    'nll_spice', 
    'nll_rnn', 
    'trial_likelihood_spice', 
    'trial_likelihood_rnn', 
    'bic_spice', 
    'aic_spice'
]

metric_names = {
    'nll_spice': 'Negative Log-Likelihood (SPICE)',
    'nll_rnn': 'Negative Log-Likelihood (RNN)',
    'trial_likelihood_spice': 'Trial Likelihood (SPICE)',
    'trial_likelihood_rnn': 'Trial Likelihood (RNN)',
    'bic_spice': 'BIC (SPICE)',
    'aic_spice': 'AIC (SPICE)'
}

def calculate_correlation(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if sum(mask) < 3:  # Need at least 3 data points for correlation
        return "Insufficient data", 1.0
    
    r, p = stats.pearsonr(x[mask], y[mask])
    significance = ""
    if p < 0.001:
        significance = "***"
    elif p < 0.01:
        significance = "**"
    elif p < 0.05:
        significance = "*"
    
    corr_text = f"r = {r:.3f}{significance}"
    return corr_text, p

plt.figure(figsize=(18, 12))

for i, metric in enumerate(model_metrics):
    ax = plt.subplot(2, 3, i+1)
    
    # Drop rows with NaN values for this metric or age
    valid_data = df.dropna(subset=[metric, 'age'])
    
    # Only proceed if we have enough data points
    if len(valid_data) > 2:
        sns.scatterplot(x='age', y=metric, data=valid_data, 
                        alpha=0.7, edgecolor='w', s=80, ax=ax)
        
        sns.regplot(x='age', y=metric, data=valid_data, 
                    scatter=False, ci=95, line_kws={'color': 'red'}, ax=ax)
        
        corr_text, p_value = calculate_correlation(valid_data['age'], valid_data[metric])
        
        # Add correlation text
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add linear equation if the correlation is significant
        if p_value < 0.05:
            # Calculate linear regression
            slope, intercept, r_value, _, _ = stats.linregress(
                valid_data['age'].dropna(), valid_data[metric].dropna())
            equation = f"y = {slope:.3f}x + {intercept:.3f}"
            ax.text(0.05, 0.87, equation, transform=ax.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, 
                fontsize=12, ha='center')
    
    # Set titles and labels
    ax.set_title(metric_names[metric], fontsize=14)
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel(metric_names[metric], fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('analysis/plots/age_vs_model_metrics.png', dpi=300, bbox_inches='tight')

for metric in model_metrics:
    plt.figure(figsize=(8, 6))
    
    valid_data = df.dropna(subset=[metric, 'age'])
    
    if len(valid_data) > 2:
        # Create scatter plot
        sns.scatterplot(x='age', y=metric, data=valid_data, 
                        alpha=0.7, edgecolor='w', s=100)
        
        # Add regression line with confidence interval
        sns.regplot(x='age', y=metric, data=valid_data, 
                    scatter=False, ci=95, line_kws={'color': 'red'})
        
        corr_text, p_value = calculate_correlation(valid_data['age'], valid_data[metric])
        
        # Add correlation text
        plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add linear equation if significant
        if p_value < 0.05:
            slope, intercept, r_value, _, _ = stats.linregress(
                valid_data['age'].dropna(), valid_data[metric].dropna())
            equation = f"y = {slope:.3f}x + {intercept:.3f}"
            plt.text(0.05, 0.87, equation, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add jitter to points if there are many overlapping points
        if len(valid_data) > 20:
     
            pass  # Tfis is for proper indentation
            
        # Add data points count
        plt.text(0.05, 0.79, f"n = {len(valid_data)}", transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        plt.text(0.5, 0.5, "Insufficient data", transform=plt.gca().transAxes, 
                fontsize=12, ha='center')
    
    plt.title(f"Age vs {metric_names[metric]}", fontsize=14)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel(metric_names[metric], fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'analysis/plots/age_vs_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()





#  correlation summary table
correlation_data = []
for metric in model_metrics:
    valid_data = df.dropna(subset=[metric, 'age'])
    if len(valid_data) > 2:
        r, p = stats.pearsonr(valid_data['age'], valid_data[metric])
        correlation_data.append({
            'Metric': metric_names[metric],
            'Correlation': r,
            'p-value': p,
            'n': len(valid_data)
        })

corr_summary = pd.DataFrame(correlation_data)
print("\nCorrelation Summary:")
print(corr_summary.to_string(index=False))

corr_summary.to_csv('analysis/plots/age_correlations_summary.csv', index=False)
