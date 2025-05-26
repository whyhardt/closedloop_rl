"""
For each SINDy coefficient column, plots its value vs. four behavioral metrics 
(switch_rate, stay_after_reward, perseveration, avg_reward) in a 2×2 grid. 
Points are color‐coded by participant age, and only participants ≤45 years old are included. 
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def main():
    file_path = Path('AAAAsindy_analysis_with_metrics.csv')
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found—update `file_path` to your CSV location.")
    df = pd.read_csv(file_path)
    
    age_col = 'slcn_age - years'
    if age_col not in df.columns:
        raise KeyError(f"Column '{age_col}' not found in data.")
    df = df[df[age_col] <= 45].copy()
    df['age'] = df[age_col]
    
    #just so the rest will be corelated with 4 behavs
    exclude = [
        'participant_id', age_col,
        'switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward',
        'beta_reward', 'beta_choice', 'params_', 'total_params',
        'nll_', 'trial_likelihood_', 'bic_', 'aic_',
        'n_parameters_', 'metric_n_trials', 'embedding_', 'n_trials'
    ]
    coeffs = [c for c in df.columns if not any(c.startswith(pref) for pref in exclude)]
    behavioral = ['switch_rate', 'stay_after_reward', 'perseveration', 'avg_reward']
    
    # normalization for colorbar
    ages = df['age'].values
    norm = plt.Normalize(vmin=ages.min(), vmax=ages.max())
    cmap = 'viridis'
    
    output_dir = Path('/Users/martynaplomecka/closedloop_rl/analysis/plots/new_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    

    for coeff in coeffs:
        vals = df[coeff].values
        if np.allclose(vals, 0):
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
        axes = axes.flatten()
        
        for ax, metric in zip(axes, behavioral):
            scatter = ax.scatter(vals, df[metric], c=ages, cmap=cmap, norm=norm, alpha=0.8)
            ax.set_xlabel(coeff)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f"{metric.replace('_', ' ').title()} vs {coeff}")
        
        # = room for colorbar
        plt.subplots_adjust(left=0.08, right=0.85, top=0.92, bottom=0.08, 
                           wspace=0.3, hspace=0.3)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Age', rotation=270, labelpad=15)
        
        fig.suptitle(f"{coeff} Coefficient vs Behavioral Metrics", fontsize=16)
        
        out_path = output_dir / f"{coeff}_vs_behavior.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    main()