import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.stats.multitest import multipletests
import textwrap
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='dezfouli_final_df_sindy_analysis_with_metrics.csv'):
    """Load data and extract SINDY coefficients + beta_reward & beta_choice."""
    df = pd.read_csv(csv_path)
    sindy_cols = [c for c in df.columns if c.startswith('x_')]
    for extra in ('beta_reward', 'beta_choice'):
        if extra in df.columns:
            sindy_cols.append(extra)
    return df, sindy_cols


def clean_coefficient_name(col):
    """Clean coefficient names for display."""
    return (col.replace('x_', '').replace('_', ' ').title())[:200]  # allow longer names


def run_pairwise_logistic(presence, is_healthy):
    """Run logistic regression for binary comparison."""
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(is_healthy.reshape(-1, 1)).flatten()

        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=0)
        model.fit(X.reshape(-1, 1), presence)

        # Odds ratio
        or_val = np.exp(model.coef_[0][0])

        # Likelihood ratio test
        p_hat = model.predict_proba(X.reshape(-1, 1))[:, 1]
        eps = 1e-15
        ll = np.sum(presence * np.log(np.clip(p_hat, eps, 1 - eps)) +
                    (1 - presence) * np.log(np.clip(1 - p_hat, eps, 1 - eps)))
        p0 = presence.mean()
        ll0 = np.sum(presence * np.log(p0) + (1 - presence) * np.log(1 - p0))
        lr = -2 * (ll0 - ll)
        p_val = 1 - stats.chi2.cdf(max(0, lr), df=1)

        return or_val, p_val
    except Exception:
        return np.nan, np.nan


def perform_pairwise_diagnosis_analysis(df, cols, output_dir):
    """Perform Healthy vs Bipolar and Healthy vs Depression comparisons."""
    df_clean = df[df['Diagnosis'].notna()].copy()
    unique_diagnoses = df_clean['Diagnosis'].unique()
    print(f"Using {len(df_clean)} participants; Diagnoses: {unique_diagnoses}")

    if 'Healthy' not in unique_diagnoses:
        raise ValueError("'Healthy' group not found in data")

    diagnosis_groups = [d for d in unique_diagnoses if d != 'Healthy']
    print(f"Comparing Healthy vs: {diagnosis_groups}")

    for diag in unique_diagnoses:
        n = sum(df_clean['Diagnosis'] == diag)
        print(f"  {diag}: n={n}")

    results = []
    skipped = []

    for col in cols:
        vals = df_clean[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            skipped.append((col, "<2 obs"))
            continue

        vals_clean = vals[mask]
        diagnosis_clean = df_clean['Diagnosis'].values[mask]
        presence = (vals_clean != 0).astype(int)
        presence_rate = presence.mean()

        if presence_rate == 0:
            skipped.append((col, "all zero"))
            continue
        elif presence_rate == 1.0:
            skipped.append((col, "always present"))
            continue

        result = {
            'coefficient': col,
            'coefficient_clean': clean_coefficient_name(col),
            'n_total': int(len(presence)),
            'presence_rate': presence_rate
        }

        for diag_group in diagnosis_groups:
            pairwise_mask = (diagnosis_clean == 'Healthy') | (diagnosis_clean == diag_group)
            if pairwise_mask.sum() < 2:
                result[f'healthy_vs_{diag_group.lower()}_OR'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_p'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_sig'] = 'insufficient_data'
                continue

            pair_diagnosis = diagnosis_clean[pairwise_mask]
            pair_presence = presence[pairwise_mask]
            is_healthy = (pair_diagnosis == 'Healthy').astype(int)

            n_healthy = sum(is_healthy)
            n_diag = sum(1 - is_healthy)
            result[f'n_healthy_{diag_group.lower()}'] = n_healthy
            result[f'n_{diag_group.lower()}'] = n_diag

            if pair_presence.std() == 0:
                result[f'healthy_vs_{diag_group.lower()}_OR'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_p'] = np.nan
                result[f'healthy_vs_{diag_group.lower()}_sig'] = 'no_variation'
                continue

            or_val, p_val = run_pairwise_logistic(pair_presence, is_healthy)
            result[f'healthy_vs_{diag_group.lower()}_OR'] = or_val
            result[f'healthy_vs_{diag_group.lower()}_p'] = p_val
            result[f'healthy_vs_{diag_group.lower()}_sig'] = p_val  # store p_val

        results.append(result)

    df_results = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'pairwise_analysis_results.csv')
    df_results.to_csv(csv_path, index=False)

    print(f"\nSkipped {len(skipped)} coefficients")
    print(f"Analyzed {len(results)} coefficients")

    for diag_group in diagnosis_groups:
        sig_col = f'healthy_vs_{diag_group.lower()}_sig'
        if sig_col in df_results.columns:
            significant = df_results[df_results[sig_col] < 0.05]
            print(f"\nHealthy vs {diag_group}: {len(significant)} significant")
            for _, row in significant.head(5).iterrows():
                or_val = row[f'healthy_vs_{diag_group.lower()}_OR']
                p_val = row[f'healthy_vs_{diag_group.lower()}_p']
                direction = "↑ healthy" if or_val > 1 else "↓ healthy"
                print(f"  {row['coefficient_clean']}: OR={or_val:.2f}, p={p_val:.4f} ({direction})")

    create_pairwise_plots(df_results, df_clean, output_dir, diagnosis_groups)
    return df_results


def _wrap_text_list(strings, width=20):
    """Wrap a list of strings to given width with newlines."""
    return ["\n".join(textwrap.wrap(s, width)) if isinstance(s, str) else s for s in strings]


def create_pairwise_plots(df_results, df_clean, output_dir, diagnosis_groups):
    """Create odds ratio and presence rate plots with exact p-values, and FDR‐corrected versions.
       Additionally, create a *wrapped-label* vertical-bar variant to avoid cut-off names.
    """
    # 1. Odds ratio plots (raw p-values)
    n_plots = len(diagnosis_groups)
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for i, diag_group in enumerate(diagnosis_groups):
        or_col = f'healthy_vs_{diag_group.lower()}_OR'
        p_col = f'healthy_vs_{diag_group.lower()}_p'

        valid_data = df_results.dropna(subset=[or_col, p_col]).copy()
        if valid_data.empty:
            axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=14)
            continue

        valid_data = valid_data.sort_values(or_col)
        y_pos = np.arange(len(valid_data))

        axes[i].barh(y_pos, valid_data[or_col], color='lightgrey', edgecolor='black')
        axes[i].axvline(x=1, color='black', linestyle='--', alpha=0.7)

        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(valid_data['coefficient_clean'], fontsize=10)
        axes[i].set_xlabel('Odds Ratio (OR > 1: more in healthy)', fontsize=12)
        axes[i].set_title(f'Healthy vs {diag_group}', fontsize=14)
        axes[i].set_xscale('log')
        axes[i].tick_params(axis='x', labelsize=10)

        for j, (_, row) in enumerate(valid_data.iterrows()):
            or_val = row[or_col]
            p_val = row[p_col]
            axes[i].text(or_val, j,
                         f' {or_val:.2f}, p={p_val:.3f}',
                         va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'odds_ratios.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Presence rates for significant coefficients
    sig_coeffs = []
    for _, row in df_results.iterrows():
        for diag_group in diagnosis_groups:
            if row.get(f'healthy_vs_{diag_group.lower()}_p', 1) < 0.05:
                sig_coeffs.append(row)
                break

    if not sig_coeffs:
        sig_coeffs = df_results.nlargest(6, 'presence_rate').to_dict('records')
    else:
        sig_coeffs = sig_coeffs[:6]

    if sig_coeffs:
        presence_data = []
        all_groups = ['Healthy'] + diagnosis_groups

        for coef in sig_coeffs:
            col = coef['coefficient']
            for diag in all_groups:
                data = df_clean[df_clean['Diagnosis'] == diag][col].dropna()
                if len(data) > 0:
                    presence_data.append({
                        'Coefficient': coef['coefficient_clean'],
                        'Diagnosis': diag,
                        'Presence_Rate': (data != 0).mean(),
                        'N': len(data)
                    })

        if presence_data:
            presence_df = pd.DataFrame(presence_data)
            pivot_df = presence_df.pivot(index='Coefficient', columns='Diagnosis', values='Presence_Rate')
            col_order = ['Healthy'] + [d for d in pivot_df.columns if d != 'Healthy']
            pivot_df = pivot_df[col_order]

            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_df.plot(kind='bar', ax=ax, color='lightgrey', edgecolor='black', width=0.8)

            ax.set_ylabel('Presence Rate', fontsize=12)
            ax.set_title('Presence Rates for Significant Coefficients', fontsize=14)
            ax.set_xticklabels(_wrap_text_list(pivot_df.index.tolist(), width=18),
                               rotation=30, ha='right', fontsize=10)
            ax.tick_params(axis='y', labelsize=10)

            legend = ax.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.setp(legend.get_title(), fontsize=10)
            for text in legend.get_texts():
                text.set_fontsize(10)

            ax.set_ylim(0, 1)
            plt.subplots_adjust(bottom=0.30)  # extra room for wrapped labels
            plt.savefig(os.path.join(output_dir, 'presence_rates.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

    # 3. Odds ratio plots with FDR‐corrected p-values (barh as before)
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        fdr_axes = [ax]
    else:
        fig, fdr_axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 8))
        fdr_axes = fdr_axes if isinstance(fdr_axes, (list, np.ndarray)) else [fdr_axes]

    for i, diag_group in enumerate(diagnosis_groups):
        or_col = f'healthy_vs_{diag_group.lower()}_OR'
        p_col = f'healthy_vs_{diag_group.lower()}_p'

        df_plot = df_results.dropna(subset=[or_col, p_col]).copy()
        if df_plot.empty:
            fdr_axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center',
                             transform=fdr_axes[i].transAxes, fontsize=14)
            continue

        # FDR correction
        p_vals = df_plot[p_col].values
        _, p_fdr, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')
        df_plot['p_fdr'] = p_fdr
        df_plot = df_plot.sort_values(or_col)
        y_pos = np.arange(len(df_plot))

        fdr_axes[i].barh(y_pos, df_plot[or_col], color='lightgrey', edgecolor='black')
        fdr_axes[i].axvline(x=1, color='black', linestyle='--', alpha=0.7)

        fdr_axes[i].set_yticks(y_pos)
        fdr_axes[i].set_yticklabels(df_plot['coefficient_clean'], fontsize=10)
        fdr_axes[i].set_xlabel('Odds Ratio (OR > 1: more in healthy)', fontsize=12)
        fdr_axes[i].set_title(f'Healthy vs {diag_group} (FDR)', fontsize=14)
        fdr_axes[i].set_xscale('log')
        fdr_axes[i].tick_params(axis='x', labelsize=10)

        for j, (_, row) in enumerate(df_plot.iterrows()):
            fdr_axes[i].text(row[or_col], j,
                             f' {row[or_col]:.2f}, p_fdr={row["p_fdr"]:.3f}',
                             va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'odds_ratios_fdr_new.png'), dpi=500, bbox_inches='tight')
    plt.close()

    # 4. NEW: Wrapped-label vertical-bar variant (per diagnosis) to guarantee full names
    #    This is the "another plot" you asked for.
    for diag_group in diagnosis_groups:
        or_col = f'healthy_vs_{diag_group.lower()}_OR'
        p_col = f'healthy_vs_{diag_group.lower()}_p'
        df_plot = df_results.dropna(subset=[or_col, p_col]).copy()
        if df_plot.empty:
            continue

        # FDR correction again for consistency
        _, p_fdr, _, _ = multipletests(df_plot[p_col].values, alpha=0.05, method='fdr_bh')
        df_plot['p_fdr'] = p_fdr

        # Sort by effect size magnitude (distance from OR=1) to prioritize bigger effects
        df_plot['effect_mag'] = np.abs(np.log(df_plot[or_col]))
        df_plot = df_plot.sort_values('effect_mag', ascending=False)

        labels = df_plot['coefficient_clean'].tolist()
        wrapped = _wrap_text_list(labels, width=18)
        x = np.arange(len(df_plot))
        or_vals = df_plot[or_col].values

        # Figure width scales with number of labels to reduce crowding
        fig_width = max(12, min(28, 0.40 * len(wrapped)))  # clamp to avoid monster images
        fig, ax = plt.subplots(figsize=(fig_width, 8))
        ax.bar(x, or_vals, color='lightgrey', edgecolor='black')
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Odds Ratio (OR > 1: more in healthy)', fontsize=12)
        ax.set_title(f'Healthy vs {diag_group} (FDR), Wrapped Labels', fontsize=14)
        ax.set_yscale('log')
        ax.tick_params(axis='y', labelsize=10)

        # Annotate a few values to keep plot readable (top 15 by effect)
        for j in range(min(15, len(x))):
            ax.text(x[j], or_vals[j],
                    f'{or_vals[j]:.2f}\nq={df_plot["p_fdr"].values[j]:.3f}',
                    ha='center', va='bottom', fontsize=8)

        # Extra space at bottom for multi-line labels
        plt.subplots_adjust(bottom=0.35)
        out_path = os.path.join(output_dir, f'odds_ratios_fdr_wrapped_{diag_group}.png')
        plt.savefig(out_path, dpi=500, bbox_inches='tight')
        plt.close()

    print(f"Plots saved to {output_dir} (including FDR and wrapped-label variants)")


def main(
    csv_path='dezfouli_final_df_sindy_analysis_with_metrics.csv',
    output_dir='/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis_dezfouli/plots/pairwise_analysis'
):
    """Run pairwise diagnosis analysis."""
    df, cols = load_and_prepare_data(csv_path)
    results = perform_pairwise_diagnosis_analysis(df, cols, output_dir)
    return results


if __name__ == '__main__':
    main()
