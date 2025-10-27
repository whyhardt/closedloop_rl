import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import textwrap
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path='final_df_sindy_analysis_with_metrics.csv'):
    """
    Load data and extract SINDY coefficients + beta_reward & beta_choice for logistic regression.
    """
    df = pd.read_csv(csv_path)
    # Pick up all x_ columns
    sindy_cols = [c for c in df.columns if c.startswith('x_')]
    # Also include these two if present
    for extra in ('beta_reward', 'beta_choice'):
        if extra in df.columns:
            sindy_cols.append(extra)
    return df, sindy_cols


def clean_coefficient_name(col):
    # allow much longer names so we don't truncate content here
    return (col.replace('x_', '').replace('_', ' ').title())[:200]


def _wrap_text_list(strings, width=18):
    """
    Wrap a list of strings to given width with newlines.
    """
    wrapped = []
    for s in strings:
        if not isinstance(s, str):
            wrapped.append(s)
        else:
            w = "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s
            wrapped.append(w)
    return wrapped


def perform_logistic_regression_analysis(df, cols, output_dir):
    """
    Runs logistic regression on each coefficient vs age, prints never/always present,
    then plots only the middle group sorted by |β|, with both raw and FDR‐corrected p‐values.
    Also emits wrapped-label variants to avoid cut-off.
    """
    # 1) filter missing ages
    df_clean = df[df['Age'].notna()].copy()
    if df_clean.empty:
        raise ValueError("No valid ages found.")
    age_min, age_max = df_clean['Age'].min(), df_clean['Age'].max()
    print(f"Using {len(df_clean)} participants; Age range {age_min:.1f}–{age_max:.1f}")

    # 2) standardize
    scaler = StandardScaler()
    age_std = scaler.fit_transform(df_clean[['Age']]).flatten()

    results, skipped = [], []

    for col in cols:
        vals = df_clean[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 10:
            skipped.append((col, f"<10 obs ({mask.sum()})"))
            continue

        y = (vals[mask] != 0).astype(int)
        rate = y.mean()

        if rate == 0:
            skipped.append((col, "all zero"))
            continue
        if rate == 1.0:
            results.append(dict(
                coefficient=col,
                beta_age=np.nan,
                p_value=np.nan,
                n_nonzero=int(y.sum()),
                n_total=int(len(y)),
                coefficient_clean=clean_coefficient_name(col),
                note='always present'
            ))
            continue

        # fit
        solver = 'saga' if rate < 0.1 else 'liblinear'
        its = 2000 if rate < 0.1 else 1000
        model = LogisticRegression(solver=solver, max_iter=its, random_state=0)
        model.fit(age_std[mask].reshape(-1, 1), y)
        beta_age = model.coef_[0][0]

        # LRT
        p_hat = model.predict_proba(age_std[mask].reshape(-1, 1))[:, 1]
        eps = 1e-15
        ll = np.sum(y * np.log(np.clip(p_hat, eps, 1 - eps)) +
                    (1 - y) * np.log(np.clip(1 - p_hat, eps, 1 - eps)))
        p0 = y.mean()
        ll0 = np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0))
        lr = -2 * (ll0 - ll)
        pval = 1 - chi2.cdf(max(0, lr), df=1)

        results.append(dict(
            coefficient=col,
            beta_age=beta_age,
            p_value=pval,
            n_nonzero=int(y.sum()),
            n_total=int(len(y)),
            coefficient_clean=clean_coefficient_name(col)
        ))

    if skipped:
        print(f"Skipped {len(skipped)} coeffs (e.g. {skipped[:3]})")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        raise ValueError("No valid regressions run.")

    # only those with a real regression (i.e., not always-present)
    mask_reg = ~df_res.get('note', '').eq('always present')
    print("\nLogistic regressions (0 < rate < 1):")
    for _, r in df_res[mask_reg].iterrows():
        print(f" - {r['coefficient_clean']}: p={r['p_value']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    all_csv = os.path.join(output_dir, 'sindy_age_logistic_regression_all.csv')
    df_res.to_csv(all_csv, index=False)
    print(f"\nFull results → {all_csv}")

    # never / always
    never = [clean_coefficient_name(c) for c, n in skipped if n == 'all zero']
    always = df_res.loc[df_res.get('note', '') == 'always present', 'coefficient_clean'].tolist()
    print("\nNever-present:\n ", "\n  ".join(never))
    print("\nAlways-present:\n ", "\n  ".join(always))

    # remaining & sort by |β|
    df_rem = df_res[mask_reg].copy()
    df_rem['abs_beta'] = df_rem['beta_age'].abs()
    df_rem.sort_values('abs_beta', ascending=False, inplace=True)
    df_rem.drop('abs_beta', axis=1, inplace=True)

    rem_csv = os.path.join(output_dir, 'sindy_age_logistic_regression_remaining.csv')
    df_rem.to_csv(rem_csv, index=False)
    print(f"\nRemaining results → {rem_csv}")

    # plot remaining
    if df_rem.empty:
        print("No remaining to plot.")
    else:
        print(f"\nPlotting {len(df_rem)} remaining (sorted by |β|)…")
        create_beta_bar_plot(df_rem, output_dir)
        create_beta_bar_plot_wrapped(df_rem, output_dir)          # NEW
        create_logistic_regression_plot(df_rem, output_dir, age_min, age_max)
        create_beta_bar_plot_fdr(df_rem, output_dir)
        create_beta_bar_plot_fdr_wrapped(df_rem, output_dir)      # NEW
        create_logistic_regression_plot_fdr(df_rem, output_dir, age_min, age_max)

    return df_res


def _dynamic_fig_width(n_labels, base=10, per_label=0.35, max_w=30):
    return max(base, min(max_w, base + per_label * max(0, n_labels - 12)))


def create_beta_bar_plot(df, output_dir):
    """
    Bar-plot of β sorted by magnitude, all bars in light grey,
    annotated with exact p-values.
    """
    fig_width = _dynamic_fig_width(len(df))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(df['coefficient_clean'], df['beta_age'],
                  color='lightgrey', edgecolor='black')
    ax.axhline(0, linestyle='--', color='black')
    ax.set_ylabel('Age Effect (β)')
    ax.set_title('Age Effect (β) by Coefficient (raw p-values)')
    ax.tick_params(axis='x', labelsize=9)
    plt.xticks(rotation=30, ha='right')

    # annotate p-values above each bar
    for bar, p in zip(bars, df['p_value']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + np.sign(height) * 0.02,
                f"p={p:.3f}",
                ha='center', va='bottom', fontsize=8)

    # more room for rotated labels
    plt.subplots_adjust(bottom=0.28)
    out = os.path.join(output_dir, 'beta_vs_coefficient_remaining_sorted.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • β-bar (raw) → {out}")


def create_beta_bar_plot_wrapped(df, output_dir):
    """
    Same as create_beta_bar_plot but with wrapped, multi-line x tick labels to avoid truncation.
    """
    labels = df['coefficient_clean'].tolist()
    wrapped = _wrap_text_list(labels, width=18)

    fig_width = _dynamic_fig_width(len(wrapped))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.arange(len(df))
    bars = ax.bar(x, df['beta_age'].values, color='lightgrey', edgecolor='black')
    ax.axhline(0, linestyle='--', color='black')
    ax.set_ylabel('Age Effect (β)')
    ax.set_title('Age Effect (β) by Coefficient (raw p-values, wrapped labels)')
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped, rotation=15, ha='right', fontsize=9)

    for bar, p in zip(bars, df['p_value']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + np.sign(height) * 0.02,
                f"p={p:.3f}",
                ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(bottom=0.35)  # extra space for multi-line labels
    out = os.path.join(output_dir, 'beta_vs_coefficient_remaining_sorted_wrapped.png')
    plt.savefig(out, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"  • β-bar (raw, wrapped) → {out}")


def create_logistic_regression_plot(df, output_dir, age_min, age_max):
    """
    Curve-plot for the remaining sorted coefficients, all curves in light grey,
    subplot titles annotated with exact p-values.
    """
    ages = np.linspace(age_min, age_max, 200)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Age-Dependent Presence (raw p-values)', y=0.98)

    for ax, (_, row) in zip(axes.flatten(), df.iterrows()):
        beta_age = row['beta_age']
        # visual-only standardization over the plotted grid
        p = 1 / (1 + np.exp(-beta_age * ((ages - ages.mean()) / ages.std())))
        title = f"{row['coefficient_clean']} (p={row['p_value']:.3f})"
        # wrap long titles
        ax.set_title("\n".join(textwrap.wrap(title, width=45)), fontsize=10)
        ax.plot(ages, p, color='lightgrey')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Age')
        ax.set_ylabel('Prob.')

    plt.tight_layout()
    out = os.path.join(output_dir, 'logistic_curves_remaining_sorted.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • curves (raw) → {out}")


def create_beta_bar_plot_fdr(df, output_dir):
    """
    Bar-plot of β sorted by magnitude, annotated with FDR-corrected p-values.
    """
    # apply FDR correction
    p_raw = df['p_value'].values
    _, p_fdr, _, _ = multipletests(p_raw, alpha=0.05, method='fdr_bh')
    df2 = df.copy()
    df2['p_fdr'] = p_fdr

    fig_width = _dynamic_fig_width(len(df2))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    bars = ax.bar(df2['coefficient_clean'], df2['beta_age'],
                  color='lightgrey', edgecolor='black')
    ax.axhline(0, linestyle='--', color='black')
    ax.set_ylabel('Age Effect (β)')
    ax.set_title('Age Effect (β) by Coefficient (FDR p-values)')
    ax.tick_params(axis='x', labelsize=9)
    plt.xticks(rotation=30, ha='right')

    for bar, p in zip(bars, df2['p_fdr']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + np.sign(height) * 0.02,
                f"p={p:.3f}",
                ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(bottom=0.28)
    out = os.path.join(output_dir, 'new_beta_vs_coefficient_remaining_sorted_fdr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • β-bar (FDR) → {out}")


def create_beta_bar_plot_fdr_wrapped(df, output_dir):
    """
    FDR version with wrapped, multi-line x tick labels to avoid truncation.
    """
    p_raw = df['p_value'].values
    _, p_fdr, _, _ = multipletests(p_raw, alpha=0.05, method='fdr_bh')
    df2 = df.copy()
    df2['p_fdr'] = p_fdr

    labels = df2['coefficient_clean'].tolist()
    wrapped = _wrap_text_list(labels, width=18)
    x = np.arange(len(df2))

    fig_width = _dynamic_fig_width(len(wrapped))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(x, df2['beta_age'].values, color='lightgrey', edgecolor='black')
    ax.axhline(0, linestyle='--', color='black')
    ax.set_ylabel('Age Effect (β)')
    ax.set_title('Age Effect (β) by Coefficient (FDR p-values, wrapped labels)')
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped, rotation=45, ha='right', fontsize=9)

    for bar, p in zip(bars, df2['p_fdr']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + np.sign(height) * 0.02,
                f"p={p:.3f}",
                ha='center', va='bottom', fontsize=8)

    plt.subplots_adjust(bottom=0.35)
    out = os.path.join(output_dir, 'new_beta_vs_coefficient_remaining_sorted_fdr_wrapped.png')
    plt.savefig(out, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"  • β-bar (FDR, wrapped) → {out}")


def create_logistic_regression_plot_fdr(df, output_dir, age_min, age_max):
    """
    Curve-plot for the remaining sorted coefficients, annotated with FDR-corrected p-values.
    """
    ages = np.linspace(age_min, age_max, 200)
    # apply FDR correction
    p_raw = df['p_value'].values
    _, p_fdr, _, _ = multipletests(p_raw, alpha=0.05, method='fdr_bh')
    df2 = df.copy()
    df2['p_fdr'] = p_fdr

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Age-Dependent Presence (FDR p-values)', y=0.98)

    for ax, (_, row) in zip(axes.flatten(), df2.iterrows()):
        beta_age = row['beta_age']
        p = 1 / (1 + np.exp(-beta_age * ((ages - ages.mean()) / ages.std())))
        title = f"{row['coefficient_clean']} (q={row['p_fdr']:.3f})"
        ax.set_title("\n".join(textwrap.wrap(title, width=45)), fontsize=10)
        ax.plot(ages, p, color='lightgrey')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Age')
        ax.set_ylabel('Prob.')

    plt.tight_layout()
    out = os.path.join(output_dir, 'logistic_curves_remaining_sorted_fdr.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  • curves (FDR) → {out}")


def main(
    csv_path='final_df_sindy_analysis_with_metrics.csv',
    output_dir='/Users/martynaplomecka/closedloop_rl/analysis/participants_analysis/plots/logistic_regression'
):
    df, cols = load_and_prepare_data(csv_path)
    perform_logistic_regression_analysis(df, cols, output_dir)


if __name__ == '__main__':
    main()