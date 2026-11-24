#!/usr/bin/env python
"""
Collect and analyze steady-state segmentation results from mKate experiments.

Folder structure:
    <input>/{gene}/{gene}_ss_{temp}_{replicate}/{field}_out/cell_measurements.csv

Examples:
    with_mkate/33C/flu/flu_ss_33C_1/01_out/cell_measurements.csv
    with_mkate/33C/gadB/gadB_ss_33C_3/12_out/cell_measurements.csv

Usage:
    # All genes at one temperature:
    python collect_results_mkate_ss.py --input with_mkate/33C --output with_mkate/33C_analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse


# =============================================================================
# Data loading
# =============================================================================

def parse_condition_folder(folder_name):
    """
    Parse e.g. 'flu_ss_33C_2' -> (gene='flu', temp='33C', replicate=2).
    Returns None if not matched.
    """
    m = re.fullmatch(r'(.+?)_ss_([^_]+)_(\d+)', folder_name)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


def collect_measurements(input_dir):
    input_path = Path(input_dir)
    all_data = []

    for csv_path in sorted(input_path.rglob('*_out/cell_measurements.csv')):
        condition_dir = csv_path.parent.parent
        parsed = parse_condition_folder(condition_dir.name)
        if parsed is None:
            print(f"  Skipping (unrecognised): {condition_dir.name}")
            continue

        gene, temp, replicate = parsed
        field_str = csv_path.parent.name  # e.g. "01_out"
        field = int(re.search(r'\d+', field_str).group())

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Skipping {csv_path}: {e}")
            continue
        if df.empty:
            continue

        df.insert(0, 'gene', gene)
        df.insert(1, 'temp', temp)
        df.insert(2, 'replicate', replicate)
        df.insert(3, 'field', field)
        all_data.append(df)
        print(f"  {gene} {temp} rep={replicate} field={field:>2d}: {len(df)} cells")

    if not all_data:
        raise ValueError(f"No data found under {input_dir}")

    combined = pd.concat(all_data, ignore_index=True)
    genes = sorted(combined['gene'].unique())
    print(f"\nTotal: {len(combined)} cells, {len(genes)} gene(s): {genes}")
    return combined


def filter_saturated(df):
    sat_cols = [c for c in df.columns if 'saturated' in c.lower()]
    if not sat_cols:
        return df
    mask = pd.Series(True, index=df.index)
    for col in sat_cols:
        mask &= (df[col] == 0)
    n = (~mask).sum()
    if n:
        print(f"  Removed {n} saturated cells ({100*n/len(df):.1f}%)")
    return df[mask].copy()


def compute_stats(df, value_cols):
    """Per-(gene, replicate) stats."""
    rows = []
    for (gene, rep), grp in df.groupby(['gene', 'replicate']):
        row = {'gene': gene, 'replicate': rep, 'n_cells': len(grp)}
        for col in value_cols:
            vals = grp[col].dropna()
            row[f'{col}_median'] = np.median(vals) if len(vals) else np.nan
            row[f'{col}_p25']    = np.percentile(vals, 25) if len(vals) else np.nan
            row[f'{col}_p75']    = np.percentile(vals, 75) if len(vals) else np.nan
            row[f'{col}_mean']   = vals.mean() if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['gene', 'replicate']).reset_index(drop=True)


# =============================================================================
# Colors
# =============================================================================

GENE_COLORS   = plt.cm.tab10.colors
REP_COLORS    = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
FIELD_COLORS  = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                 '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
                 '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#b3b3b3']


def _color(val, vals, palette):
    return palette[list(vals).index(val) % len(palette)]


# =============================================================================
# Plots
# =============================================================================

def plot_ridge(df, value_col, output_path, n_bins=40, title_cols=None, log_scale=False,
               per_field=True):
    """
    Ridge plot: one column per gene, one row per replicate.
    If per_field=True, overlaid per-field histograms colored by field.
    If per_field=False, one combined histogram per replicate (colored by replicate).
    """
    genes      = sorted(df['gene'].unique())
    replicates = sorted(df['replicate'].unique())
    col_label  = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    if log_scale:
        col_label = f'log({col_label})'
    n_genes = len(genes)

    fig, axes = plt.subplots(1, n_genes,
                             figsize=(8 * n_genes, max(6, 3 + 1.2 * len(replicates))),
                             sharey=True)
    if n_genes == 1:
        axes = [axes]

    for ax, gene in zip(axes, genes):
        gdf = df[df['gene'] == gene]
        all_vals = gdf[value_col].dropna()
        if log_scale:
            all_vals = np.log10(all_vals[all_vals > 0])
        vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        bin_edges  = np.linspace(vmin, vmax, n_bins + 1)

        # spacing from max fraction across groups
        max_h = 0
        for rep in replicates:
            groups = [(gdf[(gdf['replicate'] == rep) & (gdf['field'] == fld)][value_col].dropna(),)
                      for fld in sorted(gdf[gdf['replicate'] == rep]['field'].unique())] \
                     if per_field else \
                     [(gdf[gdf['replicate'] == rep][value_col].dropna(),)]
            for (vals,) in groups:
                if log_scale:
                    vals = np.log10(vals[vals > 0])
                if len(vals):
                    counts, _ = np.histogram(vals, bins=bin_edges)
                    max_h = max(max_h, (counts / counts.sum()).max())

        spacing = max_h * 0.9 if max_h > 0 else 1
        n_rows   = len(replicates)

        for i, rep in enumerate(replicates):
            y_base = (n_rows - 1 - i) * spacing
            ax.axhline(y_base, color='black', linewidth=0.3, alpha=0.4)

            if per_field:
                fields = sorted(gdf[gdf['replicate'] == rep]['field'].unique())
                for fld in fields:
                    color = _color(fld, fields, FIELD_COLORS)
                    vals  = gdf[(gdf['replicate'] == rep) & (gdf['field'] == fld)][value_col].dropna()
                    if log_scale:
                        vals = np.log10(vals[vals > 0])
                    if not len(vals):
                        continue
                    counts, _ = np.histogram(vals, bins=bin_edges)
                    fracs = counts / counts.sum()
                    for j in range(n_bins):
                        ax.bar(bin_edges[j], fracs[j], width=bin_edges[j+1] - bin_edges[j],
                               bottom=y_base, align='edge', color=color,
                               edgecolor='none', alpha=0.5)
            else:
                color = _color(rep, replicates, REP_COLORS)
                vals  = gdf[gdf['replicate'] == rep][value_col].dropna()
                if log_scale:
                    vals = np.log10(vals[vals > 0])
                if len(vals):
                    counts, _ = np.histogram(vals, bins=bin_edges)
                    fracs = counts / counts.sum()
                    for j in range(n_bins):
                        ax.bar(bin_edges[j], fracs[j], width=bin_edges[j+1] - bin_edges[j],
                               bottom=y_base, align='edge', color=color,
                               edgecolor='none', alpha=0.8)

        ax.set_yticks([(n_rows - 1 - i) * spacing for i in range(n_rows)])
        ax.set_yticklabels([f'rep {r}' for r in replicates], fontsize=12)
        ax.set_xlabel(col_label, fontsize=11)
        ax.set_title(gene, fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'{col_label} distribution', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_comparison(df, stats_df, value_col, output_path, title_cols=None):
    """
    Strip chart: x = gene, one dot per replicate (colored by replicate),
    individual cells as background scatter, median bar per replicate.
    """
    genes      = sorted(df['gene'].unique())
    replicates = sorted(df['replicate'].unique())
    col_label  = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())

    n_reps  = len(replicates)
    width   = 0.6 / n_reps
    x_ticks = range(len(genes))

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(genes)), 5))

    for gi, gene in enumerate(genes):
        for ri, rep in enumerate(replicates):
            color  = _color(rep, replicates, REP_COLORS)
            offset = (ri - (n_reps - 1) / 2) * width
            x      = gi + offset

            vals = df[(df['gene'] == gene) & (df['replicate'] == rep)][value_col].dropna()
            if not len(vals):
                continue
            jitter = np.random.uniform(-width * 0.3, width * 0.3, len(vals))
            ax.scatter(x + jitter, vals, color=color, alpha=0.15, s=3, edgecolors='none')

            med = np.median(vals)
            ax.plot([x - width * 0.4, x + width * 0.4], [med, med],
                    color=color, linewidth=2.5, zorder=5)
            ax.scatter([x], [med], color=color, s=50, zorder=6,
                       edgecolors='black', linewidths=0.5,
                       label=f'rep {rep}' if gi == 0 else '')

    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels(genes, fontsize=12)
    ax.set_ylabel(col_label, fontsize=12)
    ax.set_yscale('log')
    ax.set_title(col_label, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# =============================================================================
# QC plots
# =============================================================================

def plot_qc_histograms(df, value_col, qc_dir, title_cols=None, n_bins=40):
    """Per-gene figure: overlaid per-field histograms, one panel per replicate."""
    col_label  = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    genes      = sorted(df['gene'].unique())
    replicates = sorted(df['replicate'].unique())

    for gene in genes:
        gdf    = df[df['gene'] == gene]
        n_reps = len(replicates)
        fig, axes = plt.subplots(1, n_reps, figsize=(5 * n_reps, 4), sharey=True)
        if n_reps == 1:
            axes = [axes]

        all_vals = gdf[value_col].dropna()
        if len(all_vals) < 2:
            plt.close(fig)
            continue
        vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        bins = np.linspace(vmin, vmax, n_bins + 1)

        for ax, rep in zip(axes, replicates):
            rdf    = gdf[gdf['replicate'] == rep]
            fields = sorted(rdf['field'].unique())
            for fld in fields:
                vals = rdf[rdf['field'] == fld][value_col].dropna()
                if not len(vals):
                    continue
                color = _color(fld, fields, FIELD_COLORS)
                ax.hist(vals, bins=bins, histtype='step', linewidth=1.8,
                        color=color, label=f'field {fld} (n={len(vals)})', density=True)
            ax.set_title(f'rep {rep}', fontsize=11)
            ax.set_xlabel(col_label, fontsize=10)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel('Density', fontsize=11)
        fig.suptitle(f'{col_label} per field — {gene}', fontsize=13)
        plt.tight_layout()
        fname = f'{value_col}_hist.png'
        plt.savefig(qc_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fname}")


def plot_qc_centroid_scatter(df, value_col, qc_dir, title_cols=None):
    """
    Per-gene figure: 2 rows (centroid_x, centroid_y) × n_fields columns.
    One figure per gene × replicate.
    """
    col_label  = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    genes      = sorted(df['gene'].unique())
    replicates = sorted(df['replicate'].unique())

    for gene in genes:
        for rep in replicates:
            gdf    = df[(df['gene'] == gene) & (df['replicate'] == rep)]
            fields = sorted(gdf['field'].unique())
            if not fields:
                continue

            n_fields = len(fields)
            fig, axes = plt.subplots(2, n_fields,
                                     figsize=(3.5 * n_fields, 7),
                                     squeeze=False, sharey=True)

            for fi, fld in enumerate(fields):
                fdf   = gdf[gdf['field'] == fld]
                color = _color(fld, fields, FIELD_COLORS)

                for row, coord in enumerate(['centroid_x', 'centroid_y']):
                    ax     = axes[row][fi]
                    vals   = fdf[value_col].values
                    coords = fdf[coord].values
                    ax.scatter(coords, vals, s=3, alpha=0.3, color=color, edgecolors='none')

                    mask = np.isfinite(vals) & np.isfinite(coords)
                    if mask.sum() > 5:
                        z  = np.polyfit(coords[mask], vals[mask], 1)
                        xs = np.linspace(coords[mask].min(), coords[mask].max(), 50)
                        ax.plot(xs, np.polyval(z, xs), color='k', linewidth=1.5,
                                linestyle='--', alpha=0.8)
                        r = np.corrcoef(coords[mask], vals[mask])[0, 1]
                        ax.set_title(f'field {fld}  r={r:.2f}', fontsize=9)
                    else:
                        ax.set_title(f'field {fld}', fontsize=9)

                    if fi == 0:
                        ax.set_ylabel(col_label, fontsize=9)
                    ax.set_xlabel(coord.replace('_', ' '), fontsize=9)
                    ax.grid(True, alpha=0.3)

            fig.suptitle(f'{col_label} vs position — {gene} rep {rep}', fontsize=12)
            plt.tight_layout()
            fname = f'{value_col}_centroid_rep{rep}.png'
            plt.savefig(qc_dir / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect steady-state mKate segmentation results')
    parser.add_argument('--input', '-i', required=True,
                        help='Directory containing gene subfolders (e.g. with_mkate/33C)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: <input>_analysis)')
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else \
                  input_path.parent / (input_path.name + '_analysis')
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_path}\n")

    print("Collecting measurements...")
    df = collect_measurements(args.input)

    print("\nFiltering saturated cells...")
    df = filter_saturated(df)

    df.to_csv(output_path / 'all_cells.csv', index=False)
    print(f"  Saved: all_cells.csv ({len(df)} cells)")

    candidate_cols = ['mcherry_rate', 'gfp_rate', 'gfp_mcherry_ratio']
    value_cols = [c for c in candidate_cols if c in df.columns]
    title_cols = {
        'mcherry_rate':      'mCherry rate (a.u./ms)',
        'gfp_rate':          'GFP rate (a.u./ms)',
        'gfp_mcherry_ratio': 'GFP/mCherry ratio',
    }
    print(f"\nValue columns: {value_cols}")

    print("\nComputing statistics...")
    stats_df = compute_stats(df, value_cols)
    stats_df.to_csv(output_path / 'stats.csv', index=False)
    print(f"  Saved: stats.csv")

    genes = sorted(df['gene'].unique())

    # Per-gene plots
    for gene in genes:
        gene_dir = output_path / gene
        gene_dir.mkdir(exist_ok=True)
        qc_dir = gene_dir / 'QC'
        qc_dir.mkdir(exist_ok=True)
        gdf       = df[df['gene'] == gene]
        gstats_df = stats_df[stats_df['gene'] == gene]
        print(f"\n  [{gene}]")

        for col in value_cols:
            plot_ridge(gdf, col, gene_dir / f'{col}_ridge.png', title_cols=title_cols)
            plot_ridge(gdf, col, gene_dir / f'{col}_ridge_combined.png',
                       title_cols=title_cols, per_field=False)
            if col == 'gfp_mcherry_ratio':
                plot_ridge(gdf, col, gene_dir / f'{col}_ridge_log.png',
                           title_cols=title_cols, log_scale=True)
                plot_ridge(gdf, col, gene_dir / f'{col}_ridge_log_combined.png',
                           title_cols=title_cols, log_scale=True, per_field=False)
            plot_comparison(gdf, gstats_df, col,
                            gene_dir / f'{col}_comparison.png',
                            title_cols=title_cols)
            plot_qc_histograms(gdf, col, qc_dir, title_cols=title_cols)
            if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
                plot_qc_centroid_scatter(gdf, col, qc_dir, title_cols=title_cols)

    # Multi-gene comparison at top level (only when >1 gene)
    if len(genes) > 1:
        print("\n  [all genes]")
        for col in value_cols:
            plot_comparison(df, stats_df, col,
                            output_path / f'{col}_comparison_all.png',
                            title_cols=title_cols)

    print("\nDone!")


if __name__ == '__main__':
    main()
