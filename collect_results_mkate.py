#!/usr/bin/env python
"""
Collect and analyze segmentation results from mKate (3-channel) T-shift experiments.

Folder structure:
    <input>/{gene}_T{time}_{replicate}/{field}_out/cell_measurements.csv

Examples:
    with_mkate/gabB/gadB_T0_1/1_out/cell_measurements.csv
    with_mkate/gabB/gadB_T2.5_3/4_out/cell_measurements.csv

Usage:
    # Single gene:
    python collect_results_mkate.py --input with_mkate/gabB --output with_mkate/gabB_analysis

    # All genes together:
    python collect_results_mkate.py --input with_mkate --output with_mkate/analysis
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
    Parse a condition folder name like 'gadB_T0_1' or 'flu_T2.5_3'.

    Returns (gene, timepoint_hr, replicate) or None if not matched.
    """
    m = re.fullmatch(r'(.+?)_T(\d+(?:\.\d+)?)_(\d+)', folder_name)
    if not m:
        return None
    gene = m.group(1)
    timepoint = float(m.group(2))
    replicate = int(m.group(3))
    return gene, timepoint, replicate


def find_measurement_files(input_dir):
    """
    Recursively find all cell_measurements.csv files under input_dir.
    Returns list of (csv_path, gene, timepoint_hr, replicate, field).
    """
    input_path = Path(input_dir)
    results = []

    for csv_path in sorted(input_path.rglob('*_out/cell_measurements.csv')):
        field_out = csv_path.parent.name          # e.g. "3_out"
        condition_dir = csv_path.parent.parent    # e.g. gadB_T0_1/

        parsed = parse_condition_folder(condition_dir.name)
        if parsed is None:
            print(f"  Skipping (unrecognised folder): {csv_path.parent.parent.name}")
            continue

        gene, timepoint, replicate = parsed
        field = int(re.sub(r'[^\d]', '', field_out)) if re.search(r'\d', field_out) else 0
        results.append((csv_path, gene, timepoint, replicate, field))

    return results


def collect_measurements(input_dir):
    """
    Load all cell_measurements.csv files and return a combined DataFrame with
    columns: gene, timepoint_hr, replicate, field, + all original columns.
    """
    files = find_measurement_files(input_dir)
    if not files:
        raise ValueError(f"No cell_measurements.csv files found under {input_dir}")

    all_data = []
    for csv_path, gene, timepoint, replicate, field in files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Skipping {csv_path}: {e}")
            continue
        if df.empty:
            continue
        df.insert(0, 'gene', gene)
        df.insert(1, 'timepoint_hr', timepoint)
        df.insert(2, 'replicate', replicate)
        df.insert(3, 'field', field)
        all_data.append(df)
        print(f"  {gene} T={timepoint}h rep={replicate} field={field}: {len(df)} cells")

    if not all_data:
        raise ValueError("No data loaded.")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(combined)} cells, {combined['gene'].nunique()} gene(s), "
          f"{sorted(combined['timepoint_hr'].unique())} hr timepoints")
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


# =============================================================================
# Per-group stats
# =============================================================================

def compute_stats(df, value_cols, group_cols=('gene', 'timepoint_hr')):
    """Compute per-group median, IQR, mean, SEM, n_cells."""
    rows = []
    for keys, grp in df.groupby(list(group_cols)):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row['n_cells'] = len(grp)
        for col in value_cols:
            vals = grp[col].dropna()
            row[f'{col}_median'] = np.median(vals) if len(vals) else np.nan
            row[f'{col}_p25']    = np.percentile(vals, 25) if len(vals) else np.nan
            row[f'{col}_p75']    = np.percentile(vals, 75) if len(vals) else np.nan
            row[f'{col}_mean']   = vals.mean() if len(vals) else np.nan
            row[f'{col}_sem']    = vals.sem() if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================

GENE_COLORS = plt.cm.tab10.colors
REPLICATE_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']


def _gene_color(gene, genes):
    return GENE_COLORS[list(genes).index(gene) % len(GENE_COLORS)]


def _rep_color(replicate, replicates):
    return REPLICATE_COLORS[list(replicates).index(replicate) % len(REPLICATE_COLORS)]


def plot_time_evolution(df, stats_df, value_col, output_path, title_cols=None):
    """
    One panel per gene: scatter + median line per replicate, each replicate a distinct color.
    """
    genes = sorted(df['gene'].unique())
    replicates = sorted(df['replicate'].unique())
    n = len(genes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())

    for ax, gene in zip(axes, genes):
        gdf = df[df['gene'] == gene]
        timepoints = sorted(gdf['timepoint_hr'].unique())

        for rep in replicates:
            color = _rep_color(rep, replicates)
            rdf = gdf[gdf['replicate'] == rep]

            # scatter individual cells, jittered
            jitter = np.random.uniform(-0.05, 0.05, len(rdf))
            ax.scatter(rdf['timepoint_hr'] + jitter, rdf[value_col],
                       color=color, alpha=0.15, s=3, edgecolors='none')

            # per-replicate median line
            rep_med = rdf.groupby('timepoint_hr')[value_col].median().reindex(timepoints)
            ax.plot(timepoints, rep_med.values, color=color, linewidth=2,
                    marker='o', markersize=4, label=f'rep {rep}')

        ax.set_title(gene, fontsize=13)
        ax.set_xlabel('Time (hr)', fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(timepoints)
        ax.legend(fontsize=8, loc='upper left')

    axes[0].set_ylabel(col_label, fontsize=12)
    fig.suptitle(col_label, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_comparison(df, stats_df, value_col, output_path, title_cols=None):
    """
    Single panel: all genes overlaid, median ± IQR, x = timepoint.
    """
    genes = sorted(df['gene'].unique())
    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())

    fig, ax = plt.subplots(figsize=(8, 5))
    for gene in genes:
        color = _gene_color(gene, genes)
        st = stats_df[stats_df['gene'] == gene].sort_values('timepoint_hr')
        ax.plot(st['timepoint_hr'], st[f'{value_col}_median'],
                color=color, linewidth=2, marker='o', markersize=5, label=gene)
        ax.fill_between(st['timepoint_hr'], st[f'{value_col}_p25'], st[f'{value_col}_p75'],
                        color=color, alpha=0.2)

    ax.set_xlabel('Time (hr)', fontsize=12)
    ax.set_ylabel(col_label, fontsize=12)
    ax.set_title(col_label, fontsize=13)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df['timepoint_hr'].unique()))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_ridge(df, value_col, output_path, n_bins=40, title_cols=None, log_scale=False):
    """
    Ridge plot per gene: one row per timepoint, stacked vertically.
    If multiple genes, one column per gene.
    If log_scale=True, bins and x-axis are in log10 space.
    """
    genes = sorted(df['gene'].unique())
    timepoints = sorted(df['timepoint_hr'].unique())
    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    if log_scale:
        col_label = f'log({col_label})'
    n_genes = len(genes)

    fig, axes = plt.subplots(1, n_genes, figsize=(8 * n_genes, max(8, 4 + 0.6 * len(timepoints))),
                             sharey=True)
    if n_genes == 1:
        axes = [axes]

    replicates = sorted(df['replicate'].unique())

    for ax, gene in zip(axes, genes):
        gdf = df[df['gene'] == gene]
        all_vals = gdf[value_col].dropna()
        if log_scale:
            all_vals = np.log10(all_vals[all_vals > 0])
        vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
        bin_edges = np.linspace(vmin, vmax, n_bins + 1)

        # Compute max fraction across all timepoints/replicates for spacing
        max_h = 0
        for tp in timepoints:
            for rep in replicates:
                vals = gdf[(gdf['timepoint_hr'] == tp) & (gdf['replicate'] == rep)][value_col].dropna()
                if log_scale:
                    vals = np.log10(vals[vals > 0])
                if len(vals):
                    counts, _ = np.histogram(vals, bins=bin_edges)
                    fracs = counts / counts.sum()
                    max_h = max(max_h, fracs.max())

        spacing = max_h * 0.9 if max_h > 0 else 1
        n_rows = len(timepoints)

        for i, tp in enumerate(timepoints):
            y_base = (n_rows - 1 - i) * spacing
            ax.axhline(y_base, color='black', linewidth=0.3, alpha=0.4)
            for rep in replicates:
                color = _rep_color(rep, replicates)
                vals = gdf[(gdf['timepoint_hr'] == tp) & (gdf['replicate'] == rep)][value_col].dropna()
                if log_scale:
                    vals = np.log10(vals[vals > 0])
                if not len(vals):
                    continue
                counts, _ = np.histogram(vals, bins=bin_edges)
                fracs = counts / counts.sum()
                for j in range(n_bins):
                    ax.bar(bin_edges[j], fracs[j], width=bin_edges[j+1] - bin_edges[j],
                           bottom=y_base, align='edge', color=color, edgecolor='none', alpha=0.5)

        ax.set_yticks([(n_rows - 1 - i) * spacing for i in range(n_rows)])
        ax.set_yticklabels([f'T={tp}h' for tp in timepoints], fontsize=13)
        ax.set_xlabel(col_label, fontsize=11)
        ax.set_title(gene, fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Replicate legend on last axis
    from matplotlib.patches import Patch
    handles = [Patch(color=_rep_color(r, replicates), alpha=0.7, label=f'rep {r}') for r in replicates]
    axes[-1].legend(handles=handles, fontsize=8, loc='upper right')

    fig.suptitle(f'{col_label} distribution', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_stripplot(df, value_col, output_path, title_cols=None):
    """
    Strip chart: x = timepoint, one strip per gene (jittered), with per-replicate medians.
    """
    genes = sorted(df['gene'].unique())
    timepoints = sorted(df['timepoint_hr'].unique())
    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())

    n_genes = len(genes)
    width = 0.7 / n_genes
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(timepoints)), 5))

    for gi, gene in enumerate(genes):
        color = _gene_color(gene, genes)
        offset = (gi - (n_genes - 1) / 2) * width
        gdf = df[df['gene'] == gene]

        for tp in timepoints:
            tdf = gdf[gdf['timepoint_hr'] == tp]
            x = tp + offset
            vals = tdf[value_col].dropna()
            jitter = np.random.uniform(-width * 0.3, width * 0.3, len(vals))
            ax.scatter(x + jitter, vals, color=color, alpha=0.2, s=3, edgecolors='none')

            # Per-replicate medians
            rep_meds = tdf.groupby('replicate')[value_col].median()
            ax.scatter([x] * len(rep_meds), rep_meds, color=color, s=40,
                       zorder=5, edgecolors='black', linewidths=0.5)
            # Overall median bar
            ax.plot([x - width * 0.4, x + width * 0.4],
                    [np.median(vals)] * 2, color=color, linewidth=2.5, zorder=6)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=_gene_color(g, genes), linewidth=2, label=g) for g in genes]
    ax.legend(handles=handles, fontsize=9, loc='upper left')

    ax.set_xticks(timepoints)
    ax.set_xticklabels([f'T={t}h' for t in timepoints])
    ax.set_ylabel(col_label, fontsize=12)
    ax.set_yscale('log')
    ax.set_title(col_label, fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# =============================================================================
# QC: position-dependent checks
# =============================================================================

FIELD_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62']


def _field_color(field, fields):
    return FIELD_COLORS[list(fields).index(field) % len(FIELD_COLORS)]


def plot_qc_histograms(df, value_col, qc_dir, title_cols=None, n_bins=40):
    """
    For each gene × timepoint, plot overlaid per-field histograms of value_col.
    One figure per timepoint, one panel (column) per gene.
    """
    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    genes = sorted(df['gene'].unique())
    timepoints = sorted(df['timepoint_hr'].unique())

    for tp in timepoints:
        tdf = df[df['timepoint_hr'] == tp]
        n_genes = len(genes)
        fig, axes = plt.subplots(1, n_genes, figsize=(6 * n_genes, 4), sharey=True)
        if n_genes == 1:
            axes = [axes]

        for ax, gene in zip(axes, genes):
            gdf = tdf[tdf['gene'] == gene]
            fields = sorted(gdf['field'].unique())
            all_vals = gdf[value_col].dropna()
            if len(all_vals) < 2:
                continue
            vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
            bins = np.linspace(vmin, vmax, n_bins + 1)

            for fld in fields:
                vals = gdf[gdf['field'] == fld][value_col].dropna()
                if len(vals) == 0:
                    continue
                color = _field_color(fld, fields)
                ax.hist(vals, bins=bins, histtype='step', linewidth=1.8,
                        color=color, label=f'field {fld} (n={len(vals)})',
                        density=True)

            ax.set_title(f'{gene}', fontsize=12)
            ax.set_xlabel(col_label, fontsize=10)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel('Density', fontsize=11)
        fig.suptitle(f'{col_label} per field — T={tp}h', fontsize=13)
        plt.tight_layout()
        fname = f'{value_col}_hist_T{tp}h.png'
        plt.savefig(qc_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {fname}")


def plot_qc_centroid_scatter(df, value_col, qc_dir, title_cols=None):
    """
    For each gene × timepoint, subplots per field showing value_col vs
    centroid_x (top row) and centroid_y (bottom row).
    """
    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    genes = sorted(df['gene'].unique())
    timepoints = sorted(df['timepoint_hr'].unique())

    for tp in timepoints:
        tdf = df[df['timepoint_hr'] == tp]

        for gene in genes:
            gdf = tdf[tdf['gene'] == gene]
            fields = sorted(gdf['field'].unique())
            n_fields = len(fields)
            if n_fields == 0:
                continue

            fig, axes = plt.subplots(2, n_fields,
                                     figsize=(4 * n_fields, 7),
                                     squeeze=False, sharey=True)

            for fi, fld in enumerate(fields):
                fdf = gdf[gdf['field'] == fld]
                color = _field_color(fld, fields)

                for row, coord in enumerate(['centroid_x', 'centroid_y']):
                    ax = axes[row][fi]
                    vals = fdf[value_col].values
                    coords = fdf[coord].values
                    ax.scatter(coords, vals, s=3, alpha=0.3, color=color, edgecolors='none')

                    # trend line
                    mask = np.isfinite(vals) & np.isfinite(coords)
                    if mask.sum() > 5:
                        z = np.polyfit(coords[mask], vals[mask], 1)
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

            fig.suptitle(f'{col_label} vs position — {gene} T={tp}h', fontsize=12)
            plt.tight_layout()
            fname = f'{value_col}_centroid_{gene}_T{tp}h.png'
            plt.savefig(qc_dir / fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fname}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect mKate T-shift segmentation results')
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory (single gene folder or parent of all genes)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: <input>_analysis)')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / (input_path.name + '_analysis')
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

    print("\nGenerating plots...")
    for col in value_cols:
        plot_time_evolution(df, stats_df, col,
                            output_path / f'{col}_time_evolution.png',
                            title_cols=title_cols)
        plot_ridge(df, col,
                   output_path / f'{col}_ridge.png',
                   title_cols=title_cols)
        if col == 'gfp_mcherry_ratio':
            plot_ridge(df, col,
                       output_path / f'{col}_ridge_log.png',
                       title_cols=title_cols, log_scale=True)
        plot_stripplot(df, col,
                       output_path / f'{col}_stripplot.png',
                       title_cols=title_cols)
        if df['gene'].nunique() > 1:
            plot_comparison(df, stats_df, col,
                            output_path / f'{col}_comparison.png',
                            title_cols=title_cols)

    # ----- QC plots -----
    print("\nGenerating QC plots...")
    qc_dir = output_path / 'QC'
    qc_dir.mkdir(exist_ok=True)
    for col in value_cols:
        plot_qc_histograms(df, col, qc_dir, title_cols=title_cols)
        if 'centroid_x' in df.columns and 'centroid_y' in df.columns:
            plot_qc_centroid_scatter(df, col, qc_dir, title_cols=title_cols)

    print("\nDone!")


if __name__ == '__main__':
    main()
