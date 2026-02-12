#!/usr/bin/env python
"""
Collect and analyze 2-channel segmentation results across genes and timepoints.

Folder naming convention:
  02092026_LB_25_{gene}_{hours}h{minutes}m_to37_{sample}/
    1/cell_measurements.csv   <- replicate 1 (FOV)
    2/cell_measurements.csv   <- replicate 2 (FOV)
    ...

Replicates within a sample are pooled into a single distribution.
Plots show time evolution per gene, with samples as separate curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse


def parse_folder_name_2ch(folder_name):
    """
    Parse folder name to extract gene, timepoint, and sample.

    Pattern: {date}_LB_25_{gene}_{H}h{M}m_to37_{sample}
    Examples:
    - 02092026_LB_25_flu3_0h00m_to37_1  -> gene=flu3, time=0.0h, sample=1
    - 02092026_LB_25_ompT_1h30m_to37_2  -> gene=ompT, time=1.5h, sample=2
    """
    match = re.match(
        r'\d+_LB_25_(.+?)_(\d+)h(\d+)m_to37_(\d+)$',
        folder_name
    )
    if match:
        gene = match.group(1)
        hours = int(match.group(2))
        minutes = int(match.group(3))
        sample = int(match.group(4))
        timepoint_min = hours * 60 + minutes
        return {
            'gene': gene,
            'timepoint_min': timepoint_min,
            'timepoint_hr': hours + minutes / 60.0,
            'timepoint_label': f'{hours}h{minutes:02d}m',
            'sample': sample,
            'folder': folder_name,
        }
    return None


def collect_measurements_2ch(input_dir, folder_pattern=None):
    """
    Collect all cell measurements from 2-channel output folders.

    Searches for cell_measurements.csv recursively within each top-level folder
    (replicates are subfolders like 1.nd2/, 2.nd2/).

    Args:
        input_dir: Path to directory containing sample folders
        folder_pattern: Optional regex pattern to filter top-level folders

    Returns:
        DataFrame with all measurements and parsed metadata
    """
    input_path = Path(input_dir)
    all_data = []

    for folder in sorted(input_path.iterdir()):
        if not folder.is_dir():
            continue

        if folder_pattern and not re.match(folder_pattern, folder.name):
            continue

        metadata = parse_folder_name_2ch(folder.name)
        if metadata is None:
            print(f"  Skipping {folder.name} - could not parse naming pattern")
            continue

        # Find all cell_measurements.csv in subfolders (replicates)
        csv_files = sorted(folder.glob('*/cell_measurements.csv'))
        if not csv_files:
            # Also check for cell_measurements.csv directly in the folder
            direct = folder / 'cell_measurements.csv'
            if direct.exists():
                csv_files = [direct]

        if not csv_files:
            print(f"  Skipping {folder.name} - no cell_measurements.csv found")
            continue

        n_cells_total = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError:
                continue
            if len(df) == 0:
                continue
            replicate_name = csv_file.parent.name if csv_file.parent != folder else 'single'
            df['replicate'] = replicate_name
            for key, value in metadata.items():
                df[key] = value
            all_data.append(df)
            n_cells_total += len(df)

        print(f"  {folder.name}: {len(csv_files)} replicates, {n_cells_total} cells "
              f"(gene={metadata['gene']}, t={metadata['timepoint_label']}, s={metadata['sample']})")

    if not all_data:
        raise ValueError("No data found!")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal cells collected: {len(combined)}")

    return combined


def filter_saturated(df):
    """Remove cells with saturated GFP pixels."""
    if 'gfp_saturated' not in df.columns:
        print("No gfp_saturated column found, keeping all cells")
        return df

    mask = df['gfp_saturated'] == 0
    n_removed = (~mask).sum()
    print(f"Removed {n_removed} saturated cells ({100*n_removed/len(df):.1f}%)")
    return df[mask].copy()


def calculate_percentiles(df, group_cols, value_cols, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """Calculate percentiles for specified columns grouped by group_cols."""
    results = []
    for name, group in df.groupby(group_cols):
        if not isinstance(name, tuple):
            name = (name,)
        row = dict(zip(group_cols, name))
        row['n_cells'] = len(group)
        for col in value_cols:
            values = group[col].dropna()
            if len(values) > 0:
                for p in percentiles:
                    row[f'{col}_p{p}'] = np.percentile(values, p)
            else:
                for p in percentiles:
                    row[f'{col}_p{p}'] = np.nan
        results.append(row)
    return pd.DataFrame(results)


def calculate_variability_stats(df, group_cols, value_cols):
    """Calculate CV and robust spread grouped by group_cols."""
    results = []
    if not group_cols:
        groups = [(None, df)]
    else:
        groups = df.groupby(group_cols)

    for name, group in groups:
        if name is None:
            row = {}
        elif not isinstance(name, tuple):
            row = {group_cols[0]: name}
        else:
            row = dict(zip(group_cols, name))

        row['n_cells'] = len(group)
        for col in value_cols:
            values = group[col].dropna()
            if len(values) < 2:
                row[f'{col}_cv'] = np.nan
                row[f'{col}_robust_spread'] = np.nan
                continue
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            p10 = np.percentile(values, 10)
            p90 = np.percentile(values, 90)
            row[f'{col}_cv'] = std_val / mean_val if mean_val != 0 else np.nan
            row[f'{col}_robust_spread'] = (p90 - p10) / median_val if median_val != 0 else np.nan
        results.append(row)

    return pd.DataFrame(results)


TITLE_COLS = {
    'gfp_rate': 'GFP rate (bkg-sub / ms)',
    'gfp_flux': 'GFP flux (rate / area)',
    'gfp_sum_bkg_sub': 'GFP sum (bkg-sub)',
}


def plot_time_evolution(percentile_df, raw_df, value_col, output_path, title=None):
    """
    Plot time evolution of median with 25-75 percentile error bars.
    One curve per sample, individual data points shown with transparency.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    samples = sorted(percentile_df['sample'].unique())

    for i, sample in enumerate(samples):
        color = colors[i % len(colors)]

        # Individual data points
        sample_raw = raw_df[raw_df['sample'] == sample]
        jitter = np.random.uniform(-0.04, 0.04, len(sample_raw))
        ax.scatter(sample_raw['timepoint_hr'] + jitter, sample_raw[value_col],
                   color=color, alpha=0.2, s=8, edgecolors='none')

        # Median with error bars
        sample_data = percentile_df[percentile_df['sample'] == sample].sort_values('timepoint_hr')
        x = sample_data['timepoint_hr'].values
        median = sample_data[f'{value_col}_p50'].values
        p25 = sample_data[f'{value_col}_p25'].values
        p75 = sample_data[f'{value_col}_p75'].values

        ax.errorbar(x, median, yerr=[median - p25, p75 - median],
                    marker='o', capsize=5, capthick=2, linewidth=2,
                    label=f'Sample {sample}', color=color, zorder=10)

    col_label = TITLE_COLS.get(value_col, value_col.replace('_', ' ').title())
    ax.set_xlabel('Time after shift to 37C (hours)', fontsize=12)
    ax.set_ylabel(col_label, fontsize=12)
    ax.set_title(title or col_label, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(percentile_df['timepoint_hr'].unique()))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_variability_evolution(variability_df, value_cols, output_path):
    """Plot CV and robust spread evolution over time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    colors = plt.cm.tab10.colors
    variability_df = variability_df.sort_values('timepoint_hr')
    x = variability_df['timepoint_hr'].values

    for i, col in enumerate(value_cols):
        col_label = TITLE_COLS.get(col, col.replace('_', ' ').title())
        if f'{col}_cv' in variability_df.columns:
            axes[0].plot(x, variability_df[f'{col}_cv'], marker='o', linewidth=2,
                         label=col_label, color=colors[i % len(colors)])
        if f'{col}_robust_spread' in variability_df.columns:
            axes[1].plot(x, variability_df[f'{col}_robust_spread'], marker='o', linewidth=2,
                         label=col_label, color=colors[i % len(colors)])

    axes[0].set_ylabel('CV (std/mean)', fontsize=12)
    axes[0].set_title('Coefficient of Variation over Time', fontsize=14)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time after shift to 37C (hours)', fontsize=12)
    axes[1].set_ylabel('(p90-p10)/median', fontsize=12)
    axes[1].set_title('Robust Spread over Time', fontsize=14)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(sorted(variability_df['timepoint_hr'].unique()))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_histogram_ridge(df, value_col, output_path, n_bins=30):
    """
    Ridge plot: stacked histograms offset vertically per timepoint.
    """
    timepoints = sorted(df['timepoint_hr'].unique())
    timepoint_labels = []
    for tp in timepoints:
        tp_rows = df[df['timepoint_hr'] == tp]
        label = tp_rows['timepoint_label'].iloc[0]
        timepoint_labels.append(label)

    all_values = df[value_col].dropna()
    all_values = all_values[all_values > 0]
    if len(all_values) == 0:
        return

    global_min = all_values.min()
    global_max = all_values.max()
    bin_edges = np.linspace(global_min, global_max, n_bins + 1)

    rows = []
    for tp, label in zip(timepoints, timepoint_labels):
        tp_values = df[df['timepoint_hr'] == tp][value_col].dropna()
        tp_values = tp_values[tp_values > 0]
        if len(tp_values) > 0:
            counts, _ = np.histogram(tp_values, bins=bin_edges)
            fractions = counts / counts.sum()
        else:
            fractions = np.zeros(n_bins)
        rows.append((f't = {label}', fractions))

    n_rows = len(rows)
    if n_rows == 0:
        return

    max_height = max(h.max() for _, h in rows)
    spacing = max_height * 0.8
    tp_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_rows))

    fig, ax = plt.subplots(figsize=(12, 3 + 2 * n_rows))

    for i in range(n_rows):
        y_pos = (n_rows - 1 - i) * spacing
        label, fractions = rows[i]
        color = tp_colors[i]

        for j in range(n_bins):
            left = bin_edges[j]
            width = bin_edges[j + 1] - bin_edges[j]
            ax.bar(left, fractions[j], width=width, bottom=y_pos,
                   align='edge', color=color, edgecolor='black',
                   linewidth=0.5, alpha=0.85)

        ax.plot([bin_edges[0], bin_edges[-1]], [y_pos, y_pos],
                color='black', linewidth=0.5, alpha=0.3)

    col_label = TITLE_COLS.get(value_col, value_col.replace('_', ' ').title())
    ax.set_xlabel(col_label, fontsize=12)
    ax.set_title(f'{col_label} Distribution', fontsize=14)
    ax.set_yticks([(n_rows - 1 - i) * spacing for i in range(n_rows)])
    ax.set_yticklabels([label for label, _ in rows], fontsize=10)
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_gene_comparison(df, value_col, output_path, genes=None):
    """
    Box/violin plot comparing genes at each timepoint.
    """
    if genes is None:
        genes = sorted(df['gene'].unique())

    timepoints = sorted(df['timepoint_hr'].unique())
    n_tp = len(timepoints)

    if n_tp == 0:
        return

    fig, axes = plt.subplots(1, n_tp, figsize=(5 * n_tp, 6), sharey=True, squeeze=False)
    axes = axes[0]
    colors = plt.cm.Set2.colors

    for t_idx, tp in enumerate(timepoints):
        ax = axes[t_idx]
        tp_data = df[df['timepoint_hr'] == tp]
        tp_label = tp_data['timepoint_label'].iloc[0] if len(tp_data) > 0 else f'{tp}h'

        positions = []
        labels = []
        for g_idx, gene in enumerate(genes):
            gene_data = tp_data[tp_data['gene'] == gene][value_col].dropna()
            if len(gene_data) == 0:
                continue
            pos = g_idx
            positions.append(pos)
            labels.append(gene)
            bp = ax.boxplot([gene_data.values], positions=[pos], widths=0.6,
                           patch_artist=True, showfliers=False)
            bp['boxes'][0].set_facecolor(colors[g_idx % len(colors)])
            bp['boxes'][0].set_alpha(0.7)

            # Overlay individual points
            jitter = np.random.uniform(-0.15, 0.15, len(gene_data))
            ax.scatter(pos + jitter, gene_data.values,
                      color=colors[g_idx % len(colors)], alpha=0.3, s=10, edgecolors='none')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_title(f't = {tp_label}', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    col_label = TITLE_COLS.get(value_col, value_col.replace('_', ' ').title())
    axes[0].set_ylabel(col_label, fontsize=12)
    fig.suptitle(f'{col_label} by Gene', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect and analyze 2-channel segmentation results')
    parser.add_argument('--input', '-i', type=str, default='segmentation_output',
                        help='Path to directory containing sample output folders')
    parser.add_argument('--output', '-o', type=str, default='analysis_2ch',
                        help='Path to save analysis results')
    parser.add_argument('--pattern', '-p', type=str, default=None,
                        help='Regex pattern to filter top-level folders')
    parser.add_argument('--gene', '-g', type=str, nargs='+', default=None,
                        help='Filter to specific gene(s)')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all measurements
    print("Collecting measurements...")
    df = collect_measurements_2ch(args.input, args.pattern)

    # Filter to specific genes if requested
    if args.gene:
        df = df[df['gene'].isin(args.gene)].copy()
        print(f"Filtered to genes: {args.gene} ({len(df)} cells)")

    # Filter saturated cells
    print("\nFiltering saturated cells...")
    df_filtered = filter_saturated(df)

    # Save merged data
    merged_file = output_path / 'all_cells_merged.csv'
    df_filtered.to_csv(merged_file, index=False)
    print(f"Saved merged data: {merged_file}")

    # Determine available value columns
    value_cols = ['gfp_rate', 'gfp_flux', 'gfp_sum_bkg_sub']
    value_cols = [col for col in value_cols if col in df_filtered.columns]
    if not value_cols:
        print("Warning: No GFP metric columns found. Available:", df_filtered.columns.tolist())
        return

    genes = sorted(df_filtered['gene'].unique())
    print(f"\nGenes found: {genes}")
    print(f"Analyzing columns: {value_cols}")

    # Per-gene analysis
    for gene in genes:
        gene_df = df_filtered[df_filtered['gene'] == gene]
        gene_dir = output_path / gene
        gene_dir.mkdir(exist_ok=True)

        print(f"\n--- Gene: {gene} ({len(gene_df)} cells) ---")

        # Percentiles grouped by timepoint + sample (replicates pooled)
        group_cols = ['gene', 'timepoint_hr', 'sample']
        percentile_df = calculate_percentiles(gene_df, group_cols, value_cols)

        percentile_file = gene_dir / 'percentiles.csv'
        percentile_df.to_csv(percentile_file, index=False)

        # Time evolution plots
        print("  Plotting time evolution...")
        for col in value_cols:
            plot_time_evolution(
                percentile_df, gene_df, col,
                gene_dir / f'{col}_time_evolution.png',
                title=f'{gene}: {TITLE_COLS.get(col, col)}',
            )

        # Variability stats (grouped by timepoint, pooling samples)
        var_group_cols = ['gene', 'timepoint_hr']
        var_df = calculate_variability_stats(gene_df, var_group_cols, value_cols)
        var_df.to_csv(gene_dir / 'variability.csv', index=False)

        print("  Plotting variability...")
        plot_variability_evolution(var_df, value_cols, gene_dir / 'variability_evolution.png')

        # Ridge plots
        print("  Plotting ridge histograms...")
        for col in value_cols:
            plot_histogram_ridge(gene_df, col, gene_dir / f'{col}_histogram_ridge.png')

    # Cross-gene comparison (if multiple genes)
    if len(genes) > 1:
        print(f"\n--- Cross-gene comparison ---")
        compare_dir = output_path / 'comparison'
        compare_dir.mkdir(exist_ok=True)

        for col in value_cols:
            plot_gene_comparison(df_filtered, col, compare_dir / f'{col}_gene_comparison.png',
                                genes=genes)

    # Summary table across all genes and timepoints
    summary_cols = ['gene', 'timepoint_hr', 'timepoint_label']
    summary_df = calculate_percentiles(df_filtered, summary_cols, value_cols)
    summary_df.to_csv(output_path / 'summary_percentiles.csv', index=False)
    print(f"\nSaved summary: {output_path / 'summary_percentiles.csv'}")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
