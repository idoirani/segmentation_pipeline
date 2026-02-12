#!/usr/bin/env python
"""
Collect and analyze segmentation pipeline results across samples and timepoints.

Naming convention:
- M9glyc_25_2/2.nd2 -> M9glyc at 25C, time 0, sample 2, replicate 2
- M9glyc_25_2hr_1/1.nd2 -> M9glyc at 25C, 2hr, sample 1, replicate 1

Output folder pattern: segmentation_output/{condition}_{temp}_{time}hr_{sample}_{replicate}/
or for time 0: segmentation_output/{condition}_{temp}_{sample}_{replicate}/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import argparse


def parse_folder_name(folder_name):
    """
    Parse folder name to extract condition, temperature, timepoint, sample, and replicate.

    Examples:
    - M9glyc_25_2_2 -> condition=M9glyc, temp=25, time=0, sample=2, replicate=2
    - M9glyc_25_2hr_1_1 -> condition=M9glyc, temp=25, time=2, sample=1, replicate=1
    - M9glyc_25_3hr_2_3 -> condition=M9glyc, temp=25, time=3, sample=2, replicate=3
    """
    # Pattern for timepoint > 0: {condition}_{temp}_{N}hr_{sample}_{replicate}
    match_hr = re.match(r'(.+?)_(\d+)_(\d+)hr_(\d+)_(\d+)$', folder_name)
    if match_hr:
        return {
            'condition': match_hr.group(1),
            'temperature': int(match_hr.group(2)),
            'timepoint_hr': int(match_hr.group(3)),
            'sample': int(match_hr.group(4)),
            'replicate': int(match_hr.group(5)),
            'folder': folder_name
        }

    # Pattern for time 0: {condition}_{temp}_{sample}_{replicate}
    match_t0 = re.match(r'(.+?)_(\d+)_(\d+)_(\d+)$', folder_name)
    if match_t0:
        return {
            'condition': match_t0.group(1),
            'temperature': int(match_t0.group(2)),
            'timepoint_hr': 0,
            'sample': int(match_t0.group(3)),
            'replicate': int(match_t0.group(4)),
            'folder': folder_name
        }

    return None


def collect_measurements(output_dir, folder_pattern=None):
    """
    Collect all cell measurements from segmentation output folders.

    Args:
        output_dir: Path to segmentation_output directory
        folder_pattern: Optional regex pattern to filter folders

    Returns:
        DataFrame with all measurements and parsed metadata
    """
    output_path = Path(output_dir)
    all_data = []

    for folder in output_path.iterdir():
        if not folder.is_dir():
            continue

        if folder_pattern and not re.match(folder_pattern, folder.name):
            continue

        # Parse folder name
        metadata = parse_folder_name(folder.name)
        if metadata is None:
            print(f"  Skipping {folder.name} - could not parse naming pattern")
            continue

        # Load cell measurements
        measurements_file = folder / 'cell_measurements.csv'
        if not measurements_file.exists():
            print(f"  Skipping {folder.name} - no cell_measurements.csv")
            continue

        df = pd.read_csv(measurements_file)

        # Add metadata columns
        for key, value in metadata.items():
            df[key] = value

        all_data.append(df)
        print(f"  Loaded {len(df)} cells from {folder.name}")

    if not all_data:
        raise ValueError("No data found!")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal cells collected: {len(combined)}")

    return combined


def filter_saturated(df):
    """Remove cells with saturation warnings."""
    # Check which saturation columns exist
    sat_cols = [col for col in df.columns if 'saturated' in col.lower()]

    if not sat_cols:
        print("No saturation columns found, keeping all cells")
        return df

    # Create mask for non-saturated cells
    mask = pd.Series(True, index=df.index)
    for col in sat_cols:
        mask &= (df[col] == 0)

    n_removed = (~mask).sum()
    print(f"Removed {n_removed} saturated cells ({100*n_removed/len(df):.1f}%)")

    return df[mask].copy()


def calculate_percentiles(df, group_cols, value_cols, percentiles=[5, 10, 25, 50, 75, 90, 95]):
    """
    Calculate percentiles for specified columns grouped by group_cols.

    Args:
        df: DataFrame with measurements
        group_cols: Columns to group by (e.g., ['condition', 'temperature', 'timepoint_hr'])
        value_cols: Columns to calculate percentiles for
        percentiles: List of percentiles to calculate

    Returns:
        DataFrame with percentiles
    """
    results = []
    for name, group in df.groupby(group_cols):
        if not isinstance(name, tuple):
            name = (name,)

        row = dict(zip(group_cols, name))
        row['n_cells'] = len(group)
        for col in value_cols:
            for p in percentiles:
                row[f'{col}_p{p}'] = np.percentile(group[col].dropna(), p)

        results.append(row)

    return pd.DataFrame(results)


def plot_time_evolution(percentile_df, raw_df, value_col, output_path,
                        control_groups=None, title_cols=None, title=None):
    """
    Plot time evolution of median with 25-75 percentile error bars.
    One curve per sample (accumulated over replicates).
    Individual data points shown with transparency.
    Optional narrow right panel showing control group distributions.

    Args:
        percentile_df: DataFrame with percentile data (grouped by condition, temp, timepoint, sample)
        raw_df: DataFrame with all individual cell measurements
        value_col: Base column name (e.g., 'gfp_rate')
        output_path: Path to save the plot
        control_groups: dict of {label: DataFrame} for control groups, or None
        title: Optional plot title
    """
    if control_groups:
        fig, (ax, ax_ctrl) = plt.subplots(1, 2, figsize=(13, 6), sharey=True,
                                           gridspec_kw={'width_ratios': [4, 1], 'wspace': 0.05})
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    samples = sorted(percentile_df['sample'].unique())

    for i, sample in enumerate(samples):
        color = colors[i % len(colors)]

        # Plot individual data points with jitter and transparency
        sample_raw = raw_df[raw_df['sample'] == sample]
        jitter = np.random.uniform(-0.1, 0.1, len(sample_raw))
        ax.scatter(sample_raw['timepoint_hr'] + jitter, sample_raw[value_col],
                   color=color, alpha=0.25, s=10, edgecolors='none')

        # Plot median with error bars
        sample_data = percentile_df[percentile_df['sample'] == sample].sort_values('timepoint_hr')

        x = sample_data['timepoint_hr'].values
        median = sample_data[f'{value_col}_p50'].values
        p25 = sample_data[f'{value_col}_p25'].values
        p75 = sample_data[f'{value_col}_p75'].values

        yerr_lower = median - p25
        yerr_upper = p75 - median

        ax.errorbar(x, median, yerr=[yerr_lower, yerr_upper],
                    marker='o', capsize=5, capthick=2, linewidth=2,
                    label=f'Sample {sample}', color=color, zorder=10)

    col_label = title_cols[value_col] if title_cols and value_col in title_cols else value_col.replace('_', ' ').title()
    ax.set_xlabel('Time after shift to 37C (hours)', fontsize=12)
    ax.set_ylabel(col_label, fontsize=12)
    ax.set_title(title or f'{col_label}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(percentile_df['timepoint_hr'].unique()))
    ax.set_yscale('log')

    # Control group panel - samples colored same as main plot, controls by marker
    if control_groups:
        ctrl_markers = ['x', 'o', 's', 'D']
        positions = list(range(len(control_groups)))
        labels = []
        for ctrl_idx, (label, ctrl_df) in enumerate(control_groups.items()):
            marker = ctrl_markers[ctrl_idx % len(ctrl_markers)]
            ctrl_samples = sorted(ctrl_df['sample'].unique())

            for s_idx, sample in enumerate(ctrl_samples):
                sample_data = ctrl_df[ctrl_df['sample'] == sample]
                values = sample_data[value_col].dropna()
                values = values[values > 0]
                # Use same color as main plot samples
                color = colors[s_idx % len(colors)]

                if len(values) > 0:
                    median = np.median(values)
                    p25 = np.percentile(values, 25)
                    p75 = np.percentile(values, 75)

                    jitter = np.random.uniform(-0.2, 0.2, len(values))
                    ax_ctrl.scatter(ctrl_idx + jitter, values,
                                    color=color, alpha=0.25, s=10,
                                    marker=marker, edgecolors='none')
                    ax_ctrl.errorbar(ctrl_idx, median,
                                     yerr=[[median - p25], [p75 - median]],
                                     marker=marker, capsize=5, capthick=2, linewidth=2,
                                     color=color, zorder=10, markersize=8,
                                     label=f'{label} S{sample}')

            labels.append(label)

        ax_ctrl.set_xticks(positions)
        ax_ctrl.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax_ctrl.set_title('Controls', fontsize=12)
        ax_ctrl.grid(True, alpha=0.3)
        ax_ctrl.tick_params(labelleft=False)
        ax_ctrl.legend(loc='best', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def calculate_variability_stats(df, group_cols, value_cols):
    """
    Calculate variability metrics (CV and robust spread) grouped by group_cols.

    Args:
        df: DataFrame with measurements
        group_cols: Columns to group by (e.g., ['condition', 'temperature', 'timepoint_hr']).
                   If empty, treats entire df as one group.
        value_cols: Columns to calculate variability for

    Returns:
        DataFrame with variability metrics
    """
    results = []

    # Handle empty group_cols - treat entire df as one group
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
            mean_val = np.mean(values)
            std_val = np.std(values)
            median_val = np.median(values)
            p10 = np.percentile(values, 10)
            p90 = np.percentile(values, 90)

            # Coefficient of variation (std/mean)
            row[f'{col}_cv'] = std_val / mean_val if mean_val != 0 else np.nan
            # Robust spread: (p90-p10)/median
            row[f'{col}_robust_spread'] = (p90 - p10) / median_val if median_val != 0 else np.nan

        results.append(row)

    return pd.DataFrame(results)


def plot_variability_evolution(variability_df, value_cols, output_path, title_cols=None):
    """
    Plot CV and robust spread evolution over time (combined across samples).

    Args:
        variability_df: DataFrame with variability metrics by timepoint
        value_cols: List of base column names to plot
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    colors = plt.cm.tab10.colors
    variability_df = variability_df.sort_values('timepoint_hr')
    x = variability_df['timepoint_hr'].values

    # Plot CV (std/mean)
    ax = axes[0]
    for i, col in enumerate(value_cols):
        col_label = title_cols[col] if title_cols and col in title_cols else col.replace('_', ' ').title()
        ax.plot(x, variability_df[f'{col}_cv'], marker='o', linewidth=2,
                label=col_label, color=colors[i % len(colors)])
    ax.set_ylabel('CV (std/mean)', fontsize=12)
    ax.set_title('Coefficient of Variation over Time', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Plot robust spread (p90-p10)/median
    ax = axes[1]
    for i, col in enumerate(value_cols):
        col_label = title_cols[col] if title_cols and col in title_cols else col.replace('_', ' ').title()
        ax.plot(x, variability_df[f'{col}_robust_spread'], marker='o', linewidth=2,
                label=col_label, color=colors[i % len(colors)])
    ax.set_xlabel('Time after shift to 37C (hours)', fontsize=12)
    ax.set_ylabel('(p90-p10)/median', fontsize=12)
    ax.set_title('Robust Spread over Time', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set x-axis to show integer hours
    ax.set_xticks(sorted(variability_df['timepoint_hr'].unique()))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_histogram_ridge(df, value_col, output_path, n_bins=30, control_groups=None, title_cols=None):
    """
    Plot histogram evolution as a ridge plot: stacked 1D histograms
    offset vertically per timepoint, so each distribution's shape is visible.
    Control groups appear as separate rows at the bottom.

    Args:
        df: DataFrame with all cell measurements
        value_col: Column name to histogram (e.g., 'gfp_rate')
        output_path: Path to save the plot
        n_bins: Number of bins (in log space)
        control_groups: dict of {label: DataFrame} for control groups, or None
    """
    timepoints = sorted(df['timepoint_hr'].unique())

    # Collect all values (including controls) for global bin range
    all_values = df[value_col].dropna()
    all_values = all_values[all_values > 0]
    if control_groups:
        for ctrl_df in control_groups.values():
            ctrl_vals = ctrl_df[value_col].dropna()
            all_values = pd.concat([all_values, ctrl_vals[ctrl_vals > 0]])

    global_min = all_values.min()
    global_max = all_values.max()
    bin_edges = np.linspace(global_min, global_max, n_bins + 1)

    # Build list of (label, histogram) rows: timepoints top-to-bottom, then controls at bottom
    rows = []
    for tp in timepoints:
        tp_values = df[df['timepoint_hr'] == tp][value_col].dropna()
        tp_values = tp_values[tp_values > 0]
        if len(tp_values) > 0:
            counts, _ = np.histogram(tp_values, bins=bin_edges)
            fractions = counts / counts.sum()
        else:
            fractions = np.zeros(n_bins)
        rows.append((f't = {tp}hr', fractions))

    n_tp = len(timepoints)
    n_ctrl = 0
    if control_groups:
        for label, ctrl_df in control_groups.items():
            ctrl_vals = ctrl_df[value_col].dropna()
            ctrl_vals = ctrl_vals[ctrl_vals > 0]
            if len(ctrl_vals) > 0:
                counts, _ = np.histogram(ctrl_vals, bins=bin_edges)
                fractions = counts / counts.sum()
            else:
                fractions = np.zeros(n_bins)
            rows.append((label, fractions))
            n_ctrl += 1

    n_rows = len(rows)

    # Vertical spacing
    max_height = max(h.max() for _, h in rows) if rows else 1
    spacing = max_height * 0.8

    # Colors: viridis for timepoints, pink/grey for controls
    ctrl_colors = ['#e377c2', '#7f7f7f', '#8c564b', '#17becf']
    tp_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_tp))

    fig, ax = plt.subplots(figsize=(12, 3 + 2 * n_rows))

    # Row 0 is at the top (highest y), row n_rows-1 at bottom (y=0)
    for i in range(n_rows):
        y_pos = (n_rows - 1 - i) * spacing
        label, fractions = rows[i]

        if i < n_tp:
            color = tp_colors[i]
        else:
            color = ctrl_colors[(i - n_tp) % len(ctrl_colors)]

        for j in range(n_bins):
            left = bin_edges[j]
            width = bin_edges[j + 1] - bin_edges[j]
            ax.bar(left, fractions[j], width=width, bottom=y_pos,
                   align='edge', color=color, edgecolor='black',
                   linewidth=0.5, alpha=0.85)

        ax.plot([bin_edges[0], bin_edges[-1]], [y_pos, y_pos],
                color='black', linewidth=0.5, alpha=0.3)

    col_label = title_cols[value_col] if title_cols and value_col in title_cols else value_col.replace('_', ' ').title()
    ax.set_xlabel(col_label, fontsize=12)
    ax.set_title(f'{col_label} Distribution', fontsize=14)

    # Y-axis labels: row 0 at top, row n_rows-1 at bottom
    ax.set_yticks([(n_rows - 1 - i) * spacing for i in range(n_rows)])
    ax.set_yticklabels([label for label, _ in rows], fontsize=10)
    ax.set_ylabel('')

    # Draw separator between timepoints and controls
    if n_ctrl > 0:
        sep_y = (n_ctrl - 0.5) * spacing
        ax.axhline(sep_y, color='black', linewidth=1, linestyle='--', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Collect and analyze segmentation results')
    parser.add_argument('--input', '-i', type=str, default='segmentation_output',
                        help='Path to segmentation output directory')
    parser.add_argument('--output', '-o', type=str, default='analysis_results',
                        help='Path to save analysis results')
    parser.add_argument('--pattern', '-p', type=str, default=None,
                        help='Regex pattern to filter folders (e.g., "M9glyc_25.*")')
    parser.add_argument('--controls', '-c', type=str, nargs='+', default=None,
                        help='Control group patterns as label:regex pairs '
                             '(e.g., "M9glyc_37C:M9glyc_37_.*" "LB_37C:LB_37_.*")')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all measurements
    print("Collecting measurements...")
    df = collect_measurements(args.input, args.pattern)

    # Filter saturated cells
    print("\nFiltering saturated cells...")
    df_filtered = filter_saturated(df)

    # Save merged data
    merged_file = output_path / 'all_cells_merged.csv'
    df_filtered.to_csv(merged_file, index=False)
    print(f"\nSaved merged data: {merged_file}")

    # Define columns to analyze
    value_cols = ['mcherry_rate', 'gfp_rate', 'gfp_mcherry_ratio']
    title_cols = {'mcherry_rate': 'mCherry rate', 'gfp_rate': 'GFP rate', 'gfp_mcherry_ratio': 'GFP/mCherry'}

    # Check which columns exist
    value_cols = [col for col in value_cols if col in df_filtered.columns]
    if not value_cols:
        print("Warning: No rate columns found. Available columns:", df_filtered.columns.tolist())
        return

    print(f"\nAnalyzing columns: {value_cols}")

    # Collect control groups if specified
    control_groups = None
    if args.controls:
        print("\nCollecting control groups...")
        control_groups = {}
        for ctrl_spec in args.controls:
            label, pattern = ctrl_spec.split(':', 1)
            print(f"  Control '{label}' (pattern: {pattern}):")
            ctrl_df = collect_measurements(args.input, pattern)
            ctrl_df = filter_saturated(ctrl_df)
            control_groups[label] = ctrl_df
        print()

    # Calculate percentiles grouped by condition, temperature, timepoint, and sample
    # Replicates are accumulated within each sample
    group_cols = ['condition', 'temperature', 'timepoint_hr', 'sample']
    print("\nCalculating percentiles...")
    percentile_df = calculate_percentiles(df_filtered, group_cols, value_cols)
    percentile_df['is_control'] = False
    percentile_df['control_label'] = ''

    # Calculate percentiles for control groups and append
    if control_groups:
        print("Calculating control group percentiles...")
        ctrl_group_cols = ['sample']
        for label, ctrl_df in control_groups.items():
            ctrl_percentile = calculate_percentiles(ctrl_df, ctrl_group_cols, value_cols)
            # Add identifying columns
            ctrl_percentile['is_control'] = True
            ctrl_percentile['control_label'] = label
            # Fill in missing columns with values from control data
            ctrl_percentile['condition'] = ctrl_df['condition'].iloc[0] if 'condition' in ctrl_df.columns else label
            ctrl_percentile['temperature'] = ctrl_df['temperature'].iloc[0] if 'temperature' in ctrl_df.columns else None
            ctrl_percentile['timepoint_hr'] = ctrl_df['timepoint_hr'].iloc[0] if 'timepoint_hr' in ctrl_df.columns else None
            percentile_df = pd.concat([percentile_df, ctrl_percentile], ignore_index=True)

    # Save percentile table
    percentile_file = output_path / 'percentiles_by_timepoint.csv'
    percentile_df.to_csv(percentile_file, index=False)
    print(f"Saved percentile table: {percentile_file}")

    # Print summary
    print("\nPercentile summary:")
    print(percentile_df.to_string())

    # Generate plots
    print("\nGenerating plots...")
    for col in value_cols:
        plot_file = output_path / f'{col}_time_evolution.png'
        plot_time_evolution(percentile_df, df_filtered, col, plot_file,
                            control_groups=control_groups, title_cols=title_cols)

    # Calculate variability stats across both samples (grouped by timepoint only)
    print("\nCalculating variability statistics...")
    variability_group_cols = ['condition', 'temperature', 'timepoint_hr']
    variability_df = calculate_variability_stats(df_filtered, variability_group_cols, value_cols)
    variability_df['is_control'] = False
    variability_df['control_label'] = ''

    # Calculate variability stats for control groups and append
    if control_groups:
        print("Calculating control group variability statistics...")
        for label, ctrl_df in control_groups.items():
            # Calculate variability for entire control group (no subgrouping)
            ctrl_variability = calculate_variability_stats(ctrl_df, [], value_cols)
            ctrl_variability['is_control'] = True
            ctrl_variability['control_label'] = label
            ctrl_variability['condition'] = ctrl_df['condition'].iloc[0] if 'condition' in ctrl_df.columns else label
            ctrl_variability['temperature'] = ctrl_df['temperature'].iloc[0] if 'temperature' in ctrl_df.columns else None
            ctrl_variability['timepoint_hr'] = ctrl_df['timepoint_hr'].iloc[0] if 'timepoint_hr' in ctrl_df.columns else None
            variability_df = pd.concat([variability_df, ctrl_variability], ignore_index=True)

    # Save variability table
    variability_file = output_path / 'variability_by_timepoint.csv'
    variability_df.to_csv(variability_file, index=False)
    print(f"Saved variability table: {variability_file}")

    # Plot variability evolution
    variability_plot_file = output_path / 'variability_evolution.png'
    plot_variability_evolution(variability_df, value_cols, variability_plot_file, title_cols=title_cols)

    # Generate ridge plot histograms for each metric
    print("\nGenerating histogram ridge plots...")
    for col in value_cols:
        hist_file = output_path / f'{col}_histogram_ridge.png'
        plot_histogram_ridge(df_filtered, col, hist_file, control_groups=control_groups, title_cols=title_cols)

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
