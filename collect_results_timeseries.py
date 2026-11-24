#!/usr/bin/env python
"""
Collect and analyze segmentation results from a CellASIC time-series run.

Folder structure:
    <input_dir>/t0/cell_measurements.csv
    <input_dir>/t1/cell_measurements.csv
    ...
    <input_dir>/t{N}/cell_measurements.csv

Frame-to-time conversion is read from nd2_metadata.csv (TimeLoop_periodMs).
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

def get_frame_period_ms(input_dir):
    """Read TimeLoop_periodMs from the first available nd2_metadata.csv."""
    input_path = Path(input_dir)
    for meta_file in sorted(input_path.glob('t*/nd2_metadata.csv')):
        try:
            meta = pd.read_csv(meta_file)
            if 'TimeLoop_periodMs' in meta.columns:
                val = meta['TimeLoop_periodMs'].iloc[0]
                if pd.notna(val) and val > 0:
                    return float(val)
        except Exception:
            continue
    return None


def collect_measurements(input_dir):
    """
    Collect all cell_measurements.csv from t{N}/ subfolders.

    Returns:
        DataFrame with all cells, plus columns:
          frame_index, timepoint_min, timepoint_hr
    """
    input_path = Path(input_dir)
    period_ms = get_frame_period_ms(input_dir)
    if period_ms is not None:
        print(f"  Frame period: {period_ms/1000:.1f} s ({period_ms/60000:.2f} min)")
    else:
        print("  WARNING: could not read frame period from metadata; using frame index as time")

    all_data = []
    frame_dirs = sorted(
        [(int(m.group(1)), d)
         for d in input_path.iterdir()
         if d.is_dir() and (m := re.fullmatch(r't(\d+)', d.name))],
        key=lambda x: x[0]
    )

    if not frame_dirs:
        raise ValueError(f"No t{{N}} folders found in {input_dir}")

    for frame_idx, folder in frame_dirs:
        csv_path = folder / 'cell_measurements.csv'
        if not csv_path.exists():
            print(f"  Skipping {folder.name}: no cell_measurements.csv")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Skipping {folder.name}: {e}")
            continue
        if df.empty:
            print(f"  Skipping {folder.name}: empty CSV")
            continue

        df['frame_index'] = frame_idx
        if period_ms is not None:
            df['timepoint_min'] = frame_idx * period_ms / 60000.0
            df['timepoint_hr'] = df['timepoint_min'] / 60.0
        else:
            df['timepoint_min'] = float(frame_idx)
            df['timepoint_hr'] = float(frame_idx)

        all_data.append(df)
        print(f"  t{frame_idx:>3d}: {len(df)} cells  ({df['timepoint_min'].iloc[0]:.1f} min)")

    if not all_data:
        raise ValueError("No data loaded.")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal cells: {len(combined)}  across {len(all_data)} frames")
    return combined


def filter_saturated(df):
    """Remove saturated cells."""
    sat_cols = [c for c in df.columns if 'saturated' in c.lower()]
    if not sat_cols:
        return df
    mask = pd.Series(True, index=df.index)
    for col in sat_cols:
        mask &= (df[col] == 0)
    n_removed = (~mask).sum()
    if n_removed:
        print(f"  Removed {n_removed} saturated cells ({100*n_removed/len(df):.1f}%)")
    return df[mask].copy()


def load_run_log(run_log_path):
    """
    Load CellONIX experiment log (run_data.csv).

    Expected columns: time_s, temperature_C
    Returns DataFrame with time_min and temperature_C.
    """
    df = pd.read_csv(run_log_path)
    df['time_min'] = df['time_s'] / 60.0
    return df[['time_min', 'temperature_C']].dropna().reset_index(drop=True)


# =============================================================================
# Per-frame stats
# =============================================================================

def compute_frame_stats(df, value_cols, percentiles=(10, 25, 50, 75, 90)):
    """Compute per-frame percentiles and cell count for each value column."""
    rows = []
    for (frame_idx, tp_min, tp_hr), grp in df.groupby(['frame_index', 'timepoint_min', 'timepoint_hr']):
        row = {'frame_index': frame_idx, 'timepoint_min': tp_min, 'timepoint_hr': tp_hr,
               'n_cells': len(grp)}
        for col in value_cols:
            vals = grp[col].dropna()
            for p in percentiles:
                row[f'{col}_p{p}'] = np.percentile(vals, p) if len(vals) > 0 else np.nan
            row[f'{col}_mean'] = vals.mean() if len(vals) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values('timepoint_min').reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================

def _add_bottom_panel(ax_bot, stats_df, run_log_df):
    """
    Populate the bottom panel with temperature (left y) and cell count (right y).
    Both share the same x-axis as the top panel via sharex=True.
    """
    ax_bot.plot(run_log_df['time_min'], run_log_df['temperature_C'],
                color='crimson', linewidth=1, label='Temperature')
    ax_bot.set_ylabel('Temp (°C)', color='crimson', fontsize=10)
    ax_bot.tick_params(axis='y', labelcolor='crimson')
    ax_bot.grid(True, alpha=0.3)

    ax_count = ax_bot.twinx()
    ax_count.plot(stats_df['timepoint_min'], stats_df['n_cells'],
                  color='darkorange', marker='o', markersize=3, linewidth=1.5,
                  label='Cell count')
    ax_count.set_ylabel('Cell count', color='darkorange', fontsize=10)
    ax_count.tick_params(axis='y', labelcolor='darkorange')

    ax_bot.set_xlabel('Time (min)', fontsize=12)

    lines1, labels1 = ax_bot.get_legend_handles_labels()
    lines2, labels2 = ax_count.get_legend_handles_labels()
    ax_bot.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')


def plot_time_evolution(stats_df, raw_df, value_col, output_path, title_cols=None,
                        log_scale=False, run_log_df=None):
    """
    Time evolution plot: scatter of individual cells (transparent) +
    median line with 25-75 percentile shaded band.

    If run_log_df is provided, a second panel below shows temperature and cell
    count over time, sharing the same x-axis.
    """
    has_bottom = run_log_df is not None

    if has_bottom:
        fig, (ax, ax_bot) = plt.subplots(
            2, 1, figsize=(12, 7),
            gridspec_kw={'height_ratios': [3, 1.5]},
            sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=(12, 5))

    x_raw = raw_df['timepoint_min'].values
    y_raw = raw_df[value_col].values
    ax.scatter(x_raw, y_raw, color='steelblue', alpha=0.15, s=4, edgecolors='none', label='cells')

    x = stats_df['timepoint_min'].values
    median = stats_df[f'{value_col}_p50'].values
    p25 = stats_df[f'{value_col}_p25'].values
    p75 = stats_df[f'{value_col}_p75'].values

    ax.plot(x, median, color='navy', linewidth=2, label='median')
    ax.fill_between(x, p25, p75, color='steelblue', alpha=0.35, label='25–75%')

    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    ax.set_ylabel(col_label, fontsize=12)
    ax.set_title(col_label, fontsize=14)
    if log_scale:
        ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    if not has_bottom:
        ax.set_xlabel('Time (min)', fontsize=12)

    # Secondary x-axis in hours on the top panel
    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) / 60)
    ax2.set_xlabel('Time (hr)', fontsize=11)

    if has_bottom:
        _add_bottom_panel(ax_bot, stats_df, run_log_df)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_cell_count(stats_df, output_path):
    """Plot number of detected cells per frame over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(stats_df['timepoint_min'], stats_df['n_cells'], marker='o', markersize=3,
            linewidth=1.5, color='darkorange')
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Number of cells', fontsize=12)
    ax.set_title('Cell count over time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twiny()
    ax2.set_xlim(np.array(ax.get_xlim()) / 60)
    ax2.set_xlabel('Time (hr)', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_histogram_ridge(df, value_col, output_path, n_bins=40, title_cols=None):
    """
    Ridge plot: one row per timepoint, stacked vertically, showing distribution shape.
    """
    timepoints = sorted(df['timepoint_min'].unique())
    all_vals = df[value_col].dropna()

    # Use linear bins (not log) across the data range
    vmin, vmax = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    bin_edges = np.linspace(vmin, vmax, n_bins + 1)

    rows = []
    for tp in timepoints:
        vals = df[df['timepoint_min'] == tp][value_col].dropna()
        counts, _ = np.histogram(vals, bins=bin_edges)
        fracs = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
        label = f'{tp:.0f} min'
        rows.append((label, fracs))

    n_rows = len(rows)
    max_h = max(h.max() for _, h in rows) if rows else 1
    spacing = max_h * 0.8

    colors = plt.cm.viridis(np.linspace(0.1, 0.95, n_rows))
    fig, ax = plt.subplots(figsize=(12, max(6, 2 + 0.25 * n_rows)))

    for i, (label, fracs) in enumerate(rows):
        y_base = (n_rows - 1 - i) * spacing
        color = colors[i]
        for j in range(n_bins):
            ax.bar(bin_edges[j], fracs[j], width=bin_edges[j+1]-bin_edges[j],
                   bottom=y_base, align='edge', color=color, edgecolor='none', alpha=0.85)
        ax.axhline(y_base, color='black', linewidth=0.3, alpha=0.4)

    col_label = (title_cols or {}).get(value_col, value_col.replace('_', ' ').title())
    ax.set_xlabel(col_label, fontsize=12)
    ax.set_title(f'{col_label} distribution over time', fontsize=14)
    ax.set_yticks([(n_rows - 1 - i) * spacing for i in range(n_rows)])
    ax.set_yticklabels([label for label, _ in rows], fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_variability(stats_df, value_cols, output_path, title_cols=None):
    """Plot CV and robust spread (p90-p10)/median over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = plt.cm.tab10.colors
    x = stats_df['timepoint_min'].values

    for i, col in enumerate(value_cols):
        col_label = (title_cols or {}).get(col, col.replace('_', ' ').title())
        p10 = stats_df[f'{col}_p10'].values
        p50 = stats_df[f'{col}_p50'].values
        p90 = stats_df[f'{col}_p90'].values
        mean = stats_df[f'{col}_mean'].values

        with np.errstate(invalid='ignore', divide='ignore'):
            cv = np.where(mean != 0, np.abs(stats_df[f'{col}_p50'].values - mean) / np.abs(mean), np.nan)
            # Use std from raw for CV: approximate via IQR/1.35
            iqr = stats_df[f'{col}_p75'].values - stats_df[f'{col}_p25'].values
            cv_robust = np.where(p50 != 0, (p90 - p10) / np.abs(p50), np.nan)

        axes[0].plot(x, iqr / np.abs(mean + 1e-10), marker='o', markersize=3,
                     linewidth=1.5, label=col_label, color=colors[i % len(colors)])
        axes[1].plot(x, cv_robust, marker='o', markersize=3,
                     linewidth=1.5, label=col_label, color=colors[i % len(colors)])

    axes[0].set_ylabel('IQR / |mean|', fontsize=11)
    axes[0].set_title('Variability over time', fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel('(p90−p10) / |median|', fontsize=11)
    axes[1].set_xlabel('Time (min)', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect and analyze CellASIC time-series segmentation results')
    parser.add_argument('--input', '-i', required=True,
                        help='Directory containing t0/, t1/, ... subfolders')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory (default: <input>_analysis)')
    parser.add_argument('--log-scale', action='store_true',
                        help='Use log scale for time evolution plots')
    parser.add_argument('--run-log', default=None,
                        help='CellONIX run_data.csv with temperature log; '
                             'adds a temperature + cell-count panel below each time evolution plot')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / (input_path.name + '_analysis')
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_path}")

    # Load optional run log
    run_log_df = None
    if args.run_log:
        print(f"\nLoading run log: {args.run_log}")
        run_log_df = load_run_log(args.run_log)
        t_range = run_log_df['time_min'].max()
        print(f"  {len(run_log_df)} entries, {t_range:.0f} min total "
              f"({t_range/60:.1f} hr), "
              f"T = {run_log_df['temperature_C'].min():.1f}–{run_log_df['temperature_C'].max():.1f} °C")

    # Collect data
    print("\nCollecting measurements...")
    df = collect_measurements(args.input)

    # Filter saturated
    print("\nFiltering saturated cells...")
    df = filter_saturated(df)

    # Save merged
    merged_path = output_path / 'all_cells.csv'
    df.to_csv(merged_path, index=False)
    print(f"  Saved: all_cells.csv ({len(df)} cells)")

    # Value columns present in data
    candidate_cols = ['mcherry_rate', 'gfp_rate', 'gfp_mcherry_ratio']
    value_cols = [c for c in candidate_cols if c in df.columns]
    title_cols = {
        'mcherry_rate': 'mCherry rate (a.u./ms)',
        'gfp_rate': 'GFP rate (a.u./ms)',
        'gfp_mcherry_ratio': 'GFP/mCherry ratio',
    }
    print(f"\nValue columns: {value_cols}")

    # Per-frame stats
    print("\nComputing per-frame statistics...")
    stats_df = compute_frame_stats(df, value_cols)
    stats_df.to_csv(output_path / 'frame_stats.csv', index=False)
    print(f"  Saved: frame_stats.csv")

    # Plots
    print("\nGenerating plots...")

    plot_cell_count(stats_df, output_path / 'cell_count.png')

    for col in value_cols:
        plot_time_evolution(stats_df, df, col,
                            output_path / f'{col}_time_evolution.png',
                            title_cols=title_cols,
                            log_scale=True,
                            run_log_df=run_log_df)

    for col in value_cols:
        plot_histogram_ridge(df, col,
                             output_path / f'{col}_ridge.png',
                             title_cols=title_cols)

    plot_variability(stats_df, value_cols,
                     output_path / 'variability.png',
                     title_cols=title_cols)

    print("\nDone!")


if __name__ == '__main__':
    main()
