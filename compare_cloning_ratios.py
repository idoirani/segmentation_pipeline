#!/usr/bin/env python3
"""
Compare GFP/mCherry ratios across multiple cloning experiments.
Normalizes intensities to 1 second exposure time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import nd2

# Configuration
OUTPUT_DIR = Path('./segmentation_output')
INPUT_DIR = Path('./cloning_wt')

# Channel indices
GFP_CHANNEL = 1
MCHERRY_CHANNEL = 2

DATASETS = {
    'Image 1 (1_in_wt)': {
        'dir': 'cloning_290126_fluor_omni',
        'source': INPUT_DIR / '1_in_wt' / '290126_z_gfpmcharry_ph3.nd2',
        'exposure_ms': {'Ph3': 200, 'GFP': 1000, 'mCherry': 1000},
    },
    'Image 2 (2_in_wt)': {
        'dir': 'cloning_290126_fluor_omni2',
        'source': INPUT_DIR / '2_in_wt' / '290126_z_gfpmcharry_ph3002.nd2',
        'exposure_ms': {'Ph3': 200, 'GFP': 1000, 'mCherry': 1000},
    },
    'Image 3 (3_in_wt)': {
        'dir': 'cloning_290126_fluor_omni3',
        'source': INPUT_DIR / '3_in_wt' / '290126_z_gfpmcharry_ph3003.nd2',
        'exposure_ms': {'Ph3': 200, 'GFP': 1000, 'mCherry': 100},  # Note: 100ms mCherry
    },
}

# Target exposure for normalization (1 second = 1000 ms)
TARGET_EXPOSURE_MS = 1000


def estimate_background(dataset_info):
    """
    Estimate background intensity from pixels outside segmentation masks.

    Returns median pixel values for GFP and mCherry outside the masks,
    normalized to 1 second exposure.
    """
    # Load mask
    mask_path = OUTPUT_DIR / dataset_info['dir'] / 'masks.tif'
    masks = tifffile.imread(mask_path)

    # Load z-selection to get best z
    z_csv = OUTPUT_DIR / dataset_info['dir'] / 'z_selection.csv'
    z_df = pd.read_csv(z_csv)
    best_z = z_df[z_df['selected'] == True]['z'].iloc[0]

    # Load source image
    with nd2.ND2File(dataset_info['source']) as f:
        data = f.asarray()

    # Get images at best z
    gfp_image = data[best_z, GFP_CHANNEL, :, :]
    mcherry_image = data[best_z, MCHERRY_CHANNEL, :, :]

    # Background = pixels where mask == 0
    background_mask = masks == 0

    gfp_bg_median = np.median(gfp_image[background_mask])
    mcherry_bg_median = np.median(mcherry_image[background_mask])

    # Normalize to 1 second exposure
    gfp_factor = TARGET_EXPOSURE_MS / dataset_info['exposure_ms']['GFP']
    mcherry_factor = TARGET_EXPOSURE_MS / dataset_info['exposure_ms']['mCherry']

    return {
        'gfp_bg_per_pixel': gfp_bg_median * gfp_factor,
        'mcherry_bg_per_pixel': mcherry_bg_median * mcherry_factor,
    }


def load_and_normalize(dataset_name, dataset_info):
    """Load cell measurements and normalize to 1 second exposure."""
    csv_path = OUTPUT_DIR / dataset_info['dir'] / 'cell_measurements.csv'
    df = pd.read_csv(csv_path)

    # Normalize intensities to 1 second exposure
    gfp_factor = TARGET_EXPOSURE_MS / dataset_info['exposure_ms']['GFP']
    mcherry_factor = TARGET_EXPOSURE_MS / dataset_info['exposure_ms']['mCherry']

    df['gfp_sum_norm'] = df['gfp_sum'] * gfp_factor
    df['mcherry_sum_norm'] = df['mcherry_sum'] * mcherry_factor
    df['gfp_mean_norm'] = df['gfp_mean'] * gfp_factor
    df['mcherry_mean_norm'] = df['mcherry_mean'] * mcherry_factor

    # Recalculate ratio with normalized values
    df['ratio_norm'] = df['gfp_sum_norm'] / df['mcherry_sum_norm']

    df['dataset'] = dataset_name

    return df


def main():
    # Load all datasets and estimate backgrounds
    all_data = []
    backgrounds = {}
    for name, info in DATASETS.items():
        df = load_and_normalize(name, info)
        all_data.append(df)
        bg = estimate_background(info)
        backgrounds[name] = bg
        print(f"Loaded {name}: {len(df)} cells")
        print(f"  Normalization factors: GFP={TARGET_EXPOSURE_MS/info['exposure_ms']['GFP']:.1f}x, "
              f"mCherry={TARGET_EXPOSURE_MS/info['exposure_ms']['mCherry']:.1f}x")
        print(f"  Background (per pixel): GFP={bg['gfp_bg_per_pixel']:.1f}, mCherry={bg['mcherry_bg_per_pixel']:.1f}")

    combined_df = pd.concat(all_data, ignore_index=True)

    # Colors for each dataset
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    dataset_names = list(DATASETS.keys())

    # Figure 1: Scatter plots - Area vs Intensity for each image
    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 9))
    fig1.suptitle('Cell Area vs Integrated Intensity (Normalized to 1s exposure)', fontsize=14)

    for i, (name, color) in enumerate(zip(dataset_names, colors)):
        df_sub = combined_df[combined_df['dataset'] == name]
        bg = backgrounds[name]

        # Area range for background line
        area_range = np.linspace(df_sub['area_pixels'].min(), df_sub['area_pixels'].max(), 100)

        # GFP (semi-log: log y-axis)
        axes1[0, i].scatter(df_sub['area_pixels'], df_sub['gfp_sum_norm'],
                           alpha=0.6, c=color, s=30, label='Cells')
        # Background line: integrated_bg = bg_per_pixel * area
        gfp_bg_line = bg['gfp_bg_per_pixel'] * area_range
        axes1[0, i].plot(area_range, gfp_bg_line, 'k--', alpha=0.7,
                        label=f"Background ({bg['gfp_bg_per_pixel']:.0f}/px)")
        axes1[0, i].set_xlabel('Cell Area (pixels)')
        axes1[0, i].set_ylabel('GFP Integrated Intensity')
        axes1[0, i].set_title(f'{name}\nGFP (n={len(df_sub)})')
        axes1[0, i].set_yscale('log')
        axes1[0, i].legend(fontsize=8)

        # mCherry (semi-log: log y-axis)
        axes1[1, i].scatter(df_sub['area_pixels'], df_sub['mcherry_sum_norm'],
                           alpha=0.6, c=color, s=30, label='Cells')
        # Background line
        mcherry_bg_line = bg['mcherry_bg_per_pixel'] * area_range
        axes1[1, i].plot(area_range, mcherry_bg_line, 'k--', alpha=0.7,
                        label=f"Background ({bg['mcherry_bg_per_pixel']:.0f}/px)")
        axes1[1, i].set_xlabel('Cell Area (pixels)')
        axes1[1, i].set_ylabel('mCherry Integrated Intensity')
        axes1[1, i].set_title(f'mCherry (n={len(df_sub)})')
        axes1[1, i].set_yscale('log')
        axes1[1, i].legend(fontsize=8)

    plt.tight_layout()
    fig1.savefig(OUTPUT_DIR / 'comparison_area_vs_intensity.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_DIR / 'comparison_area_vs_intensity.png'}")

    # Figure 2: GFP vs mCherry scatter for each image
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
    fig2.suptitle('GFP vs mCherry Integrated Intensity (Normalized to 1s exposure)', fontsize=14)

    for i, (name, color) in enumerate(zip(dataset_names, colors)):
        df_sub = combined_df[combined_df['dataset'] == name]

        axes2[i].scatter(df_sub['mcherry_sum_norm'], df_sub['gfp_sum_norm'],
                        alpha=0.6, c=color, s=30)
        axes2[i].set_xlabel('mCherry Integrated Intensity')
        axes2[i].set_ylabel('GFP Integrated Intensity')
        axes2[i].set_title(f'{name}\n(n={len(df_sub)})')
        axes2[i].ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        # Add diagonal line for reference (ratio=1)
        max_val = max(df_sub['mcherry_sum_norm'].max(), df_sub['gfp_sum_norm'].max())
        axes2[i].plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Ratio=1')
        axes2[i].legend()

    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'comparison_gfp_vs_mcherry.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'comparison_gfp_vs_mcherry.png'}")

    # Figure 3: Summary histograms
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle('Intensity Distributions Across Images (Normalized to 1s exposure)', fontsize=14)

    # Determine common bins for absolute values
    gfp_all = combined_df['gfp_sum_norm']
    mcherry_all = combined_df['mcherry_sum_norm']

    gfp_bins = np.linspace(gfp_all.min(), gfp_all.max(), 15)
    mcherry_bins = np.linspace(mcherry_all.min(), mcherry_all.max(), 15)

    # Panel A: GFP absolute distribution (log scale)
    log_gfp_all = np.log10(gfp_all[gfp_all > 0])
    gfp_log_bins = np.linspace(log_gfp_all.min(), log_gfp_all.max(), 15)
    for name, color in zip(dataset_names, colors):
        df_sub = combined_df[combined_df['dataset'] == name]
        log_gfp = np.log10(df_sub['gfp_sum_norm'][df_sub['gfp_sum_norm'] > 0])
        axes3[0, 0].hist(log_gfp, bins=gfp_log_bins, alpha=0.5,
                        label=name, color=color, density=True)
    axes3[0, 0].set_xlabel('log10(GFP Integrated Intensity)')
    axes3[0, 0].set_ylabel('Density')
    axes3[0, 0].set_title('GFP Distribution (Log Scale)')
    axes3[0, 0].legend(fontsize=8)

    # Panel B: mCherry absolute distribution (log scale)
    log_mcherry_all = np.log10(mcherry_all[mcherry_all > 0])
    mcherry_log_bins = np.linspace(log_mcherry_all.min(), log_mcherry_all.max(), 15)
    for name, color in zip(dataset_names, colors):
        df_sub = combined_df[combined_df['dataset'] == name]
        log_mcherry = np.log10(df_sub['mcherry_sum_norm'][df_sub['mcherry_sum_norm'] > 0])
        axes3[0, 1].hist(log_mcherry, bins=mcherry_log_bins, alpha=0.5,
                        label=name, color=color, density=True)
    axes3[0, 1].set_xlabel('log10(mCherry Integrated Intensity)')
    axes3[0, 1].set_ylabel('Density')
    axes3[0, 1].set_title('mCherry Distribution (Log Scale)')
    axes3[0, 1].legend(fontsize=8)

    # Panel C: GFP normalized to mean (log scale)
    log_norm_bins = np.linspace(-1, 1, 15)  # log10(0.1) to log10(10)
    for name, color in zip(dataset_names, colors):
        df_sub = combined_df[combined_df['dataset'] == name]
        gfp_norm_to_mean = df_sub['gfp_sum_norm'] / df_sub['gfp_sum_norm'].mean()
        log_gfp_norm = np.log10(gfp_norm_to_mean[gfp_norm_to_mean > 0])
        axes3[1, 0].hist(log_gfp_norm, bins=log_norm_bins, alpha=0.5,
                        label=f"{name} (mean={df_sub['gfp_sum_norm'].mean():.2e})",
                        color=color, density=True)
    axes3[1, 0].set_xlabel('log10(GFP / Mean(GFP))')
    axes3[1, 0].set_ylabel('Density')
    axes3[1, 0].set_title('GFP Distribution (Normalized to Mean, Log Scale)')
    axes3[1, 0].legend(fontsize=8)
    axes3[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)  # log10(1) = 0

    # Panel D: mCherry normalized to mean (log scale)
    for name, color in zip(dataset_names, colors):
        df_sub = combined_df[combined_df['dataset'] == name]
        mcherry_norm_to_mean = df_sub['mcherry_sum_norm'] / df_sub['mcherry_sum_norm'].mean()
        log_mcherry_norm = np.log10(mcherry_norm_to_mean[mcherry_norm_to_mean > 0])
        axes3[1, 1].hist(log_mcherry_norm, bins=log_norm_bins, alpha=0.5,
                        label=f"{name} (mean={df_sub['mcherry_sum_norm'].mean():.2e})",
                        color=color, density=True)
    axes3[1, 1].set_xlabel('log10(mCherry / Mean(mCherry))')
    axes3[1, 1].set_ylabel('Density')
    axes3[1, 1].set_title('mCherry Distribution (Normalized to Mean, Log Scale)')
    axes3[1, 1].legend(fontsize=8)
    axes3[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)  # log10(1) = 0

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / 'comparison_intensity_histograms.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'comparison_intensity_histograms.png'}")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (Normalized to 1s exposure)")
    print("="*70)

    for name in dataset_names:
        df_sub = combined_df[combined_df['dataset'] == name]
        print(f"\n{name}:")
        print(f"  Cells: {len(df_sub)}")
        print(f"  GFP mean:     {df_sub['gfp_sum_norm'].mean():.2e} +/- {df_sub['gfp_sum_norm'].std():.2e}")
        print(f"  mCherry mean: {df_sub['mcherry_sum_norm'].mean():.2e} +/- {df_sub['mcherry_sum_norm'].std():.2e}")
        print(f"  Ratio (median): {df_sub['ratio_norm'].median():.2f}")

    # plt.show()  # Uncomment for interactive use


if __name__ == '__main__':
    main()
