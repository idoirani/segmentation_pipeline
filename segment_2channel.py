#!/usr/bin/env python3
"""
Segmentation script for 2-channel images (GFP + Ph3) with single z-frame.

Based on segment_cloning.py but simplified for:
- 2 channels only (GFP and Ph3)
- Single z-frame (no z-selection)
- Segmentation on Ph3 channel
- GFP intensity measurements only
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
import tifffile
import nd2
import cv2

from cellpose_omni import models as omni_models
from segment_cloning import (extract_nd2_metadata, save_nd2_metadata, extract_exposure_times,
                              find_optimal_channel_shift, apply_mask_shift)
from autofocus import select_best_z_fluorescence

# Channel indices (may vary by file)
GFP_CHANNEL = 1
PH3_CHANNEL = 0

# Omnipose parameters
CELLPOSE_PARAMS = {
    'model_type': 'bact_phase_omni',
    'diameter': 30,
    'channels': [0, 0],  # grayscale
    'flow_threshold': 0.4,
    'mask_threshold': -2,
    'min_size': 250,
    'omni': True,
    'cluster': True,
}

# Area filtering
MAX_CELL_AREA = 1100  # pixels

# Visualization settings
VIS_DPI = 150


def load_nd2_2channel(filepath, time_index=None):
    """Load nd2 file with 2 channels and optional time series.

    Returns:
        data: Array with shape (C, Y, X) or (Z, C, Y, X) if z-stack
        sizes: Dict of dimension sizes
        channels: List of channel names
        has_z: True if file has a Z dimension
    """
    print(f"Loading {filepath}...")
    with nd2.ND2File(filepath) as f:
        data = f.asarray()
        sizes = f.sizes
        try:
            channels = [c.channel.name for c in f.metadata.channels]
        except:
            channels = ['Ch0', 'Ch1']

        # Handle time series
        if 'T' in sizes:
            n_timepoints = sizes['T']
            if time_index is None:
                time_index = 0
            print(f"  Time series detected: {n_timepoints} timepoints")
            print(f"  Extracting frame T={time_index}")
            data = data[time_index]
            sizes = {k: v for k, v in sizes.items() if k != 'T'}

        has_z = 'Z' in sizes and sizes['Z'] > 1
        if has_z:
            print(f"  Z-stack detected: {sizes['Z']} slices")

        print(f"  Shape: {data.shape}")
        print(f"  Channels: {channels}")

    return data, sizes, channels, has_z


def segment_frame(model, image, params):
    """Segment a single frame using Omnipose."""
    masks, flows, styles = model.eval(
        image,
        diameter=params['diameter'],
        channels=params['channels'],
        flow_threshold=params.get('flow_threshold', 0.4),
        mask_threshold=params.get('mask_threshold', -2),
        min_size=params.get('min_size', 200),
        omni=params.get('omni', True),
        cluster=params.get('cluster', True),
    )
    return masks


def measure_cells_gfp_only(masks, gfp_image, saturation_value=65535):
    """
    Measure GFP intensity for each cell.

    Args:
        masks: Segmentation masks
        gfp_image: GFP fluorescence image
        saturation_value: Pixel value indicating saturation (default: 65535 for 16-bit)

    Returns a DataFrame with per-cell measurements including gfp_saturated flag.
    """
    from skimage.measure import regionprops_table

    if masks.max() == 0:
        return pd.DataFrame()

    props = regionprops_table(
        masks,
        intensity_image=gfp_image,
        properties=['label', 'area', 'centroid', 'intensity_mean']
    )

    df = pd.DataFrame(props)
    df.columns = ['cell_label', 'area_pixels', 'centroid_y', 'centroid_x', 'gfp_mean']

    # Calculate additional metrics
    gfp_sums = []
    gfp_maxs = []
    gfp_stds = []
    gfp_saturated = []

    for label in df['cell_label']:
        mask = masks == label
        gfp_vals = gfp_image[mask]

        gfp_sums.append(gfp_vals.sum())
        gfp_maxs.append(gfp_vals.max())
        gfp_stds.append(gfp_vals.std())
        gfp_saturated.append(1 if np.any(gfp_vals >= saturation_value) else 0)

    df['gfp_sum'] = gfp_sums
    df['gfp_max'] = gfp_maxs
    df['gfp_std'] = gfp_stds
    df['gfp_saturated'] = gfp_saturated

    # Reorder columns
    df = df[['cell_label', 'area_pixels', 'centroid_x', 'centroid_y',
             'gfp_sum', 'gfp_mean', 'gfp_max', 'gfp_std', 'gfp_saturated']]

    return df


def save_segmentation_visualization_2ch(ph3_image, gfp_bkg_sub, masks, n_cells, output_path,
                                         best_z=None, best_z_gfp=None,
                                         z_scores=None, z_max_gfp=None):
    """Save segmentation visualization for 2-channel data with optional focus info.

    Args:
        ph3_image: Phase contrast image (2D)
        gfp_bkg_sub: Background-subtracted GFP image (may contain negative values)
        masks: Segmentation masks
        n_cells: Number of cells detected
        output_path: Path to save the figure
        best_z: Best z-slice for Ph3 (segmentation)
        best_z_gfp: Best z-slice for GFP (quantification)
        z_scores: Focus scores per z-slice (for Ph3 autofocus)
        z_max_gfp: Max GFP in masks per z-slice
    """
    from skimage.segmentation import find_boundaries

    # Display ranges for bkg-subtracted GFP
    cell_mask_bool = masks > 0
    gfp_vmin = np.percentile(gfp_bkg_sub, 1)
    if np.any(cell_mask_bool):
        gfp_vmax = np.percentile(gfp_bkg_sub[cell_mask_bool], 99)
    else:
        gfp_vmax = np.percentile(gfp_bkg_sub, 99)
    if gfp_vmax <= gfp_vmin:
        gfp_vmax = gfp_vmin + 1

    has_focus = best_z is not None and z_scores is not None
    n_cols = 4 if has_focus else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

    # Row 1: Raw images
    z_ph3_label = f' (z={best_z})' if best_z is not None else ''
    z_gfp_label = f' (z={best_z_gfp})' if best_z_gfp is not None else z_ph3_label
    axes[0, 0].imshow(ph3_image, cmap='gray',
                      vmin=np.percentile(ph3_image, 1),
                      vmax=np.percentile(ph3_image, 99))
    axes[0, 0].set_title(f'Phase Contrast{z_ph3_label}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gfp_bkg_sub, cmap='Greens',
                      vmin=gfp_vmin, vmax=gfp_vmax)
    axes[0, 1].set_title(f'GFP bkg-sub{z_gfp_label}')
    axes[0, 1].axis('off')

    # GFP green channel
    gfp_norm = np.clip(gfp_bkg_sub / (gfp_vmax + 1e-10), 0, 1)
    composite = np.zeros((*gfp_bkg_sub.shape, 3))
    composite[:, :, 1] = gfp_norm
    axes[0, 2].imshow(composite)
    axes[0, 2].set_title('GFP bkg-sub (green)')
    axes[0, 2].axis('off')

    # Focus scores panel
    if has_focus:
        ax_focus = axes[0, 3]
        z_indices = range(len(z_scores))
        scores_norm = np.array(z_scores) / max(z_scores) if max(z_scores) > 0 else z_scores
        ax_focus.bar(z_indices, scores_norm, color='steelblue', alpha=0.8,
                     label='Ph3 focus')
        ax_focus.axvline(best_z, color='red', linestyle='--', linewidth=2,
                         label=f'Ph3 z={best_z}')
        # Overlay GFP max in masks if available
        if z_max_gfp is not None:
            ax2 = ax_focus.twinx()
            gfp_max_arr = np.array(z_max_gfp)
            gfp_norm_scores = gfp_max_arr / max(gfp_max_arr) if max(gfp_max_arr) > 0 else gfp_max_arr
            ax2.plot(z_indices, gfp_norm_scores, 'g-o', linewidth=2, markersize=5,
                     label='GFP max in masks')
            if best_z_gfp is not None:
                ax2.axvline(best_z_gfp, color='green', linestyle=':', linewidth=2,
                            label=f'GFP z={best_z_gfp}')
            ax2.set_ylabel('Norm. GFP max', color='green', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.set_ylim(0, 1.1)
            ax2.legend(loc='upper left', fontsize=7)
        ax_focus.set_xlabel('Z-slice')
        ax_focus.set_ylabel('Norm. Focus Score')
        ax_focus.set_title('Autofocus')
        ax_focus.legend(loc='upper right', fontsize=7)
        ax_focus.set_ylim(0, 1.1)

    # Row 2: Segmentation results
    boundaries = find_boundaries(masks, mode='outer')

    # Ph3 with outlines
    ph3_disp = (ph3_image - ph3_image.min()) / (ph3_image.max() - ph3_image.min() + 1e-10)
    ph3_overlay = np.stack([ph3_disp] * 3, axis=-1)
    ph3_overlay[boundaries] = [1, 0, 0]
    axes[1, 0].imshow(ph3_overlay)
    axes[1, 0].set_title(f'Ph3 + Segmentation ({n_cells} cells)')
    axes[1, 0].axis('off')

    # GFP with outlines
    gfp_overlay = np.stack([gfp_norm] * 3, axis=-1)
    gfp_overlay[boundaries] = [1, 0, 0]
    axes[1, 1].imshow(gfp_overlay)
    axes[1, 1].set_title('GFP bkg-sub + Segmentation')
    axes[1, 1].axis('off')

    # Masks
    axes[1, 2].imshow(masks, cmap='nipy_spectral')
    axes[1, 2].set_title('Cell Masks')
    axes[1, 2].axis('off')

    # Info panel
    if has_focus:
        axes[1, 3].axis('off')
        info_text = f'Ph3 z: {best_z}\nCells: {n_cells}\nZ-slices: {len(z_scores)}'
        if best_z_gfp is not None:
            info_text += f'\nGFP z: {best_z_gfp}'
        axes[1, 3].text(0.5, 0.5, info_text,
                        ha='center', va='center', fontsize=14,
                        transform=axes[1, 3].transAxes)

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_z_crop_visualization_2ch(z_stack, masks, best_z, output_path,
                                   gfp_channel=0, ph3_channel=1, crop_size=100):
    """Save visualization showing a cell crop across all z-slices for both channels."""
    n_z = z_stack.shape[0]
    img_h, img_w = z_stack.shape[2], z_stack.shape[3]

    # Find a cell near the center with typical bacterial size
    cell_labels = np.unique(masks)
    cell_labels = cell_labels[cell_labels != 0]

    if len(cell_labels) == 0:
        print("    No cells found, skipping z-crop visualization")
        return

    center_y, center_x = img_h // 2, img_w // 2
    min_cell_area, max_cell_area = 100, 800
    all_cells = []

    for label in cell_labels:
        cell_mask = masks == label
        y_coords, x_coords = np.where(cell_mask)
        area = len(y_coords)
        if min_cell_area <= area <= max_cell_area:
            cy, cx = np.mean(y_coords), np.mean(x_coords)
            dist = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
            all_cells.append({'label': label, 'cy': cy, 'cx': cx, 'area': area, 'dist': dist})

    if len(all_cells) == 0:
        print(f"    No cells in size range ({min_cell_area}-{max_cell_area} px)")
        return

    all_cells.sort(key=lambda c: c['dist'])
    selected = all_cells[np.random.randint(0, min(10, len(all_cells)))]
    cy, cx = selected['cy'], selected['cx']

    half = crop_size // 2
    y_start, y_end = int(max(0, cy - half)), int(min(img_h, cy + half))
    x_start, x_end = int(max(0, cx - half)), int(min(img_w, cx + half))

    # Sample z-slices if too many
    if n_z > 11:
        z_indices = np.linspace(0, n_z - 1, 11, dtype=int)
    else:
        z_indices = list(range(n_z))

    n_cols = len(z_indices)

    # Two rows: Ph3 and GFP
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Intensity range from best_z
    ph3_crop_best = z_stack[best_z, ph3_channel, y_start:y_end, x_start:x_end]
    gfp_crop_best = z_stack[best_z, gfp_channel, y_start:y_end, x_start:x_end]
    ph3_vmin, ph3_vmax = np.percentile(ph3_crop_best, 1), np.percentile(ph3_crop_best, 99)
    gfp_vmin, gfp_vmax = np.percentile(gfp_crop_best, 1), np.percentile(gfp_crop_best, 99)

    for i, z in enumerate(z_indices):
        # Ph3 row
        crop_ph3 = z_stack[z, ph3_channel, y_start:y_end, x_start:x_end]
        axes[0, i].imshow(crop_ph3, cmap='gray', vmin=ph3_vmin, vmax=ph3_vmax)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # GFP row
        crop_gfp = z_stack[z, gfp_channel, y_start:y_end, x_start:x_end]
        axes[1, i].imshow(crop_gfp, cmap='Greens', vmin=gfp_vmin, vmax=gfp_vmax)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        if z == best_z:
            for row in range(2):
                for spine in axes[row, i].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)
            axes[0, i].set_title(f'z={z}\n(SELECTED)', fontsize=10, fontweight='bold', color='red')
        else:
            axes[0, i].set_title(f'z={z}', fontsize=10)

    axes[0, 0].set_ylabel('Ph3', fontsize=12)
    axes[1, 0].set_ylabel('GFP', fontsize=12)

    plt.suptitle(f'Cell #{selected["label"]} (area={selected["area"]} px) across Z', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def select_best_z_gfp_in_masks(z_stack, masks, gfp_channel=0):
    """
    Select the z-slice where max(GFP) within cell masks is highest.

    After segmentation on the Ph3 channel at a given z, this scans all z-slices
    to find the one with the brightest GFP signal inside the segmented cells.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        masks: Segmentation mask from Ph3 channel (2D, Y x X)
        gfp_channel: GFP channel index

    Returns:
        best_z_gfp: Index of z-slice with highest max GFP in masks
        z_max_gfp: List of max GFP values per z-slice (within masks)
    """
    n_z = z_stack.shape[0]
    mask_bool = masks > 0
    z_max_gfp = []

    for z in range(n_z):
        gfp_slice = z_stack[z, gfp_channel, :, :]
        if np.any(mask_bool):
            max_val = float(gfp_slice[mask_bool].max())
        else:
            max_val = 0.0
        z_max_gfp.append(max_val)

    best_z_gfp = int(np.argmax(z_max_gfp))
    return best_z_gfp, z_max_gfp


def _robust_median_in_region(gfp_image, masks, y_start, y_end, x_start, x_end, iqr_factor=1.5):
    """Compute robust median of background pixels in a rectangular region."""
    region_gfp = gfp_image[y_start:y_end, x_start:x_end]
    region_masks = masks[y_start:y_end, x_start:x_end]
    bg_pixels = region_gfp[region_masks == 0].astype(np.float64)
    if len(bg_pixels) == 0:
        return np.nan
    q1 = np.percentile(bg_pixels, 25)
    q3 = np.percentile(bg_pixels, 75)
    iqr = q3 - q1
    inliers = bg_pixels[(bg_pixels >= q1 - iqr_factor * iqr) &
                         (bg_pixels <= q3 + iqr_factor * iqr)]
    if len(inliers) == 0:
        return np.median(bg_pixels)
    return np.median(inliers)


def calculate_background(gfp_image, masks, iqr_factor=1.5):
    """
    Calculate a 2D quadratic background model from a 4x4 grid of regions.

    1. Divide image into 4x4 regions (16 tiles)
    2. For each region: take pixels outside masks, IQR outlier removal, median
    3. Fit a 2D quadratic surface bkg(y, x) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    4. Generate full-resolution 2D background image

    Returns:
        dict with 'bkg_image' (2D array), 'median' (scalar, center value),
        'mean', 'std', 'n_outliers', 'quadrant_values', 'fit_coeffs'
    """
    h, w = gfp_image.shape
    n_grid = 4

    # Region boundaries
    y_edges = np.linspace(0, h, n_grid + 1, dtype=int)
    x_edges = np.linspace(0, w, n_grid + 1, dtype=int)

    # Estimate background in each region
    quad_centers_y = []
    quad_centers_x = []
    quad_values = []

    for iy in range(n_grid):
        for ix in range(n_grid):
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            cy = (y0 + y1) / 2.0
            cx = (x0 + x1) / 2.0

            med = _robust_median_in_region(gfp_image, masks, y0, y1, x0, x1, iqr_factor)
            quad_centers_y.append(cy)
            quad_centers_x.append(cx)
            quad_values.append(med)

    quad_centers_y = np.array(quad_centers_y)
    quad_centers_x = np.array(quad_centers_x)
    quad_values = np.array(quad_values)

    # Drop any NaN regions (e.g. fully masked)
    valid = ~np.isnan(quad_values)
    if valid.sum() < 6:
        # Not enough points for a quadratic fit; fall back to global median
        bg_pixels = gfp_image[masks == 0].astype(np.float64)
        q1, q3 = np.percentile(bg_pixels, 25), np.percentile(bg_pixels, 75)
        iqr = q3 - q1
        inliers = bg_pixels[(bg_pixels >= q1 - iqr_factor * iqr) &
                             (bg_pixels <= q3 + iqr_factor * iqr)]
        med = np.median(inliers) if len(inliers) > 0 else np.median(bg_pixels)
        bkg_image = np.full_like(gfp_image, med, dtype=np.float64)
        return {
            'bkg_image': bkg_image,
            'median': med,
            'mean': med,
            'std': 0.0,
            'n_outliers': 0,
            'quadrant_values': quad_values.tolist(),
            'fit_coeffs': (0.0, 0.0, 0.0, 0.0, 0.0, med),
        }

    # Fit 2D quadratic: bkg = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    xv = quad_centers_x[valid]
    yv = quad_centers_y[valid]
    A = np.column_stack([xv**2, yv**2, xv * yv, xv, yv, np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, quad_values[valid], rcond=None)

    # Generate full 2D background
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    bkg_image = (coeffs[0] * xx**2 + coeffs[1] * yy**2 + coeffs[2] * xx * yy +
                 coeffs[3] * xx + coeffs[4] * yy + coeffs[5])

    # Stats: residuals of region medians
    fitted_at_centers = (coeffs[0] * xv**2 + coeffs[1] * yv**2 + coeffs[2] * xv * yv +
                         coeffs[3] * xv + coeffs[4] * yv + coeffs[5])
    residuals = quad_values[valid] - fitted_at_centers
    cx, cy = w / 2.0, h / 2.0
    center_value = float(coeffs[0] * cx**2 + coeffs[1] * cy**2 + coeffs[2] * cx * cy +
                         coeffs[3] * cx + coeffs[4] * cy + coeffs[5])

    # Count outliers globally (for reporting)
    bg_pixels = gfp_image[masks == 0].astype(np.float64)
    q1, q3 = np.percentile(bg_pixels, 25), np.percentile(bg_pixels, 75)
    iqr = q3 - q1
    n_outliers = int(np.sum((bg_pixels < q1 - iqr_factor * iqr) |
                            (bg_pixels > q3 + iqr_factor * iqr)))

    return {
        'bkg_image': bkg_image,
        'median': center_value,
        'mean': float(np.mean(bkg_image)),
        'std': float(np.std(residuals)),
        'n_outliers': n_outliers,
        'quadrant_values': quad_values.tolist(),
        'fit_coeffs': tuple(float(c) for c in coeffs),
    }


def save_gfp_histogram(cell_df, output_path, background_median=None, max_area=None):
    """Save GFP intensity histogram with background estimate curve."""
    if len(cell_df) == 0:
        return

    # Use bkg-subtracted column if available, otherwise fall back to raw
    sum_col = 'gfp_sum_bkg_sub' if 'gfp_sum_bkg_sub' in cell_df.columns else 'gfp_sum_raw'
    raw_col = 'gfp_sum_raw' if 'gfp_sum_raw' in cell_df.columns else sum_col

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GFP sum histogram (bkg-subtracted)
    axes[0].hist(cell_df[sum_col], bins=30, color='green', alpha=0.7, edgecolor='darkgreen')
    axes[0].axvline(cell_df[sum_col].mean(), color='black', linestyle='--', label=f"Mean: {cell_df[sum_col].mean():.2e}")
    axes[0].axvline(cell_df[sum_col].median(), color='gray', linestyle=':', label=f"Median: {cell_df[sum_col].median():.2e}")
    axes[0].set_xlabel('GFP Bkg-subtracted Intensity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('GFP Sum Distribution (bkg-sub)')
    axes[0].legend()
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # GFP vs area scatter (raw, with background line)
    axes[1].scatter(cell_df['area_pixels'], cell_df[raw_col], alpha=0.6, c='green', s=30, label='Cells (raw)')

    # Add background estimate curve (background_median * area)
    if background_median is not None:
        area_range = np.linspace(0, cell_df['area_pixels'].max() * 1.1, 100)
        background_curve = background_median * area_range
        axes[1].plot(area_range, background_curve, 'r--', linewidth=2,
                     label=f'Background (median={background_median:.1f})')

    # Add max area cutoff line
    if max_area is not None:
        axes[1].axvline(max_area, color='red', linestyle=':', linewidth=2,
                        label=f'Max area cutoff ({max_area})')

    axes[1].set_xlabel('Cell Area (pixels)')
    axes[1].set_ylabel('GFP Raw Integrated Intensity')
    axes[1].set_title('GFP vs Cell Area')
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_area_cutoff_visualization(masks, cell_df, max_area, output_path):
    """Save visualization showing masks colored by area cutoff (green=kept, red=removed)."""
    if len(cell_df) == 0:
        return

    from skimage.segmentation import find_boundaries

    # Create RGB image
    rgb_masks = np.zeros((*masks.shape, 3), dtype=np.float32)

    n_kept = 0
    n_removed = 0

    for _, row in cell_df.iterrows():
        label = int(row['cell_label'])
        area = row['area_pixels']
        cell_mask = masks == label

        if area <= max_area:
            # Green for cells under cutoff (kept)
            rgb_masks[cell_mask, 1] = 0.7  # green channel
            n_kept += 1
        else:
            # Red for cells above cutoff (removed)
            rgb_masks[cell_mask, 0] = 0.9  # red channel
            n_removed += 1

    # Add boundaries
    boundaries = find_boundaries(masks, mode='outer')
    rgb_masks[boundaries] = [1, 1, 1]  # white boundaries

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb_masks)
    ax.set_title(f'Area Cutoff Filter (max={max_area} px)\n'
                 f'Green: kept ({n_kept} cells), Red: removed ({n_removed} cells)',
                 fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)

    return n_kept, n_removed


def save_area_colormap_visualization(masks, cell_df, output_path, max_area=None):
    """Save visualization showing masks colored by area (continuous colormap)."""
    if len(cell_df) == 0:
        return

    from skimage.segmentation import find_boundaries

    # Create area map
    area_map = np.zeros_like(masks, dtype=np.float64)

    for _, row in cell_df.iterrows():
        label = int(row['cell_label'])
        area = row['area_pixels']
        area_map[masks == label] = area

    fig, ax = plt.subplots(figsize=(12, 12))

    # Mask background
    area_map_masked = np.ma.masked_where(masks == 0, area_map)

    # Set color limits
    vmin = cell_df['area_pixels'].min()
    vmax = cell_df['area_pixels'].max()

    im = ax.imshow(area_map_masked, cmap='viridis', vmin=vmin, vmax=vmax)

    # Add boundaries
    boundaries = find_boundaries(masks, mode='outer')
    # Overlay boundaries in white
    boundary_overlay = np.zeros((*masks.shape, 4))
    boundary_overlay[boundaries] = [1, 1, 1, 1]
    ax.imshow(boundary_overlay)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Cell Area (pixels)', fontsize=12)

    # Add max area line to colorbar if provided
    if max_area is not None and max_area <= vmax:
        cbar.ax.axhline(y=max_area, color='red', linewidth=2, linestyle='--')
        cbar.ax.text(1.5, max_area, f'cutoff={max_area}', va='center', fontsize=10, color='red')

    title = f'Cells colored by Area\nRange: {vmin:.0f} - {vmax:.0f} px, n={len(cell_df)}'
    if max_area is not None:
        n_above = (cell_df['area_pixels'] > max_area).sum()
        title += f'\n{n_above} cells above cutoff ({max_area} px)'
    ax.set_title(title, fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_gfp_mask_visualization(masks, cell_df, gfp_bkg_sub, output_path):
    """Save masks colored by mean bkg-subtracted GFP intensity per cell."""
    if len(cell_df) == 0:
        return

    # Compute per-cell mean of bkg-subtracted GFP
    gfp_map = np.zeros_like(masks, dtype=np.float64)

    for _, row in cell_df.iterrows():
        label = int(row['cell_label'])
        cell_mask = masks == label
        mean_val = gfp_bkg_sub[cell_mask].mean()
        gfp_map[cell_mask] = mean_val

    fig, ax = plt.subplots(figsize=(12, 12))

    gfp_map_masked = np.ma.masked_where(masks == 0, gfp_map)

    vmin = np.percentile(gfp_map[masks > 0], 5) if np.any(masks > 0) else 0
    vmax = np.percentile(gfp_map[masks > 0], 95) if np.any(masks > 0) else 1

    im = ax.imshow(gfp_map_masked, cmap='Greens', vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('GFP mean intensity (bkg-sub)', fontsize=12)

    mean_gfp = gfp_map[masks > 0].mean() if np.any(masks > 0) else 0
    ax.set_title(f'Cells colored by GFP bkg-sub intensity\nMean: {mean_gfp:.0f}, n={len(cell_df)}', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_composite_visualization(ph3_image, gfp_bkg_sub, masks, output_path,
                                  best_z=None, best_z_gfp=None):
    """
    Save a composite image: Ph3 (gray) + GFP bkg-subtracted (green) with segmentation outlines.

    The Ph3 channel provides structural context while GFP signal is overlaid in green.
    """
    from skimage.segmentation import find_boundaries

    # Normalize Ph3 to 0-1
    ph3_float = ph3_image.astype(np.float64)
    ph3_low, ph3_high = np.percentile(ph3_float, 1), np.percentile(ph3_float, 99)
    ph3_norm = np.clip((ph3_float - ph3_low) / (ph3_high - ph3_low + 1e-10), 0, 1)

    # Normalize GFP bkg-sub to 0-1
    cell_mask_bool = masks > 0
    gfp_vmax = np.percentile(gfp_bkg_sub[cell_mask_bool], 99) if np.any(cell_mask_bool) else np.percentile(gfp_bkg_sub, 99)
    gfp_vmax = max(gfp_vmax, 1)
    gfp_norm = np.clip(gfp_bkg_sub / (gfp_vmax + 1e-10), 0, 1)

    # RGB composite: Ph3 in gray (all channels) + GFP in green
    composite = np.stack([ph3_norm * 0.7, ph3_norm * 0.7 + gfp_norm * 0.6, ph3_norm * 0.7], axis=-1)
    composite = np.clip(composite, 0, 1)

    boundaries = find_boundaries(masks, mode='outer')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Composite without outlines
    axes[0].imshow(composite)
    z_label = ''
    if best_z is not None:
        z_label = f' (Ph3 z={best_z}'
        if best_z_gfp is not None and best_z_gfp != best_z:
            z_label += f', GFP z={best_z_gfp}'
        z_label += ')'
    axes[0].set_title(f'Composite: Ph3 + GFP{z_label}')
    axes[0].axis('off')

    # Panel 2: Composite with segmentation outlines
    composite_outlines = composite.copy()
    composite_outlines[boundaries] = [1, 0, 0]  # red outlines
    axes[1].imshow(composite_outlines)
    n_cells = len(np.unique(masks)) - 1 if masks.max() > 0 else 0
    axes[1].set_title(f'Composite + Segmentation ({n_cells} cells)')
    axes[1].axis('off')

    # Panel 3: GFP-only green on black with outlines
    gfp_only = np.zeros((*gfp_bkg_sub.shape, 3))
    gfp_only[:, :, 1] = gfp_norm
    gfp_only[boundaries] = [1, 0, 0]
    axes[2].imshow(gfp_only)
    axes[2].set_title('GFP bkg-sub + Segmentation')
    axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_saturation_visualization_2ch(gfp_image, masks, output_path, saturation_value=65535):
    """
    Save visualization showing saturated pixels within cell masks (GFP only).

    Args:
        gfp_image: GFP fluorescence image
        masks: Segmentation masks
        output_path: Path to save the visualization
        saturation_value: Pixel value indicating saturation (default: 65535 for 16-bit)
    """
    from skimage.segmentation import find_boundaries

    boundaries = find_boundaries(masks, mode='outer')
    cell_region = masks > 0

    # Create dim grayscale background from GFP
    gfp_norm = (gfp_image - np.percentile(gfp_image, 1)) / (np.percentile(gfp_image, 99) - np.percentile(gfp_image, 1) + 1e-10)
    gfp_norm = np.clip(gfp_norm, 0, 1)
    gfp_rgb = np.stack([gfp_norm * 0.5] * 3, axis=-1)

    # White cell boundaries
    gfp_rgb[boundaries] = [1, 1, 1]

    # Bright green saturated pixels within masks
    gfp_saturated_mask = (gfp_image >= saturation_value) & cell_region
    gfp_rgb[gfp_saturated_mask] = [0, 1, 0]

    n_sat_pixels = int(np.sum(gfp_saturated_mask))
    n_sat_cells = len(np.unique(masks[gfp_saturated_mask])) if n_sat_pixels > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(gfp_rgb)
    ax.set_title(f'GFP Saturation Map\n{n_sat_pixels} saturated pixels in {n_sat_cells} cells', fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_shift_visualization_2ch(ph3_image, masks, shift_result, output_path):
    """
    Save visualization of channel shift optimization (2-channel version).

    Panel 1: Ph3 + original mask outlines (white)
    Panel 2: Ph3 + shifted mask outlines (cyan) with shift info
    Panel 3: Shift heatmap (percent increase grid)

    Args:
        ph3_image: Phase contrast image
        masks: Original segmentation masks
        shift_result: Result from find_optimal_channel_shift
        output_path: Path to save the visualization
    """
    from skimage.segmentation import find_boundaries

    dx = shift_result['optimal_shift_x']
    dy = shift_result['optimal_shift_y']
    pct_increase = shift_result['percent_increase']

    shifted_masks = apply_mask_shift(masks, dx, dy)

    # Normalize Ph3 for display
    ph3_low, ph3_high = np.percentile(ph3_image, 1), np.percentile(ph3_image, 99)
    ph3_norm = np.clip((ph3_image.astype(np.float64) - ph3_low) / (ph3_high - ph3_low + 1e-10), 0, 1)

    boundaries_original = find_boundaries(masks, mode='outer')
    boundaries_shifted = find_boundaries(shifted_masks, mode='outer')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Ph3 + original outlines (white)
    ph3_rgb = np.stack([ph3_norm] * 3, axis=-1)
    orig_overlay = ph3_rgb.copy()
    orig_overlay[boundaries_original] = [1, 1, 1]
    axes[0].imshow(orig_overlay)
    axes[0].set_title('Original (shift = 0, 0)', fontsize=12)
    axes[0].axis('off')

    # Panel 2: Ph3 + shifted outlines (cyan)
    shifted_overlay = ph3_rgb.copy()
    shifted_overlay[boundaries_shifted] = [0, 1, 1]
    axes[1].imshow(shifted_overlay)
    axes[1].set_title(f'Optimal Shift (dx={dx}, dy={dy})\n+{pct_increase:.1f}% avg GFP intensity', fontsize=12)
    axes[1].axis('off')

    # Panel 3: Shift heatmap
    if len(shift_result['all_shifts']) > 0:
        shift_range = int(np.sqrt(len(shift_result['all_shifts'])) // 2)
        heatmap_size = 2 * shift_range + 1
        heatmap = np.zeros((heatmap_size, heatmap_size))

        for shift_data in shift_result['all_shifts']:
            xi = shift_data['dx'] + shift_range
            yi = shift_data['dy'] + shift_range
            if 0 <= xi < heatmap_size and 0 <= yi < heatmap_size:
                heatmap[yi, xi] = shift_data['avg_percent_increase']

        im = axes[2].imshow(heatmap, cmap='RdYlGn', aspect='equal',
                            extent=[-shift_range-0.5, shift_range+0.5,
                                    shift_range+0.5, -shift_range-0.5])
        axes[2].plot(dx, dy, 'k*', markersize=15, label=f'Optimal ({dx}, {dy})')
        axes[2].plot(0, 0, 'ko', markersize=8, label='Origin')
        axes[2].set_xlabel('X shift (pixels)')
        axes[2].set_ylabel('Y shift (pixels)')
        axes[2].set_title('Avg % GFP Intensity Change', fontsize=12)
        axes[2].legend(loc='upper right', fontsize=9)
        plt.colorbar(im, ax=axes[2], label='% change')

    plt.suptitle('GFP Channel Alignment Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_gfp_statistics(cell_df, output_path, gfp_exposure_ms=None, bkg_median=None):
    """
    Save visualization of per-cell GFP statistics.

    Shows distributions of:
    - GFP rate (bkg-subtracted GFP sum / exposure time)
    - GFP flux (GFP rate / cell area in pixels)
    """
    if len(cell_df) == 0 or 'gfp_sum_bkg_sub' not in cell_df.columns:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: raw GFP sum vs bkg-subtracted
    axes[0, 0].hist(cell_df['gfp_sum_raw'], bins=50, color='gray', alpha=0.5, label='Raw')
    axes[0, 0].hist(cell_df['gfp_sum_bkg_sub'], bins=50, color='green', alpha=0.7, label='Bkg-subtracted')
    axes[0, 0].axvline(cell_df['gfp_sum_bkg_sub'].median(), color='black', linestyle='--',
                        label=f'Median: {cell_df["gfp_sum_bkg_sub"].median():.0f}')
    axes[0, 0].set_xlabel('GFP Sum Intensity')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('GFP Sum: Raw vs Bkg-subtracted')
    axes[0, 0].legend(fontsize=8)

    # Top-right: GFP rate distribution
    if 'gfp_rate' in cell_df.columns:
        rates = cell_df['gfp_rate'].dropna()
        axes[0, 1].hist(rates, bins=50, color='green', alpha=0.7, edgecolor='darkgreen')
        axes[0, 1].axvline(rates.median(), color='black', linestyle='--',
                            label=f'Median: {rates.median():.2f}')
        axes[0, 1].set_xlabel('GFP Rate (bkg-sub intensity / ms)')
        axes[0, 1].set_ylabel('Count')
        exp_label = f' (exposure={gfp_exposure_ms:.0f} ms)' if gfp_exposure_ms else ''
        axes[0, 1].set_title(f'GFP Rate Distribution{exp_label}')
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].text(0.5, 0.5, 'No exposure time\navailable', ha='center', va='center',
                        fontsize=14, transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('GFP Rate (not available)')

    # Bottom-left: GFP flux distribution
    if 'gfp_flux' in cell_df.columns:
        flux = cell_df['gfp_flux'].dropna()
        axes[1, 0].hist(flux, bins=50, color='teal', alpha=0.7, edgecolor='darkcyan')
        axes[1, 0].axvline(flux.median(), color='black', linestyle='--',
                            label=f'Median: {flux.median():.4f}')
        axes[1, 0].set_xlabel('GFP Flux (bkg-sub intensity / ms / pixel)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('GFP Flux Distribution (rate / cell area)')
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].text(0.5, 0.5, 'No exposure time\navailable', ha='center', va='center',
                        fontsize=14, transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('GFP Flux (not available)')

    # Bottom-right: GFP sum vs area scatter with background line
    axes[1, 1].scatter(cell_df['area_pixels'], cell_df['gfp_sum_raw'], alpha=0.4, s=20, c='gray', label='Raw')
    axes[1, 1].scatter(cell_df['area_pixels'], cell_df['gfp_sum_bkg_sub'], alpha=0.5, s=20, c='green', label='Bkg-sub')
    if bkg_median is not None:
        area_range = np.linspace(0, cell_df['area_pixels'].max() * 1.1, 100)
        axes[1, 1].plot(area_range, bkg_median * area_range, 'r--', linewidth=2,
                        label=f'Background (median={bkg_median:.1f})')
    axes[1, 1].set_xlabel('Cell Area (pixels)')
    axes[1, 1].set_ylabel('GFP Sum Intensity')
    axes[1, 1].set_title('GFP vs Cell Area')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def process_2channel_image(input_file, output_dir, params=CELLPOSE_PARAMS, time_index=None,
                           gfp_channel=0, ph3_channel=1, max_area=MAX_CELL_AREA,
                           force_z=None, focus_metric='brenner', background_percentile=25,
                           shift_range=5, saturation_value=65535,
                           autofocus_channel='gfp', segment_channel='ph3'):
    """
    Process a 2-channel image (GFP + Ph3).

    Args:
        input_file: Path to nd2 file
        output_dir: Output directory
        params: Segmentation parameters
        time_index: Time index for time series
        gfp_channel: GFP channel index
        ph3_channel: Ph3 channel index
        max_area: Maximum cell area in pixels (cells above this are excluded)
        force_z: Force a specific z-slice (None for automatic)
        focus_metric: Autofocus metric ('brenner', 'gradient', 'laplacian', 'normalized_variance')
        background_percentile: Percentile for background subtraction in autofocus (default: 25)
        shift_range: Max shift in pixels for channel alignment (default: 5)
        saturation_value: Pixel value for saturation detection (default: 65535)
        autofocus_channel: Which channel to autofocus on ('gfp' or 'ph3', default: 'gfp')
        segment_channel: Which channel to segment on ('ph3' or 'gfp', default: 'ph3')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    data, sizes, channels, has_z = load_nd2_2channel(input_file, time_index=time_index)

    # Extract and save nd2 metadata
    nd2_metadata = extract_nd2_metadata(input_file, time_index=time_index)
    save_nd2_metadata(nd2_metadata, output_dir / 'nd2_metadata.csv')

    # Handle z-stack: autofocus
    focus_ch = ph3_channel if autofocus_channel == 'ph3' else gfp_channel
    focus_ch_name = 'Ph3' if autofocus_channel == 'ph3' else 'GFP'
    best_z = None
    z_scores = None
    z_stack_data = None  # Keep full z-stack for visualizations
    if has_z:
        z_stack_data = data  # Save before slicing
        # data shape is (Z, C, Y, X)
        if force_z is not None:
            best_z = force_z
            # Still compute scores for visualization
            _, z_scores = select_best_z_fluorescence(
                data, focus_ch,
                background_percentile=background_percentile,
                metric=focus_metric,
            )
            print(f"\nUsing forced z={best_z}")
        else:
            print(f"\nAutofocusing on {focus_ch_name} channel (background_percentile={background_percentile}, metric={focus_metric})...")
            best_z, z_scores = select_best_z_fluorescence(
                data, focus_ch,
                background_percentile=background_percentile,
                metric=focus_metric,
            )
            print(f"  Z-scores: {['%.1f' % s for s in z_scores]}")
            print(f"  Selected z={best_z}")

        # Save z-selection info (GFP max will be added after segmentation)
        z_df = pd.DataFrame({
            'z': range(len(z_scores)),
            'focus_score': z_scores,
            'selected_ph3': [z == best_z for z in range(len(z_scores))],
        })
        z_df.to_csv(output_dir / 'z_selection.csv', index=False)

        # Extract the best z-slice: (Z, C, Y, X) -> (C, Y, X)
        data = data[best_z]
        print(f"  Extracted z={best_z}, shape: {data.shape}")

    # Handle different dimension orders
    if len(data.shape) == 3:
        # Shape is (C, Y, X)
        gfp_image = data[gfp_channel, :, :]
        ph3_image = data[ph3_channel, :, :]
    elif len(data.shape) == 2:
        # Single channel - shouldn't happen but handle it
        gfp_image = data
        ph3_image = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    # Initialize model and segment
    seg_image = gfp_image if segment_channel == 'gfp' else ph3_image
    seg_ch_name = 'GFP' if segment_channel == 'gfp' else 'Ph3'
    print(f"\nInitializing Omnipose model...")
    model = omni_models.CellposeModel(model_type=params['model_type'], gpu=False)
    print(f"  Model: {params['model_type']}")
    print(f"  Segmenting on {seg_ch_name} channel...")

    masks = segment_frame(model, seg_image, params)
    n_cells = len(np.unique(masks)) - 1
    print(f"  Detected {n_cells} cells")

    # Channel shift search (skip when segmenting on GFP — masks already aligned)
    if segment_channel == 'gfp':
        print(f"\nSkipping channel shift (segmented on GFP, masks already aligned)")
        shift_result = {'optimal_shift_x': 0, 'optimal_shift_y': 0,
                        'percent_increase': 0, 'all_shifts': []}
        dx, dy, pct_increase = 0, 0, 0.0
        gfp_masks = masks.copy()
    else:
        print(f"\nChannel shift search (range={shift_range})...")
        shift_result = find_optimal_channel_shift(masks, gfp_image, shift_range=shift_range)
        dx = shift_result['optimal_shift_x']
        dy = shift_result['optimal_shift_y']
        pct_increase = shift_result['percent_increase']
        if dx != 0 or dy != 0:
            gfp_masks = apply_mask_shift(masks, dx, dy)
            print(f"  Optimal shift: dx={dx}, dy={dy} (+{pct_increase:.1f}% avg GFP intensity)")
        else:
            gfp_masks = masks.copy()
            print(f"  No shift needed (best at origin)")

    # GFP z-refocus: find the z-slice where max(GFP) within masks is highest
    best_z_gfp = best_z  # default: same as Ph3
    z_max_gfp = None
    if has_z and z_stack_data is not None and n_cells > 0:
        print(f"\nGFP z-refocus: scanning z-slices for brightest GFP in masks...")
        best_z_gfp, z_max_gfp = select_best_z_gfp_in_masks(
            z_stack_data, gfp_masks, gfp_channel=gfp_channel)
        print(f"  Max GFP per z: {['%.0f' % v for v in z_max_gfp]}")
        print(f"  Best GFP z={best_z_gfp} (Ph3 z={best_z})")

        # Use the GFP-optimal z-slice for GFP quantification
        gfp_image = z_stack_data[best_z_gfp, gfp_channel, :, :].copy()
        print(f"  Using GFP from z={best_z_gfp}, Ph3 from z={best_z}")

    # Measure raw GFP intensities (using shifted masks for GFP)
    print("\nMeasuring GFP intensities...")
    cell_df_all = measure_cells_gfp_only(gfp_masks, gfp_image,
                                          saturation_value=saturation_value)

    # Rename gfp_sum -> gfp_sum_raw to distinguish from bkg-subtracted
    if len(cell_df_all) > 0:
        cell_df_all = cell_df_all.rename(columns={'gfp_sum': 'gfp_sum_raw'})

    # Report saturation
    n_gfp_saturated = 0
    if len(cell_df_all) > 0:
        n_gfp_saturated = int(cell_df_all['gfp_saturated'].sum())
        if n_gfp_saturated > 0:
            print(f"  WARNING: {n_gfp_saturated} cells have saturated GFP pixels (>={saturation_value})")

    # Calculate 2D linear background: 3x3 quadrant grid, robust median per quadrant, plane fit
    # Note: saturated cells remain in gfp_masks so their bright pixels are excluded from background
    print("\nCalculating 2D quadratic GFP background (4x4 grid)...")
    bkg_stats = calculate_background(gfp_image, gfp_masks, iqr_factor=1.5)
    bkg_image = bkg_stats['bkg_image']
    bkg_median = bkg_stats['median']  # center value of the surface
    bkg_coeffs = bkg_stats['fit_coeffs']
    print(f"  Background surface: {bkg_coeffs[0]:.2e}*x^2 + {bkg_coeffs[1]:.2e}*y^2 + "
          f"{bkg_coeffs[2]:.2e}*x*y + {bkg_coeffs[3]:.4f}*x + {bkg_coeffs[4]:.4f}*y + {bkg_coeffs[5]:.1f}")
    print(f"  Region medians (4x4): {['%.1f' % v for v in bkg_stats['quadrant_values']]}")
    print(f"  Center value: {bkg_median:.1f} (mean: {bkg_stats['mean']:.1f}, "
          f"residual std: {bkg_stats['std']:.1f}, outliers: {bkg_stats['n_outliers']})")

    # Add bkg-subtracted columns: per-cell sum of (raw - bkg_image) within mask
    if len(cell_df_all) > 0:
        cell_df_all['gfp_bkg_per_pixel'] = bkg_median  # report center value
        gfp_sub_image = gfp_image.astype(np.float64) - bkg_image
        bkg_sub_sums = []
        for label in cell_df_all['cell_label']:
            cell_mask = gfp_masks == label
            bkg_sub_sums.append(gfp_sub_image[cell_mask].sum())
        cell_df_all['gfp_sum_bkg_sub'] = bkg_sub_sums

    # Update z_selection.csv with GFP max info
    if has_z and z_max_gfp is not None:
        z_df = pd.read_csv(output_dir / 'z_selection.csv')
        z_df['gfp_max_in_masks'] = z_max_gfp
        z_df['selected_gfp'] = [z == best_z_gfp for z in range(len(z_max_gfp))]
        z_df.to_csv(output_dir / 'z_selection.csv', index=False)

    # Create background-subtracted GFP image for visualizations (keep negative values)
    gfp_bkg_sub = gfp_image.astype(np.float64) - bkg_image

    # Extract exposure time for GFP rate/flux
    exposure_times = extract_exposure_times(input_file)
    gfp_exposure_ms = exposure_times.get('GFP', None)
    if gfp_exposure_ms:
        print(f"  GFP exposure: {gfp_exposure_ms} ms")
        if len(cell_df_all) > 0:
            cell_df_all['gfp_rate'] = cell_df_all['gfp_sum_bkg_sub'] / gfp_exposure_ms
            cell_df_all['gfp_flux'] = cell_df_all['gfp_rate'] / cell_df_all['area_pixels']
    else:
        print("  GFP exposure: not found in metadata")

    # Filter cells by area and saturation
    n_cells_before = len(cell_df_all)
    if len(cell_df_all) > 0:
        cell_df = cell_df_all[
            (cell_df_all['area_pixels'] <= max_area) &
            (cell_df_all['gfp_saturated'] == 0)
        ].copy()
        n_removed_area = int((cell_df_all['area_pixels'] > max_area).sum())
        n_removed_sat = int(cell_df_all['gfp_saturated'].sum())
        n_removed = n_cells_before - len(cell_df)
        print(f"  Area filter (max={max_area}): removed {n_removed_area} cells")
        print(f"  Saturation filter: removed {n_removed_sat} cells")
        print(f"  Total kept: {len(cell_df)} / {n_cells_before}")
    else:
        cell_df = cell_df_all

    n_cells_filtered = len(cell_df)

    # Compute summary (on filtered cells)
    base_summary = {
        'n_cells_total': n_cells,
        'n_cells_filtered': n_cells_filtered,
        'n_cells_removed': n_cells - n_cells_filtered,
        'n_gfp_saturated': n_gfp_saturated,
        'gfp_shift_x': dx,
        'gfp_shift_y': dy,
        'gfp_shift_pct_increase': pct_increase,
        'best_z_ph3': best_z,
        'best_z_gfp': best_z_gfp,
        'gfp_bkg_fit_xx': bkg_coeffs[0],
        'gfp_bkg_fit_yy': bkg_coeffs[1],
        'gfp_bkg_fit_xy': bkg_coeffs[2],
        'gfp_bkg_fit_x': bkg_coeffs[3],
        'gfp_bkg_fit_y': bkg_coeffs[4],
        'gfp_bkg_fit_c': bkg_coeffs[5],
    }
    if len(cell_df) > 0:
        summary = {
            **base_summary,
            'gfp_sum_raw_mean': cell_df['gfp_sum_raw'].mean(),
            'gfp_sum_raw_median': cell_df['gfp_sum_raw'].median(),
            'gfp_bkg_per_pixel': bkg_median,
            'gfp_bkg_std': bkg_stats['std'],
            'gfp_bkg_n_outliers': bkg_stats['n_outliers'],
            'gfp_sum_bkg_sub_mean': cell_df['gfp_sum_bkg_sub'].mean(),
            'gfp_sum_bkg_sub_median': cell_df['gfp_sum_bkg_sub'].median(),
            'gfp_exposure_ms': gfp_exposure_ms,
            'gfp_rate_mean': cell_df['gfp_rate'].mean() if 'gfp_rate' in cell_df.columns else None,
            'gfp_rate_median': cell_df['gfp_rate'].median() if 'gfp_rate' in cell_df.columns else None,
            'gfp_flux_mean': cell_df['gfp_flux'].mean() if 'gfp_flux' in cell_df.columns else None,
            'gfp_flux_median': cell_df['gfp_flux'].median() if 'gfp_flux' in cell_df.columns else None,
            'area_mean': cell_df['area_pixels'].mean(),
        }
    else:
        summary = {
            **base_summary,
            'gfp_sum_raw_mean': 0,
            'gfp_sum_raw_median': 0,
            'gfp_bkg_per_pixel': bkg_median,
            'gfp_bkg_std': bkg_stats['std'],
            'gfp_bkg_n_outliers': bkg_stats['n_outliers'],
            'gfp_sum_bkg_sub_mean': 0,
            'gfp_sum_bkg_sub_median': 0,
            'gfp_exposure_ms': gfp_exposure_ms,
            'gfp_rate_mean': None,
            'gfp_rate_median': None,
            'gfp_flux_mean': None,
            'gfp_flux_median': None,
            'area_mean': 0,
        }

    # Save outputs
    print("\nSaving outputs...")

    # Masks
    tifffile.imwrite(output_dir / 'masks.tif', masks.astype(np.uint16))
    print(f"  Masks: {output_dir / 'masks.tif'}")

    # Background-subtracted GFP image (float32 TIFF, preserves negative values)
    tifffile.imwrite(output_dir / 'gfp_bkg_sub.tif', gfp_bkg_sub.astype(np.float32))
    print(f"  GFP bkg-sub image: {output_dir / 'gfp_bkg_sub.tif'}")

    # Cell measurements (all cells, before filtering)
    cell_df_all.to_csv(output_dir / 'cell_measurements_all.csv', index=False)
    print(f"  Cell measurements (all): {output_dir / 'cell_measurements_all.csv'} ({n_cells} cells)")

    # Cell measurements (filtered)
    cell_df.to_csv(output_dir / 'cell_measurements.csv', index=False)
    print(f"  Cell measurements (filtered): {output_dir / 'cell_measurements.csv'} ({n_cells_filtered} cells)")

    # Summary stats
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'summary_stats.csv', index=False)
    print(f"  Summary stats: {output_dir / 'summary_stats.csv'}")

    # Visualizations
    print("\nGenerating visualizations...")

    save_segmentation_visualization_2ch(ph3_image, gfp_bkg_sub, masks, n_cells,
                                        output_dir / 'segmentation.png',
                                        best_z=best_z, best_z_gfp=best_z_gfp,
                                        z_scores=z_scores, z_max_gfp=z_max_gfp)
    print(f"  Segmentation: segmentation.png")

    # Composite Ph3 + GFP
    save_composite_visualization(ph3_image, gfp_bkg_sub, masks,
                                  output_dir / 'composite.png',
                                  best_z=best_z, best_z_gfp=best_z_gfp)
    print(f"  Composite: composite.png")

    # Saturation map
    save_saturation_visualization_2ch(gfp_image, gfp_masks,
                                       output_dir / 'saturation_map.png',
                                       saturation_value=saturation_value)
    print(f"  Saturation map: saturation_map.png")

    # Channel shift visualization
    save_shift_visualization_2ch(ph3_image, masks, shift_result,
                                  output_dir / 'channel_shift_gfp.png')
    print(f"  Channel shift: channel_shift_gfp.png")

    # Z-crop visualization (only if z-stack was present)
    if z_stack_data is not None:
        save_z_crop_visualization_2ch(z_stack_data, masks, best_z,
                                       output_dir / 'z_crop.png',
                                       gfp_channel=gfp_channel,
                                       ph3_channel=ph3_channel)
        print(f"  Z-crop: z_crop.png")

    # Area cutoff visualization (before filtering in other plots)
    save_area_cutoff_visualization(masks, cell_df_all, max_area,
                                   output_dir / 'area_cutoff.png')
    print(f"  Area cutoff: area_cutoff.png")

    # Area colormap visualization (continuous colors)
    save_area_colormap_visualization(masks, cell_df_all, output_dir / 'area_colormap.png',
                                     max_area=max_area)
    print(f"  Area colormap: area_colormap.png")

    # GFP histogram (using bkg-subtracted values)
    save_gfp_histogram(cell_df, output_dir / 'gfp_histogram.png',
                       background_median=bkg_median, max_area=max_area)
    print(f"  GFP histogram: gfp_histogram.png")

    # GFP mask visualization (uses bkg-subtracted image for coloring)
    save_gfp_mask_visualization(masks, cell_df, gfp_bkg_sub, output_dir / 'gfp_masks.png')
    print(f"  GFP masks: gfp_masks.png")

    # GFP statistics: rate and flux distributions
    save_gfp_statistics(cell_df, output_dir / 'gfp_statistics.png',
                        gfp_exposure_ms=gfp_exposure_ms, bkg_median=bkg_median)
    print(f"  GFP statistics: gfp_statistics.png")

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if best_z is not None:
        print(f"Best z-slice (Ph3): {best_z}")
        if best_z_gfp is not None:
            print(f"Best z-slice (GFP): {best_z_gfp}")
    print(f"Channel shift: dx={dx}, dy={dy} (+{pct_increase:.1f}%)")
    print(f"Total cells detected: {n_cells}")
    print(f"Saturated cells: {n_gfp_saturated}")
    print(f"Cells after filtering (area+saturation): {n_cells_filtered}")
    print(f"Cells removed: {n_cells - n_cells_filtered}")
    print(f"\nGFP background (2D quadratic fit, 4x4 grid):")
    print(f"  Center value: {bkg_median:.1f}")
    print(f"  Region medians: {['%.1f' % v for v in bkg_stats['quadrant_values']]}")
    print(f"  Residual std: {bkg_stats['std']:.1f}")
    print(f"  Outliers removed: {bkg_stats['n_outliers']}")
    if gfp_exposure_ms:
        print(f"\nGFP exposure: {gfp_exposure_ms} ms")
    print(f"\nGFP sum (raw, filtered cells):")
    print(f"  Mean:   {summary['gfp_sum_raw_mean']:.1f}")
    print(f"  Median: {summary['gfp_sum_raw_median']:.1f}")
    print(f"\nGFP sum (bkg-subtracted, filtered cells):")
    print(f"  Mean:   {summary['gfp_sum_bkg_sub_mean']:.1f}")
    print(f"  Median: {summary['gfp_sum_bkg_sub_median']:.1f}")
    if summary.get('gfp_rate_mean') is not None:
        print(f"\nGFP rate (bkg-sub / exposure):")
        print(f"  Mean:   {summary['gfp_rate_mean']:.4f}")
        print(f"  Median: {summary['gfp_rate_median']:.4f}")
    if summary.get('gfp_flux_mean') is not None:
        print(f"\nGFP flux (rate / area):")
        print(f"  Mean:   {summary['gfp_flux_mean']:.6f}")
        print(f"  Median: {summary['gfp_flux_median']:.6f}")
    print("=" * 50)

    return cell_df, summary, masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment 2-channel images (GFP + Ph3)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input nd2 file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--model', type=str, default='bact_phase_omni',
                        help='Omnipose model type (default: bact_phase_omni)')
    parser.add_argument('--diameter', type=int, default=30,
                        help='Cell diameter for segmentation (default: 30)')
    parser.add_argument('--time-index', type=int, default=None,
                        help='For time series: which frame to process (0-indexed)')
    parser.add_argument('--gfp-channel', type=int, default=0,
                        help='GFP channel index (default: 0)')
    parser.add_argument('--ph3-channel', type=int, default=1,
                        help='Ph3 channel index (default: 1)')
    parser.add_argument('--max-area', type=int, default=MAX_CELL_AREA,
                        help=f'Maximum cell area in pixels (default: {MAX_CELL_AREA})')
    parser.add_argument('--force-z', type=int, default=None,
                        help='Force a specific z-slice (default: autofocus)')
    parser.add_argument('--focus-metric', type=str, default='brenner',
                        choices=['brenner', 'gradient', 'laplacian', 'normalized_variance'],
                        help='Autofocus metric (default: brenner)')
    parser.add_argument('--bkg-percentile', type=int, default=25,
                        help='Background percentile for autofocus (default: 25)')
    parser.add_argument('--shift-range', type=int, default=5,
                        help='Max shift in pixels for channel alignment (default: 5)')
    parser.add_argument('--saturation-value', type=int, default=65535,
                        help='Pixel value for saturation detection (default: 65535)')
    parser.add_argument('--autofocus-channel', type=str, default='gfp',
                        choices=['gfp', 'ph3'],
                        help='Channel for autofocus (default: gfp)')
    parser.add_argument('--segment-channel', type=str, default='ph3',
                        choices=['gfp', 'ph3'],
                        help='Channel for segmentation (default: ph3)')
    args = parser.parse_args()

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        input_path = Path(args.input)
        output_dir = Path('./segmentation_output') / f'2ch_{input_path.stem}'

    # Update params
    params = CELLPOSE_PARAMS.copy()
    params['model_type'] = args.model
    params['diameter'] = args.diameter

    print(f"Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {output_dir}")
    print(f"  Model: {params['model_type']}")
    print(f"  Diameter: {params['diameter']}")
    print(f"  GFP channel: {args.gfp_channel}")
    print(f"  Ph3 channel: {args.ph3_channel}")
    print(f"  Max area: {args.max_area}")
    print(f"  Force z: {args.force_z}")
    print(f"  Focus metric: {args.focus_metric}")
    print(f"  Bkg percentile: {args.bkg_percentile}")
    print(f"  Shift range: {args.shift_range}")
    print(f"  Saturation value: {args.saturation_value}")
    print(f"  Autofocus channel: {args.autofocus_channel}")
    print(f"  Segment channel: {args.segment_channel}")

    # Process
    cell_df, summary, masks = process_2channel_image(
        args.input,
        output_dir,
        params=params,
        time_index=args.time_index,
        gfp_channel=args.gfp_channel,
        ph3_channel=args.ph3_channel,
        max_area=args.max_area,
        force_z=args.force_z,
        focus_metric=args.focus_metric,
        background_percentile=args.bkg_percentile,
        shift_range=args.shift_range,
        saturation_value=args.saturation_value,
        autofocus_channel=args.autofocus_channel,
        segment_channel=args.segment_channel,
    )

    print("\nDone!")
