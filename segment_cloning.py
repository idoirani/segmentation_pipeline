	#!/usr/bin/env python
"""
Z-stack segmentation and quantification for cloning experiments.

Single time point, 3-channel (Ph3, GFP, mCherry) z-stack images.
- Finds best focus using mCherry channel
- Segments cells using mCherry channel
- Quantifies GFP and mCherry intensities per cell
- Calculates GFP/mCherry ratio

Usage:
    conda activate cellpose
    python segment_cloning.py --input ./cloning_wt/1_in_wt/290126_z_gfpmcharry_ph3.nd2
"""

import numpy as np
import nd2
import tifffile
from pathlib import Path
from cellpose_omni import models as omni_models
from cellpose_omni import utils
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import argparse
from autofocus import select_best_z_fluorescence


# =============================================================================
# Configuration
# =============================================================================

# Channel indices (defaults; overridden at runtime by detect_channel_indices)
PH3_CHANNEL = 1
GFP_CHANNEL = 0
MCHERRY_CHANNEL = 2


def detect_channel_indices(channel_names):
    """
    Detect GFP, Ph3, and mCherry channel indices from nd2 channel name list.

    Returns (gfp_idx, ph3_idx, mcherry_idx). Any value may be None if the
    corresponding channel is not found.
    """
    lo = [c.lower() for c in channel_names]
    gfp_idx     = next((i for i, n in enumerate(lo) if 'gfp' in n or 'yfp' in n), None)
    mcherry_idx = next((i for i, n in enumerate(lo)
                        if 'mcherry' in n or 'cherry' in n or 'mkate' in n or 'rfp' in n), None)
    ph3_idx     = next((i for i, n in enumerate(lo)
                        if 'ph3' in n or 'phase' in n or ('ph' in n and 'mcherry' not in n)), None)
    return gfp_idx, ph3_idx, mcherry_idx

# Omnipose parameters - using Ph3 (phase contrast) for segmentation
CELLPOSE_PARAMS = {
    'model_type': 'bact_fluor_omni',  # Omnipose bacterial phase model
    'diameter': 15,                    # increased from 15
    'channels': [0, 0],  # grayscale
    'flow_threshold': 0.4,             # 
    'mask_threshold': -2,               # 
    'min_size': 200,                    # filter out small fragments
    'omni': True,
    'cluster': True,  # Use DBSCAN clustering
}

# Visualization settings
VIS_DPI = 150


# =============================================================================
# Focus Metric Functions
# =============================================================================

def variance_of_laplacian(image):
    """
    Compute the Variance of Laplacian focus metric.
    Higher values indicate sharper/more in-focus images.
    """
    img_float = image.astype(np.float64)
    laplacian = cv2.Laplacian(img_float, cv2.CV_64F)
    return laplacian.var()


def normalized_gradient_magnitude(image):
    """
    Compute normalized gradient magnitude focus metric.

    Higher values indicate sharper/more in-focus images.
    Better suited for fluorescence than Laplacian variance.
    """
    img_float = image.astype(np.float64)

    # Compute Sobel gradients
    grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize by mean intensity to handle varying brightness
    mean_intensity = np.mean(img_float)
    if mean_intensity < 1e-6:
        return 0

    return np.mean(grad_mag) / mean_intensity


def mask_outlier_pixels(image, iqr_factor=2.0):
    """
    Create a mask for pixels with unusual intensity values.

    Masks pixels that are outliers (> iqr_factor * IQR from Q1/Q3).

    Args:
        image: Input image
        iqr_factor: Factor for IQR-based outlier detection (default: 2.0)

    Returns:
        valid_mask: Boolean mask where True = valid (non-outlier) pixels
    """
    q1 = np.percentile(image, 25)
    q3 = np.percentile(image, 75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    valid_mask = (image >= lower_bound) & (image <= upper_bound)
    return valid_mask


def highpass_gradient_magnitude(image, highpass_sigma=4.8, mask_outliers=True, iqr_factor=2.0):
    """
    Compute gradient magnitude on high-pass filtered image.

    High-pass filter suppresses illumination halos in phase contrast.
    Default sigma corresponds to ~12 pixel cutoff (12 / 2.5 = 4.8).

    Args:
        image: Input image
        highpass_sigma: Gaussian sigma for high-pass filter
        mask_outliers: Whether to mask outlier pixels before computation
        iqr_factor: Factor for IQR-based outlier detection

    Returns:
        Mean gradient magnitude of filtered image (excluding outliers if masked)
    """
    img_float = image.astype(np.float64)

    # Mask outlier pixels if requested
    if mask_outliers:
        valid_mask = mask_outlier_pixels(img_float, iqr_factor)
    else:
        valid_mask = np.ones_like(img_float, dtype=bool)

    # High-pass filter: subtract Gaussian blur (low-pass) from original
    low_pass = ndimage.gaussian_filter(img_float, sigma=highpass_sigma)
    high_pass = img_float - low_pass

    # Compute gradient magnitude on filtered image
    grad_x = cv2.Sobel(high_pass, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(high_pass, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Return mean only over valid (non-outlier) pixels
    return np.mean(grad_mag[valid_mask])


def select_best_z_highpass_gradient(z_stack, channel_idx, highpass_sigma=4.8, mask_outliers=True, iqr_factor=2.0):
    """
    Select best z-slice using high-pass filtered gradient magnitude.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use
        highpass_sigma: Gaussian sigma for high-pass filter
        mask_outliers: Whether to mask outlier pixels before computation
        iqr_factor: Factor for IQR-based outlier detection

    Returns:
        best_z: Index of best-focused z-slice
        scores: Gradient scores for all z-slices
    """
    n_z = z_stack.shape[0]
    scores = []

    for z in range(n_z):
        image = z_stack[z, channel_idx, :, :]
        score = highpass_gradient_magnitude(image, highpass_sigma, mask_outliers, iqr_factor)
        scores.append(score)

    best_z = np.argmax(scores)
    return best_z, scores


def select_best_z_brenner(z_stack, channel_idx):
    """
    Select best z-slice using Brenner gradient (sum of squared horizontal differences).

    This metric works well for fluorescence images and is less sensitive to
    background structure than high-pass gradient methods.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use

    Returns:
        best_z: Index of best-focused z-slice
        scores: Brenner scores for all z-slices
    """
    n_z = z_stack.shape[0]
    scores = []

    for z in range(n_z):
        image = z_stack[z, channel_idx, :, :].astype(np.float64)
        # Brenner gradient: sum of squared differences with 2-pixel shift
        brenner = np.sum((image[:, 2:] - image[:, :-2]) ** 2)
        scores.append(brenner)

    best_z = np.argmax(scores)
    return best_z, scores


def select_best_z_normalized_variance(z_stack, channel_idx):
    """
    Select best z-slice using normalized variance (std/mean).

    Higher values indicate more contrast, which correlates with focus.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use

    Returns:
        best_z: Index of best-focused z-slice
        scores: Normalized variance scores for all z-slices
    """
    n_z = z_stack.shape[0]
    scores = []

    for z in range(n_z):
        image = z_stack[z, channel_idx, :, :].astype(np.float64)
        # Coefficient of variation
        cv = image.std() / image.mean() if image.mean() > 0 else 0
        scores.append(cv)

    best_z = np.argmax(scores)
    return best_z, scores


def select_best_z_dual_highpass_gradient(z_stack, ph3_channel_idx, mcherry_channel_idx, highpass_sigma=4.8, mask_outliers=True, iqr_factor=2.0):
    """
    Select best z-slice using high-pass gradient on both Ph3 and mCherry.

    Takes the median of the two best z-indices.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        ph3_channel_idx: Ph3 channel index
        mcherry_channel_idx: mCherry channel index
        highpass_sigma: Gaussian sigma for high-pass filter
        mask_outliers: Whether to mask outlier pixels before computation
        iqr_factor: Factor for IQR-based outlier detection

    Returns:
        best_z: Median z-slice from both channels
        results: Dict with detailed results
    """
    # Ph3 channel
    best_z_ph3, ph3_scores = select_best_z_highpass_gradient(z_stack, ph3_channel_idx, highpass_sigma, mask_outliers, iqr_factor)

    # mCherry channel
    best_z_mcherry, mcherry_scores = select_best_z_highpass_gradient(z_stack, mcherry_channel_idx, highpass_sigma, mask_outliers, iqr_factor)

    # Take median of the two
    best_z_median = int(round(np.median([best_z_ph3, best_z_mcherry])))

    results = {
        'ph3': {
            'best_z': best_z_ph3,
            'scores': ph3_scores,
        },
        'mcherry': {
            'best_z': best_z_mcherry,
            'scores': mcherry_scores,
        },
        'combined_z': best_z_median,
    }

    return best_z_median, results


def select_best_z_by_gradient(z_stack, channel_idx, use_sqrt=True):
    """
    Select best z-slice using normalized gradient magnitude.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use
        use_sqrt: Apply sqrt transform to compress dynamic range

    Returns:
        best_z: Index of best-focused z-slice
        scores: Gradient scores for all z-slices
    """
    n_z = z_stack.shape[0]
    scores = []

    for z in range(n_z):
        image = z_stack[z, channel_idx, :, :].astype(np.float64)
        if use_sqrt:
            image = np.sqrt(image)
        score = normalized_gradient_magnitude(image)
        scores.append(score)

    best_z = np.argmax(scores)
    return best_z, scores


def select_best_z_by_cell_size(z_stack, channel_idx, use_sqrt=True):
    """
    Select best z-slice where cells appear smallest (most in focus).

    Out-of-focus cells appear as large diffuse blobs.
    In-focus cells appear compact/small.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use
        use_sqrt: Apply sqrt transform to compress dynamic range

    Returns:
        best_z: Index of best-focused z-slice
        mean_areas: Mean cell area for each z-slice
    """
    from skimage.filters import threshold_otsu

    n_z = z_stack.shape[0]
    mean_areas = []

    for z in range(n_z):
        image = z_stack[z, channel_idx, :, :].astype(np.float64)
        if use_sqrt:
            image = np.sqrt(image)

        # Otsu threshold
        try:
            thresh = threshold_otsu(image)
        except:
            mean_areas.append(np.inf)
            continue

        binary = image > thresh

        # Find connected components
        labeled, n_objects = ndimage.label(binary)

        if n_objects == 0:
            mean_areas.append(np.inf)
            continue

        # Measure areas of each object
        areas = ndimage.sum(binary, labeled, range(1, n_objects + 1))

        # Filter to cell-like sizes (exclude tiny noise and huge background)
        valid_areas = [a for a in areas if 50 < a < 5000]

        if len(valid_areas) == 0:
            mean_areas.append(np.inf)
        else:
            mean_areas.append(np.mean(valid_areas))

    # Select z with minimum mean area (smallest = most focused)
    best_z = np.argmin(mean_areas)
    return best_z, mean_areas


def select_best_z_combined(z_stack, channel_idx):
    """
    Select best z-slice using combined approach:
    1. Normalized gradient magnitude (higher = better)
    2. Minimum cell size (smaller = better)

    Returns the mean of the two best z indices.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use

    Returns:
        best_z: Combined best z-slice (rounded mean of two methods)
        results: Dict with detailed results from each method
    """
    # Method 1: Gradient magnitude
    best_z_gradient, gradient_scores = select_best_z_by_gradient(z_stack, channel_idx)

    # Method 2: Cell size
    best_z_size, mean_areas = select_best_z_by_cell_size(z_stack, channel_idx)

    # Combined: mean of the two methods
    best_z_combined = int(round((best_z_gradient + best_z_size) / 2))

    results = {
        'gradient': {
            'best_z': best_z_gradient,
            'scores': gradient_scores,
        },
        'cell_size': {
            'best_z': best_z_size,
            'mean_areas': mean_areas,
        },
        'combined_z': best_z_combined,
    }

    return best_z_combined, results


def select_best_z_dual_channel(z_stack, gfp_channel_idx, mcherry_channel_idx):
    """
    Select best z-slice using both GFP and mCherry channels.

    Runs gradient and cell_size methods on both channels,
    then takes the median of all 4 z-indices.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        gfp_channel_idx: GFP channel index
        mcherry_channel_idx: mCherry channel index

    Returns:
        best_z: Median z-slice from all methods
        results: Dict with detailed results from each method
    """
    # GFP channel
    best_z_gfp_gradient, gfp_gradient_scores = select_best_z_by_gradient(z_stack, gfp_channel_idx)
    best_z_gfp_size, gfp_mean_areas = select_best_z_by_cell_size(z_stack, gfp_channel_idx)

    # mCherry channel
    best_z_mcherry_gradient, mcherry_gradient_scores = select_best_z_by_gradient(z_stack, mcherry_channel_idx)
    best_z_mcherry_size, mcherry_mean_areas = select_best_z_by_cell_size(z_stack, mcherry_channel_idx)

    # Collect all z-indices
    all_z_indices = [
        best_z_gfp_gradient,
        best_z_gfp_size,
        best_z_mcherry_gradient,
        best_z_mcherry_size
    ]

    # Take median
    best_z_median = int(round(np.median(all_z_indices)))

    results = {
        'gfp': {
            'gradient': {
                'best_z': best_z_gfp_gradient,
                'scores': gfp_gradient_scores,
            },
            'cell_size': {
                'best_z': best_z_gfp_size,
                'mean_areas': gfp_mean_areas,
            },
        },
        'mcherry': {
            'gradient': {
                'best_z': best_z_mcherry_gradient,
                'scores': mcherry_gradient_scores,
            },
            'cell_size': {
                'best_z': best_z_mcherry_size,
                'mean_areas': mcherry_mean_areas,
            },
        },
        'all_z_indices': all_z_indices,
        'combined_z': best_z_median,
    }

    return best_z_median, results


def select_best_z(z_stack, channel_idx, metric_func=variance_of_laplacian):
    """
    Select the best-focused z-slice from a z-stack.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use for focus metric
        metric_func: Focus metric function to use

    Returns:
        best_z: Index of best-focused z-slice
        scores: Focus scores for all z-slices
    """
    n_z = z_stack.shape[0]
    scores = []

    for z in range(n_z):
        image = z_stack[z, channel_idx, :, :]
        score = metric_func(image)
        scores.append(score)

    best_z = np.argmax(scores)
    return best_z, scores


# =============================================================================
# Exposure Time Extraction
# =============================================================================

def extract_exposure_times(nd2_file):
    """
    Extract exposure times for each channel from nd2 file metadata.

    Args:
        nd2_file: Path to nd2 file

    Returns:
        dict mapping channel name to exposure time in ms
    """
    import re

    exposure_times = {}

    with nd2.ND2File(nd2_file) as f:
        if hasattr(f, 'text_info') and f.text_info:
            text = f.text_info.get('description', '')

            # Parse exposure times from text
            # Pattern matches "Name: <channel>\n ... Exposure: <value> <unit>"
            planes = re.split(r'Plane #\d+:', text)

            for plane in planes[1:]:  # Skip first split (before Plane #1)
                # Extract channel name
                name_match = re.search(r'Name:\s*(\w+)', plane)
                if not name_match:
                    continue
                channel_name = name_match.group(1)

                # Extract exposure time
                exp_match = re.search(r'Exposure:\s*([\d.]+)\s*(ms|s)', plane)
                if exp_match:
                    value = float(exp_match.group(1))
                    unit = exp_match.group(2)
                    # Convert to ms
                    if unit == 's':
                        value *= 1000
                    exposure_times[channel_name] = value

    return exposure_times


def extract_nd2_metadata(nd2_file, time_index=None, position_index=None):
    """
    Extract all metadata from an nd2 file into a flat dictionary.

    Returns a dict suitable for saving as a single-row CSV.
    Per-channel fields are prefixed with ch{i}_.
    """
    import re

    meta = {}

    with nd2.ND2File(nd2_file) as f:
        # File-level
        meta['input_file'] = str(nd2_file)
        meta['shape'] = str(f.shape)
        meta['dtype'] = str(f.dtype)
        meta['is_legacy'] = f.is_legacy
        for dim, val in f.sizes.items():
            meta[f'size_{dim}'] = val

        # Voxel size
        vs = f.voxel_size()
        meta['voxel_x_um'] = vs.x
        meta['voxel_y_um'] = vs.y
        meta['voxel_z_um'] = vs.z

        # Attributes
        attrs = f.attributes
        meta['bits_per_component'] = attrs.bitsPerComponentSignificant
        meta['pixel_data_type'] = attrs.pixelDataType
        meta['width_px'] = attrs.widthPx
        meta['height_px'] = attrs.heightPx
        meta['sequence_count'] = attrs.sequenceCount

        # Text info
        ti = f.text_info if hasattr(f, 'text_info') and f.text_info else {}
        if isinstance(ti, dict):
            meta['date'] = ti.get('date', '')
            meta['optics'] = ti.get('optics', '')
            meta['description'] = ti.get('description', '')

        # Experiment loops
        try:
            exp = f.experiment
            if exp:
                for loop in exp:
                    loop_type = getattr(loop, 'type', type(loop).__name__)
                    meta[f'{loop_type}_count'] = loop.count
                    meta[f'{loop_type}_nesting'] = loop.nestingLevel
                    params = loop.parameters
                    for pattr in dir(params):
                        if not pattr.startswith('_'):
                            try:
                                pval = getattr(params, pattr)
                                if not callable(pval):
                                    meta[f'{loop_type}_{pattr}'] = pval
                            except Exception:
                                pass
        except Exception:
            pass

        # Parse per-channel exposure/binning from description
        exposure_times = {}
        binning = {}
        scan_modes = {}
        if 'description' in meta and meta['description']:
            planes = re.split(r'Plane #\d+:', meta['description'])
            for plane in planes[1:]:
                name_match = re.search(r'Name:\s*(\w+)', plane)
                if not name_match:
                    continue
                ch_name = name_match.group(1)
                exp_match = re.search(r'Exposure:\s*([\d.]+)\s*(ms|s)', plane)
                if exp_match:
                    val = float(exp_match.group(1))
                    if exp_match.group(2) == 's':
                        val *= 1000
                    exposure_times[ch_name] = val
                bin_match = re.search(r'Binning:\s*(\S+)', plane)
                if bin_match:
                    binning[ch_name] = bin_match.group(1)
                scan_match = re.search(r'Scan Mode:\s*(\w+)', plane)
                if scan_match:
                    scan_modes[ch_name] = scan_match.group(1)

        # Per-channel metadata
        try:
            channels = f.metadata.channels
        except Exception:
            channels = []

        meta['n_channels'] = len(channels)
        for i, ch in enumerate(channels):
            prefix = f'ch{i}'
            meta[f'{prefix}_name'] = ch.channel.name
            meta[f'{prefix}_index'] = ch.channel.index
            meta[f'{prefix}_color_r'] = ch.channel.color.r
            meta[f'{prefix}_color_g'] = ch.channel.color.g
            meta[f'{prefix}_color_b'] = ch.channel.color.b
            meta[f'{prefix}_emission_nm'] = ch.channel.emissionLambdaNm
            meta[f'{prefix}_excitation_nm'] = ch.channel.excitationLambdaNm

            # Microscope
            m = ch.microscope
            meta[f'{prefix}_objective_name'] = m.objectiveName
            meta[f'{prefix}_objective_mag'] = m.objectiveMagnification
            meta[f'{prefix}_objective_na'] = m.objectiveNumericalAperture
            meta[f'{prefix}_zoom_mag'] = m.zoomMagnification
            meta[f'{prefix}_immersion_ri'] = m.immersionRefractiveIndex
            meta[f'{prefix}_modality'] = str(m.modalityFlags)

            # Volume
            v = ch.volume
            meta[f'{prefix}_axes_calibration'] = str(v.axesCalibration)
            meta[f'{prefix}_component_data_type'] = v.componentDataType
            meta[f'{prefix}_voxel_count'] = str(v.voxelCount)

            # Exposure, binning, scan mode from parsed description
            ch_name = ch.channel.name
            meta[f'{prefix}_exposure_ms'] = exposure_times.get(ch_name, '')
            meta[f'{prefix}_binning'] = binning.get(ch_name, '')
            meta[f'{prefix}_scan_mode'] = scan_modes.get(ch_name, '')

        # Per-frame metadata (frame 0, or time_index)
        frame_idx = time_index if time_index is not None else 0
        if position_index is not None and 'P' in f.sizes:
            # Frames are ordered (T, P, Z): first frame of this position
            frame_idx = (frame_idx * f.sizes['P'] + position_index) * f.sizes.get('Z', 1)
            meta['position_index'] = position_index
        try:
            fm = f.frame_metadata(frame_idx)
            ch0 = fm.channels[0]

            # Stage position
            pos = ch0.position
            if pos and pos.stagePositionUm:
                meta['stage_x_um'] = pos.stagePositionUm.x
                meta['stage_y_um'] = pos.stagePositionUm.y
                meta['stage_z_um'] = pos.stagePositionUm.z
            meta['pfs_offset'] = pos.pfsOffset if pos else None

            # Timestamp
            ts = ch0.time
            if ts:
                meta['timestamp_julian'] = ts.absoluteJulianDayNumber
                meta['timestamp_relative_ms'] = ts.relativeTimeMs
        except Exception:
            pass

    return meta


def save_nd2_metadata(metadata_dict, output_path):
    """Save metadata dict as a single-row CSV table."""
    df = pd.DataFrame([metadata_dict])
    df.to_csv(output_path, index=False)
    print(f"  Metadata: {output_path}")


# =============================================================================
# Background Estimation and Channel Shift Functions
# =============================================================================

def calculate_background_stats(image, masks, iqr_factor=1.5):
    """
    Calculate background statistics from pixels outside cell masks.

    Rejects outlier pixels where |value - median| > iqr_factor * IQR.

    Args:
        image: Fluorescence image
        masks: Segmentation masks (0 = background)
        iqr_factor: Factor for outlier rejection (default: 1.5)

    Returns:
        dict with 'mean', 'std', 'median', 'n_pixels'
    """
    # Get background pixels (outside all masks)
    background_mask = masks == 0
    bg_pixels = image[background_mask].astype(np.float64)

    if len(bg_pixels) == 0:
        return {'mean': 0, 'std': 0, 'median': 0, 'n_pixels': 0}

    # Calculate median and IQR
    median = np.median(bg_pixels)
    q1 = np.percentile(bg_pixels, 25)
    q3 = np.percentile(bg_pixels, 75)
    iqr = q3 - q1

    # Reject outliers: |value - median| > iqr_factor * IQR
    valid_mask = np.abs(bg_pixels - median) <= iqr_factor * iqr
    valid_pixels = bg_pixels[valid_mask]

    if len(valid_pixels) == 0:
        valid_pixels = bg_pixels  # Fall back to all pixels

    return {
        'mean': np.mean(valid_pixels),
        'std': np.std(valid_pixels),
        'median': np.median(valid_pixels),
        'n_pixels': len(valid_pixels),
        'n_rejected': len(bg_pixels) - len(valid_pixels),
    }


def _robust_median_in_region(image, masks, y_start, y_end, x_start, x_end, iqr_factor=1.5):
    """Compute robust median of background pixels in a rectangular region."""
    region_img = image[y_start:y_end, x_start:x_end]
    region_masks = masks[y_start:y_end, x_start:x_end]
    bg_pixels = region_img[region_masks == 0].astype(np.float64)
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


def calculate_background(image, masks, iqr_factor=1.5):
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
    h, w = image.shape
    n_grid = 4

    y_edges = np.linspace(0, h, n_grid + 1, dtype=int)
    x_edges = np.linspace(0, w, n_grid + 1, dtype=int)

    quad_centers_y = []
    quad_centers_x = []
    quad_values = []

    for iy in range(n_grid):
        for ix in range(n_grid):
            y0, y1 = y_edges[iy], y_edges[iy + 1]
            x0, x1 = x_edges[ix], x_edges[ix + 1]
            cy = (y0 + y1) / 2.0
            cx = (x0 + x1) / 2.0
            med = _robust_median_in_region(image, masks, y0, y1, x0, x1, iqr_factor)
            quad_centers_y.append(cy)
            quad_centers_x.append(cx)
            quad_values.append(med)

    quad_centers_y = np.array(quad_centers_y)
    quad_centers_x = np.array(quad_centers_x)
    quad_values = np.array(quad_values)

    valid = ~np.isnan(quad_values)
    if valid.sum() < 6:
        # Not enough points for quadratic fit; fall back to global median
        bg_pixels = image[masks == 0].astype(np.float64)
        q1, q3 = np.percentile(bg_pixels, 25), np.percentile(bg_pixels, 75)
        iqr = q3 - q1
        inliers = bg_pixels[(bg_pixels >= q1 - iqr_factor * iqr) &
                             (bg_pixels <= q3 + iqr_factor * iqr)]
        med = np.median(inliers) if len(inliers) > 0 else np.median(bg_pixels)
        bkg_image = np.full_like(image, med, dtype=np.float64)
        return {
            'bkg_image': bkg_image,
            'median': med,
            'mean': med,
            'std': 0.0,
            'n_outliers': 0,
            'quadrant_values': quad_values.tolist(),
            'fit_coeffs': (0.0, 0.0, 0.0, 0.0, 0.0, med),
        }

    xv = quad_centers_x[valid]
    yv = quad_centers_y[valid]
    A = np.column_stack([xv**2, yv**2, xv * yv, xv, yv, np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, quad_values[valid], rcond=None)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    bkg_image = (coeffs[0] * xx**2 + coeffs[1] * yy**2 + coeffs[2] * xx * yy +
                 coeffs[3] * xx + coeffs[4] * yy + coeffs[5])

    fitted_at_centers = (coeffs[0] * xv**2 + coeffs[1] * yv**2 + coeffs[2] * xv * yv +
                         coeffs[3] * xv + coeffs[4] * yv + coeffs[5])
    residuals = quad_values[valid] - fitted_at_centers
    cx, cy = w / 2.0, h / 2.0
    center_value = float(coeffs[0] * cx**2 + coeffs[1] * cy**2 + coeffs[2] * cx * cy +
                         coeffs[3] * cx + coeffs[4] * cy + coeffs[5])

    bg_pixels = image[masks == 0].astype(np.float64)
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


def find_optimal_channel_shift(masks, target_image, shift_range=5):
    """
    Find the optimal global XY shift to maximize intensity within masks.

    Tests shifts from -shift_range to +shift_range in both X and Y.
    Chooses the shift that maximizes average percent increase in summed
    intensity across all cells.

    Args:
        masks: Segmentation masks
        target_image: Image to measure intensity from (e.g., GFP or mCherry)
        shift_range: Maximum shift in pixels (default: 5)

    Returns:
        dict with 'optimal_shift_x', 'optimal_shift_y', 'percent_increase',
              'all_shifts' (list of tested shifts and their results)
    """
    # Calculate baseline intensity at 0 shift
    cell_labels = np.unique(masks)
    cell_labels = cell_labels[cell_labels != 0]

    if len(cell_labels) == 0:
        return {
            'optimal_shift_x': 0,
            'optimal_shift_y': 0,
            'percent_increase': 0,
            'all_shifts': []
        }

    # Baseline: sum intensity for each cell at 0 shift
    baseline_sums = {}
    for label in cell_labels:
        cell_mask = masks == label
        baseline_sums[label] = np.sum(target_image[cell_mask])

    # Test all shifts
    all_shifts = []
    best_shift = (0, 0)
    best_avg_percent_increase = 0

    for dy in range(-shift_range, shift_range + 1):
        for dx in range(-shift_range, shift_range + 1):
            if dx == 0 and dy == 0:
                all_shifts.append({'dx': 0, 'dy': 0, 'avg_percent_increase': 0})
                continue

            # Shift the masks (equivalent to shifting the image in opposite direction)
            # We shift the masks because it's more efficient
            shifted_masks = np.roll(np.roll(masks, dy, axis=0), dx, axis=1)

            # Handle edge cases by zeroing out rolled-in pixels
            if dy > 0:
                shifted_masks[:dy, :] = 0
            elif dy < 0:
                shifted_masks[dy:, :] = 0
            if dx > 0:
                shifted_masks[:, :dx] = 0
            elif dx < 0:
                shifted_masks[:, dx:] = 0

            # Calculate percent change for each cell
            percent_changes = []
            for label in cell_labels:
                shifted_cell_mask = shifted_masks == label
                if np.sum(shifted_cell_mask) == 0:
                    continue  # Cell shifted out of frame

                shifted_sum = float(np.sum(target_image[shifted_cell_mask]))
                baseline = float(baseline_sums[label])

                # Only include cells with significant baseline signal
                if baseline > 1000:  # Minimum baseline threshold
                    percent_change = (shifted_sum - baseline) / baseline * 100
                    # Clamp to reasonable range to avoid overflow
                    percent_change = max(-100, min(1000, percent_change))
                    percent_changes.append(percent_change)

            if len(percent_changes) > 0:
                avg_percent_increase = np.mean(percent_changes)
            else:
                avg_percent_increase = -100  # Invalid shift

            all_shifts.append({
                'dx': dx,
                'dy': dy,
                'avg_percent_increase': avg_percent_increase
            })

            if avg_percent_increase > best_avg_percent_increase:
                best_avg_percent_increase = avg_percent_increase
                best_shift = (dx, dy)

    return {
        'optimal_shift_x': best_shift[0],
        'optimal_shift_y': best_shift[1],
        'percent_increase': best_avg_percent_increase,
        'all_shifts': all_shifts
    }


def apply_mask_shift(masks, dx, dy):
    """
    Apply a global shift to masks.

    Args:
        masks: Original masks
        dx: Shift in X direction (positive = right)
        dy: Shift in Y direction (positive = down)

    Returns:
        Shifted masks with edges zeroed
    """
    if dx == 0 and dy == 0:
        return masks.copy()

    shifted_masks = np.roll(np.roll(masks, dy, axis=0), dx, axis=1)

    # Zero out rolled-in edges
    if dy > 0:
        shifted_masks[:dy, :] = 0
    elif dy < 0:
        shifted_masks[dy:, :] = 0
    if dx > 0:
        shifted_masks[:, :dx] = 0
    elif dx < 0:
        shifted_masks[:, dx:] = 0

    return shifted_masks


# =============================================================================
# Core Functions
# =============================================================================

def load_nd2_zstack(filepath, time_index=None, position_index=None):
    """Load nd2 z-stack file and return data array and metadata.

    Args:
        filepath: Path to nd2 file
        time_index: If file is a time series, extract this frame (0-indexed).
                   If None and file has T dimension, uses frame 0.
    """
    print(f"Loading {filepath}...")
    with nd2.ND2File(filepath) as f:
        data = f.asarray()
        sizes = f.sizes
        try:
            channels = [c.channel.name for c in f.metadata.channels]
        except:
            channels = ['Ch0', 'Ch1', 'Ch2']

        # Handle time series
        if 'T' in sizes:
            n_timepoints = sizes['T']
            if time_index is None:
                time_index = 0
            print(f"  Time series detected: {n_timepoints} timepoints")
            print(f"  Extracting frame T={time_index}")
            data = data[time_index]  # Extract single timepoint
            # Update sizes to remove T
            sizes = {k: v for k, v in sizes.items() if k != 'T'}

        # Handle multi-position (XY) files
        if 'P' in sizes:
            n_positions = sizes['P']
            if position_index is None:
                position_index = 0
            print(f"  Multi-position file detected: {n_positions} positions")
            print(f"  Extracting position P={position_index}")
            data = data[position_index]
            sizes = {k: v for k, v in sizes.items() if k != 'P'}

        # Ensure data always has shape (Z, C, Y, X); add dummy Z=1 if no Z dimension
        if 'Z' not in sizes:
            data = data[np.newaxis, ...]   # (C, Y, X) → (1, C, Y, X)
            sizes = {'Z': 1, **sizes}

        print(f"  Shape: {data.shape}")
        print(f"  Dimensions: Z={sizes['Z']}, C={sizes['C']}, Y={sizes['Y']}, X={sizes['X']}")
        print(f"  Channels: {channels}")
    return data, sizes, channels


def segment_frame(model, image, params):
    """Segment a single frame using Omnipose."""
    cluster = params.get('cluster', True)
    try:
        masks, flows, styles = model.eval(
            image,
            diameter=params['diameter'],
            channels=params['channels'],
            flow_threshold=params.get('flow_threshold', 0.4),
            mask_threshold=params.get('mask_threshold', 0),
            min_size=params.get('min_size', 80),
            omni=params.get('omni', True),
            cluster=cluster,
        )
    except (ValueError, RuntimeError, IndexError) as e:
        if cluster:
            print(f"  Warning: clustering failed ({type(e).__name__}), retrying with fresh model...")
            fresh_model = omni_models.CellposeModel(model_type=params['model_type'], gpu=False)
            masks, flows, styles = fresh_model.eval(
                image,
                diameter=params['diameter'],
                channels=params['channels'],
                flow_threshold=params.get('flow_threshold', 0.4),
                mask_threshold=params.get('mask_threshold', 0),
                min_size=params.get('min_size', 80),
                omni=params.get('omni', True),
                cluster=False,
            )
        else:
            raise
    return masks


def measure_cells(masks, gfp_image, mcherry_image, gfp_bkg_stats=None, mcherry_bkg_stats=None,
                   gfp_shifted_masks=None, mcherry_shifted_masks=None, saturation_value=65535,
                   gfp_exposure_ms=None, mcherry_exposure_ms=None,
                   gfp_bkg_image=None, mcherry_bkg_image=None):
    """
    Measure properties of each cell with background subtraction and exposure normalization.

    Args:
        masks: Segmentation masks
        gfp_image: GFP fluorescence image
        mcherry_image: mCherry fluorescence image
        gfp_bkg_stats: Background statistics for GFP (legacy, used if gfp_bkg_image is None)
        mcherry_bkg_stats: Background statistics for mCherry (legacy, used if mcherry_bkg_image is None)
        gfp_shifted_masks: Shifted masks for GFP measurement (optional)
        mcherry_shifted_masks: Shifted masks for mCherry measurement (optional)
        saturation_value: Pixel value indicating saturation (default: 65535 for 16-bit)
        gfp_exposure_ms: GFP exposure time in ms (for rate calculation)
        mcherry_exposure_ms: mCherry exposure time in ms (for rate calculation)
        gfp_bkg_image: 2D background image for GFP (from calculate_background)
        mcherry_bkg_image: 2D background image for mCherry (from calculate_background)

    Returns a DataFrame with per-cell measurements including rates (intensity/exposure).
    """
    cell_data = []

    # Get unique cell labels (exclude background = 0)
    cell_labels = np.unique(masks)
    cell_labels = cell_labels[cell_labels != 0]

    # Use shifted masks if provided, otherwise use original
    gfp_masks = gfp_shifted_masks if gfp_shifted_masks is not None else masks
    mcherry_masks = mcherry_shifted_masks if mcherry_shifted_masks is not None else masks

    for label in cell_labels:
        # Create mask for this cell (original for centroid/area)
        cell_mask = masks == label

        # Masks for intensity measurement (possibly shifted)
        gfp_cell_mask = gfp_masks == label
        mcherry_cell_mask = mcherry_masks == label

        # Basic measurements (from original mask)
        area = np.sum(cell_mask)

        # Centroid (from original mask)
        y_coords, x_coords = np.where(cell_mask)
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)

        # GFP measurements (from possibly shifted mask)
        if np.sum(gfp_cell_mask) > 0:
            gfp_pixels = gfp_image[gfp_cell_mask]
            gfp_sum = np.sum(gfp_pixels)
            gfp_mean = np.mean(gfp_pixels)
            gfp_max = np.max(gfp_pixels)
            gfp_std = np.std(gfp_pixels)
            gfp_area = np.sum(gfp_cell_mask)
        else:
            gfp_sum = gfp_mean = gfp_max = gfp_std = 0
            gfp_area = 0

        # mCherry measurements (from possibly shifted mask)
        if np.sum(mcherry_cell_mask) > 0:
            mcherry_pixels = mcherry_image[mcherry_cell_mask]
            mcherry_sum = np.sum(mcherry_pixels)
            mcherry_mean = np.mean(mcherry_pixels)
            mcherry_max = np.max(mcherry_pixels)
            mcherry_std = np.std(mcherry_pixels)
            mcherry_area = np.sum(mcherry_cell_mask)
        else:
            mcherry_sum = mcherry_mean = mcherry_max = mcherry_std = 0
            mcherry_area = 0

        # Background subtraction for GFP
        if gfp_bkg_image is not None and gfp_area > 0:
            gfp_bkg_estimate = np.sum(gfp_bkg_image[gfp_cell_mask])
            gfp_bkg_error = 0
            gfp_sum_bkg_sub = gfp_sum - gfp_bkg_estimate
        elif gfp_bkg_stats is not None and gfp_area > 0:
            gfp_bkg_estimate = gfp_bkg_stats['mean'] * gfp_area
            gfp_bkg_error = gfp_bkg_stats['std'] / np.sqrt(gfp_area)
            gfp_sum_bkg_sub = gfp_sum - gfp_bkg_estimate
        else:
            gfp_bkg_estimate = 0
            gfp_bkg_error = 0
            gfp_sum_bkg_sub = gfp_sum

        # Background subtraction for mCherry
        if mcherry_bkg_image is not None and mcherry_area > 0:
            mcherry_bkg_estimate = np.sum(mcherry_bkg_image[mcherry_cell_mask])
            mcherry_bkg_error = 0
            mcherry_sum_bkg_sub = mcherry_sum - mcherry_bkg_estimate
        elif mcherry_bkg_stats is not None and mcherry_area > 0:
            mcherry_bkg_estimate = mcherry_bkg_stats['mean'] * mcherry_area
            mcherry_bkg_error = mcherry_bkg_stats['std'] / np.sqrt(mcherry_area)
            mcherry_sum_bkg_sub = mcherry_sum - mcherry_bkg_estimate
        else:
            mcherry_bkg_estimate = 0
            mcherry_bkg_error = 0
            mcherry_sum_bkg_sub = mcherry_sum

        # Calculate rates (bkg-subtracted intensity / exposure time)
        if gfp_exposure_ms is not None and gfp_exposure_ms > 0:
            gfp_rate = gfp_sum_bkg_sub / gfp_exposure_ms
        else:
            gfp_rate = gfp_sum_bkg_sub  # No normalization if exposure not available

        if mcherry_exposure_ms is not None and mcherry_exposure_ms > 0:
            mcherry_rate = mcherry_sum_bkg_sub / mcherry_exposure_ms
        else:
            mcherry_rate = mcherry_sum_bkg_sub  # No normalization if exposure not available

        # GFP/mCherry ratio using rates
        if mcherry_rate > 0:
            gfp_mcherry_ratio = gfp_rate / mcherry_rate
        else:
            gfp_mcherry_ratio = np.nan

        # Check for saturated pixels in each channel
        gfp_saturated = 1 if (np.sum(gfp_cell_mask) > 0 and np.any(gfp_image[gfp_cell_mask] >= saturation_value)) else 0
        mcherry_saturated = 1 if (np.sum(mcherry_cell_mask) > 0 and np.any(mcherry_image[mcherry_cell_mask] >= saturation_value)) else 0

        cell_data.append({
            'cell_label': label,
            'area_pixels': area,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            # GFP raw
            'gfp_sum': gfp_sum,
            'gfp_mean': gfp_mean,
            'gfp_max': gfp_max,
            'gfp_std': gfp_std,
            # GFP background
            'gfp_bkg_estimate': gfp_bkg_estimate,
            'gfp_bkg_error': gfp_bkg_error,
            'gfp_sum_bkg_sub': gfp_sum_bkg_sub,
            # GFP rate (bkg-sub / exposure)
            'gfp_rate': gfp_rate,
            # mCherry raw
            'mcherry_sum': mcherry_sum,
            'mcherry_mean': mcherry_mean,
            'mcherry_max': mcherry_max,
            'mcherry_std': mcherry_std,
            # mCherry background
            'mcherry_bkg_estimate': mcherry_bkg_estimate,
            'mcherry_bkg_error': mcherry_bkg_error,
            'mcherry_sum_bkg_sub': mcherry_sum_bkg_sub,
            # mCherry rate (bkg-sub / exposure)
            'mcherry_rate': mcherry_rate,
            # Ratio (using rates)
            'gfp_mcherry_ratio': gfp_mcherry_ratio,
            # Saturation warnings
            'gfp_saturated': gfp_saturated,
            'mcherry_saturated': mcherry_saturated,
        })

    return pd.DataFrame(cell_data)


def compute_summary_stats(cell_df):
    """Compute summary statistics for the cell population."""
    if len(cell_df) == 0:
        return {}

    summary = {
        'n_cells': len(cell_df),
        # GFP stats (raw sum per cell)
        'gfp_sum_mean': cell_df['gfp_sum'].mean(),
        'gfp_sum_std': cell_df['gfp_sum'].std(),
        'gfp_sum_median': cell_df['gfp_sum'].median(),
        # mCherry stats (raw sum per cell)
        'mcherry_sum_mean': cell_df['mcherry_sum'].mean(),
        'mcherry_sum_std': cell_df['mcherry_sum'].std(),
        'mcherry_sum_median': cell_df['mcherry_sum'].median(),
        # GFP/mCherry ratio stats (using rates)
        'ratio_mean': cell_df['gfp_mcherry_ratio'].mean(),
        'ratio_std': cell_df['gfp_mcherry_ratio'].std(),
        'ratio_median': cell_df['gfp_mcherry_ratio'].median(),
        # Area stats
        'area_mean': cell_df['area_pixels'].mean(),
        'area_std': cell_df['area_pixels'].std(),
    }

    # Add rate stats if available
    if 'gfp_rate' in cell_df.columns:
        summary['gfp_rate_mean'] = cell_df['gfp_rate'].mean()
        summary['gfp_rate_std'] = cell_df['gfp_rate'].std()
        summary['gfp_rate_median'] = cell_df['gfp_rate'].median()

    if 'mcherry_rate' in cell_df.columns:
        summary['mcherry_rate_mean'] = cell_df['mcherry_rate'].mean()
        summary['mcherry_rate_std'] = cell_df['mcherry_rate'].std()
        summary['mcherry_rate_median'] = cell_df['mcherry_rate'].median()

    return summary


# =============================================================================
# Visualization Functions
# =============================================================================

def save_background_visualization(gfp_image, mcherry_image, gfp_bkg_stats, mcherry_bkg_stats, output_path):
    """Save visualization of background model and background-subtracted images."""
    gfp_bkg_image = gfp_bkg_stats['bkg_image']
    mcherry_bkg_image = mcherry_bkg_stats['bkg_image']

    gfp_bkg_sub = np.clip(gfp_image.astype(np.float64) - gfp_bkg_image, 0, None)
    mcherry_bkg_sub = np.clip(mcherry_image.astype(np.float64) - mcherry_bkg_image, 0, None)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row, (raw, bkg, sub, name, center) in enumerate([
        (gfp_image, gfp_bkg_image, gfp_bkg_sub, 'GFP', gfp_bkg_stats['median']),
        (mcherry_image, mcherry_bkg_image, mcherry_bkg_sub, 'mCherry', mcherry_bkg_stats['median']),
    ]):
        vmin_raw = np.percentile(raw, 1)
        vmax_raw = np.percentile(raw, 99)
        cmap = 'Greens' if name == 'GFP' else 'Reds'

        axes[row, 0].imshow(raw, cmap=cmap, vmin=vmin_raw, vmax=vmax_raw)
        axes[row, 0].set_title(f'{name} raw')
        axes[row, 0].axis('off')

        im = axes[row, 1].imshow(bkg, cmap='hot')
        axes[row, 1].set_title(f'{name} background (center={center:.0f})')
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        vmax_sub = np.percentile(sub, 99)
        axes[row, 2].imshow(sub, cmap=cmap, vmin=0, vmax=vmax_sub)
        axes[row, 2].set_title(f'{name} background-subtracted')
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_DPI)
    plt.close(fig)


def save_segmentation_visualization(ph3_image, gfp_image, mcherry_image, masks,
                                     n_cells, best_z, focus_results, output_path):
    """Save visualization showing segmentation results."""
    outlines = utils.outlines_list(masks)

    # Determine display ranges for fluorescence channels:
    # - Black point: from entire image (1st percentile)
    # - White point: from pixels within cell masks (99th percentile)
    cell_mask_bool = masks > 0

    if np.any(cell_mask_bool):
        # GFP display range
        gfp_vmin = np.percentile(gfp_image, 1)
        gfp_vmax = np.percentile(gfp_image[cell_mask_bool], 99)

        # mCherry display range
        mcherry_vmin = np.percentile(mcherry_image, 1)
        mcherry_vmax = np.percentile(mcherry_image[cell_mask_bool], 99)
    else:
        # Fallback if no cells detected
        gfp_vmin, gfp_vmax = np.percentile(gfp_image, 1), np.percentile(gfp_image, 99)
        mcherry_vmin, mcherry_vmax = np.percentile(mcherry_image, 1), np.percentile(mcherry_image, 99)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Channels and segmentation
    # Ph3
    axes[0, 0].imshow(ph3_image, cmap='gray',
                      vmin=np.percentile(ph3_image, 1),
                      vmax=np.percentile(ph3_image, 99))
    ph3_z = focus_results.get('z_gfp_measure', best_z)
    axes[0, 0].set_title(f'Phase Contrast (z={ph3_z})')
    axes[0, 0].axis('off')

    # GFP
    axes[0, 1].imshow(gfp_image, cmap='Greens',
                      vmin=gfp_vmin, vmax=gfp_vmax)
    axes[0, 1].set_title('GFP')
    axes[0, 1].axis('off')

    # mCherry
    axes[0, 2].imshow(mcherry_image, cmap='Reds',
                      vmin=mcherry_vmin, vmax=mcherry_vmax)
    axes[0, 2].set_title('mCherry (segmentation channel)')
    axes[0, 2].axis('off')

    # Z-selection comparison
    ax = axes[0, 3]

    if isinstance(focus_results, dict) and focus_results.get('method') == 'brenner':
        # Brenner gradient method - show all 3 channels
        ph3_scores = focus_results['ph3']['scores']
        gfp_scores = focus_results['gfp']['scores']
        mcherry_scores = focus_results['mcherry']['scores']
        z_indices = range(len(ph3_scores))

        # Normalize for display
        ph3_norm = np.array(ph3_scores) / max(ph3_scores) if max(ph3_scores) > 0 else ph3_scores
        gfp_norm = np.array(gfp_scores) / max(gfp_scores) if max(gfp_scores) > 0 else gfp_scores
        mcherry_norm = np.array(mcherry_scores) / max(mcherry_scores) if max(mcherry_scores) > 0 else mcherry_scores

        ax.plot(z_indices, ph3_norm, 'b-o', label=f'Ph3 (z={focus_results["ph3"]["best_z"]})', markersize=4)
        ax.plot(z_indices, gfp_norm, 'g-s', label=f'GFP (z={focus_results["gfp"]["best_z"]})', markersize=4)
        ax.plot(z_indices, mcherry_norm, 'r-^', label=f'mCherry (z={focus_results["mcherry"]["best_z"]})', markersize=4)

        # Mark segmentation z and GFP measurement z
        ax.axvline(best_z, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Seg z={best_z}')
        if 'z_gfp_measure' in focus_results and focus_results['z_gfp_measure'] != best_z:
            ax.axvline(focus_results['z_gfp_measure'], color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'GFP z={focus_results["z_gfp_measure"]}')

        ax.set_xlabel('Z-slice')
        ax.set_ylabel('Normalized Brenner Score')
        ax.set_title('Focus Selection (Brenner gradient)')
        ax.legend(fontsize=7, loc='best')
        ax.set_ylim(-0.1, 1.1)

    elif isinstance(focus_results, dict) and 'ph3' in focus_results and 'mcherry' in focus_results and 'gfp' not in focus_results:
        # Dual channel high-pass gradient method (Ph3 + mCherry)
        ph3_scores = focus_results['ph3']['scores']
        mcherry_scores = focus_results['mcherry']['scores']
        z_indices = range(len(ph3_scores))

        # Normalize for display
        ph3_norm = np.array(ph3_scores) / max(ph3_scores) if max(ph3_scores) > 0 else ph3_scores
        mcherry_norm = np.array(mcherry_scores) / max(mcherry_scores) if max(mcherry_scores) > 0 else mcherry_scores

        ax.plot(z_indices, ph3_norm, 'b-o', label=f'Ph3 (z={focus_results["ph3"]["best_z"]})', markersize=4)
        ax.plot(z_indices, mcherry_norm, 'r-o', label=f'mCherry (z={focus_results["mcherry"]["best_z"]})', markersize=4)
        ax.axvline(best_z, color='black', linestyle='--', linewidth=2, label=f'Median z={best_z}')
        ax.set_xlabel('Z-slice')
        ax.set_ylabel('Normalized Gradient Score')
        ax.set_title('Focus Selection (high-pass gradient)')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.1)

    elif isinstance(focus_results, dict) and focus_results.get('method') == 'fluorescence':
        mcherry_scores = focus_results['mcherry']['scores']
        z_indices = range(len(mcherry_scores))
        mcherry_norm = np.array(mcherry_scores) / max(mcherry_scores) if max(mcherry_scores) > 0 else mcherry_scores
        ax.plot(z_indices, mcherry_norm, 'r-o', label=f'mCherry (z={best_z})', markersize=4)
        ax.axvline(best_z, color='black', linestyle='--', linewidth=2, label=f'Selected z={best_z}')
        ax.set_xlabel('Z-slice')
        ax.set_ylabel('Normalized Brenner Score')
        ax.set_title('Focus Selection (bkg-sub Brenner, mCherry)')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.1)

    elif isinstance(focus_results, dict) and 'gradient_scores' in focus_results:
        # Simple gradient-only method (mCherry only)
        gradient_scores = focus_results['gradient_scores']
        z_indices = range(len(gradient_scores))

        # Normalize for display
        grad_norm = np.array(gradient_scores) / max(gradient_scores) if max(gradient_scores) > 0 else gradient_scores

        ax.plot(z_indices, grad_norm, 'r-o', label='mCherry gradient', markersize=4)
        ax.axvline(best_z, color='black', linestyle='--', linewidth=2, label=f'Selected z={best_z}')
        ax.set_xlabel('Z-slice')
        ax.set_ylabel('Normalized Gradient Score')
        ax.set_title('Focus Selection (mCherry gradient)')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.1)

    elif isinstance(focus_results, dict) and 'gfp' in focus_results:
        # Dual channel method results
        gfp_grad_scores = focus_results['gfp']['gradient']['scores']
        mcherry_grad_scores = focus_results['mcherry']['gradient']['scores']

        z_indices = range(len(gfp_grad_scores))

        # Normalize gradient scores for comparison
        gfp_grad_norm = np.array(gfp_grad_scores) / max(gfp_grad_scores) if max(gfp_grad_scores) > 0 else gfp_grad_scores
        mcherry_grad_norm = np.array(mcherry_grad_scores) / max(mcherry_grad_scores) if max(mcherry_grad_scores) > 0 else mcherry_grad_scores

        ax.plot(z_indices, gfp_grad_norm, 'g-o', label=f'GFP grad (z={focus_results["gfp"]["gradient"]["best_z"]})', markersize=3, alpha=0.7)
        ax.plot(z_indices, mcherry_grad_norm, 'r-o', label=f'mCherry grad (z={focus_results["mcherry"]["gradient"]["best_z"]})', markersize=3, alpha=0.7)

        # Mark cell_size best z for each channel
        ax.axvline(focus_results['gfp']['cell_size']['best_z'], color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(focus_results['mcherry']['cell_size']['best_z'], color='red', linestyle=':', linewidth=1.5, alpha=0.7)

        # Mark final selection
        ax.axvline(best_z, color='black', linestyle='--', linewidth=2, label=f'Median z={best_z}')

        ax.set_xlabel('Z-slice')
        ax.set_ylabel('Normalized Gradient Score')
        ax.set_title(f'Focus Selection (4 methods)\nz=[{",".join(map(str, focus_results["all_z_indices"]))}]')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_ylim(-0.1, 1.1)

    elif isinstance(focus_results, dict) and 'gradient' in focus_results:
        # Single channel combined method results
        gradient_scores = focus_results['gradient']['scores']
        mean_areas = focus_results['cell_size']['mean_areas']
        best_z_grad = focus_results['gradient']['best_z']
        best_z_size = focus_results['cell_size']['best_z']

        z_indices = range(len(gradient_scores))

        # Normalize both metrics for comparison
        grad_norm = np.array(gradient_scores) / max(gradient_scores) if max(gradient_scores) > 0 else gradient_scores
        areas_array = np.array(mean_areas)
        areas_array[np.isinf(areas_array)] = np.nan
        if np.nanmax(areas_array) > 0:
            # Invert so higher = better (smaller area = better focus)
            area_norm = 1 - (areas_array - np.nanmin(areas_array)) / (np.nanmax(areas_array) - np.nanmin(areas_array))
        else:
            area_norm = np.zeros_like(areas_array)

        ax.plot(z_indices, grad_norm, 'b-o', label=f'Gradient (best={best_z_grad})', markersize=4)
        ax.plot(z_indices, area_norm, 'g-s', label=f'Cell size (best={best_z_size})', markersize=4)
        ax.axvline(best_z, color='red', linestyle='--', linewidth=2, label=f'Combined z={best_z}')
        ax.set_xlabel('Z-slice')
        ax.set_ylabel('Normalized Score')
        ax.set_title('Focus Selection Methods')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.1)
    elif isinstance(focus_results, dict) and focus_results.get('method', '').startswith('none'):
        # No z-stack: nothing to plot
        ax.text(0.5, 0.5, 'Single z (no autofocus)', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, color='gray')
        ax.set_title('Focus Selection')
        ax.axis('off')
    else:
        # Legacy single-method results (list of scores)
        z_scores = focus_results
        ax.bar(range(len(z_scores)), z_scores, color='steelblue')
        ax.axvline(best_z, color='red', linestyle='--', linewidth=2, label=f'Selected z={best_z}')
        ax.set_xlabel('Z-slice')
        ax.set_ylabel('Focus Score')
        ax.set_title('Z-slice Focus Scores')
        ax.legend()

    # Row 2: Overlays
    # mCherry with outlines
    axes[1, 0].imshow(mcherry_image, cmap='gray',
                      vmin=mcherry_vmin, vmax=mcherry_vmax)
    for outline in outlines:
        axes[1, 0].plot(outline[:, 0], outline[:, 1], 'r-', linewidth=0.5)
    axes[1, 0].set_title(f'mCherry + Segmentation ({n_cells} cells)')
    axes[1, 0].axis('off')

    # GFP with outlines
    axes[1, 1].imshow(gfp_image, cmap='gray',
                      vmin=gfp_vmin, vmax=gfp_vmax)
    for outline in outlines:
        axes[1, 1].plot(outline[:, 0], outline[:, 1], 'lime', linewidth=0.5)
    axes[1, 1].set_title('GFP + Segmentation')
    axes[1, 1].axis('off')

    # Cell masks
    axes[1, 2].imshow(masks, cmap='nipy_spectral')
    axes[1, 2].set_title('Cell Masks')
    axes[1, 2].axis('off')

    # Composite: GFP (green) + mCherry (red)
    # Use mask-based white point and image-based black point
    composite = np.zeros((*gfp_image.shape, 3), dtype=np.float32)
    gfp_norm = (gfp_image - gfp_vmin) / (gfp_vmax - gfp_vmin) if gfp_vmax > gfp_vmin else np.zeros_like(gfp_image)
    mcherry_norm = (mcherry_image - mcherry_vmin) / (mcherry_vmax - mcherry_vmin) if mcherry_vmax > mcherry_vmin else np.zeros_like(mcherry_image)
    composite[:, :, 0] = np.clip(mcherry_norm, 0, 1)  # Red
    composite[:, :, 1] = np.clip(gfp_norm, 0, 1)      # Green
    axes[1, 3].imshow(composite)
    axes[1, 3].set_title('Composite (GFP=green, mCherry=red)')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_DPI)
    plt.close(fig)


def save_z_crop_visualization(z_stack, masks, best_z, output_path, crop_size=100,
                              mcherry_channel=MCHERRY_CHANNEL):
    """Save visualization showing a cell crop across all z-slices."""
    n_z = z_stack.shape[0]
    img_h, img_w = z_stack.shape[2], z_stack.shape[3]

    # Get cell labels
    cell_labels = np.unique(masks)
    cell_labels = cell_labels[cell_labels != 0]

    if len(cell_labels) == 0:
        print("    No cells found, skipping z-crop visualization")
        return

    # Find cells with typical bacterial size
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

    # Pick cell near center
    all_cells.sort(key=lambda c: c['dist'])
    selected = all_cells[np.random.randint(0, min(10, len(all_cells)))]
    cy, cx = selected['cy'], selected['cx']

    # Crop bounds
    half = crop_size // 2
    y_start, y_end = int(max(0, cy - half)), int(min(img_h, cy + half))
    x_start, x_end = int(max(0, cx - half)), int(min(img_w, cx + half))

    # Create figure - show mCherry channel across z
    fig, axes = plt.subplots(1, min(n_z, 11), figsize=(3 * min(n_z, 11), 3))
    if n_z == 1:
        axes = [axes]

    # Sample z slices if too many
    if n_z > 11:
        z_indices = np.linspace(0, n_z - 1, 11, dtype=int)
    else:
        z_indices = range(n_z)

    best_crop = z_stack[best_z, mcherry_channel, y_start:y_end, x_start:x_end]
    vmin, vmax = np.percentile(best_crop, 1), np.percentile(best_crop, 99)

    for i, z in enumerate(z_indices):
        ax = axes[i]
        crop = z_stack[z, mcherry_channel, y_start:y_end, x_start:x_end]
        ax.imshow(crop, cmap='gray', vmin=vmin, vmax=vmax)

        if z == best_z:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)
            ax.set_title(f'z={z}\n(SELECTED)', fontsize=10, fontweight='bold', color='red')
        else:
            ax.set_title(f'z={z}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f'Cell #{selected["label"]} (area={selected["area"]} px) - mCherry across Z', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_DPI)
    plt.close(fig)


def save_intensity_histograms(cell_df, output_path):
    """Save histograms of GFP and mCherry intensities."""
    if len(cell_df) == 0:
        print("    No cells to plot histograms")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Sum intensities
    # GFP sum histogram
    axes[0, 0].hist(cell_df['gfp_sum'], bins=50, color='green', alpha=0.7, edgecolor='darkgreen')
    axes[0, 0].axvline(cell_df['gfp_sum'].mean(), color='black', linestyle='--', label=f'Mean: {cell_df["gfp_sum"].mean():.0f}')
    axes[0, 0].set_xlabel('GFP Sum Intensity')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('GFP Total Intensity per Cell')
    axes[0, 0].legend()

    # mCherry sum histogram
    axes[0, 1].hist(cell_df['mcherry_sum'], bins=50, color='red', alpha=0.7, edgecolor='darkred')
    axes[0, 1].axvline(cell_df['mcherry_sum'].mean(), color='black', linestyle='--', label=f'Mean: {cell_df["mcherry_sum"].mean():.0f}')
    axes[0, 1].set_xlabel('mCherry Sum Intensity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('mCherry Total Intensity per Cell')
    axes[0, 1].legend()

    # GFP/mCherry ratio histogram
    valid_ratios = cell_df['gfp_mcherry_ratio'].dropna()
    axes[0, 2].hist(valid_ratios, bins=50, color='purple', alpha=0.7, edgecolor='darkviolet')
    axes[0, 2].axvline(valid_ratios.mean(), color='black', linestyle='--', label=f'Mean: {valid_ratios.mean():.2f}')
    axes[0, 2].set_xlabel('GFP/mCherry Ratio')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('GFP/mCherry Ratio per Cell')
    axes[0, 2].legend()

    # Row 2: Mean intensities and scatter
    # GFP mean histogram
    axes[1, 0].hist(cell_df['gfp_mean'], bins=50, color='green', alpha=0.7, edgecolor='darkgreen')
    axes[1, 0].axvline(cell_df['gfp_mean'].mean(), color='black', linestyle='--', label=f'Mean: {cell_df["gfp_mean"].mean():.1f}')
    axes[1, 0].set_xlabel('GFP Mean Intensity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('GFP Mean Intensity per Cell')
    axes[1, 0].legend()

    # mCherry mean histogram
    axes[1, 1].hist(cell_df['mcherry_mean'], bins=50, color='red', alpha=0.7, edgecolor='darkred')
    axes[1, 1].axvline(cell_df['mcherry_mean'].mean(), color='black', linestyle='--', label=f'Mean: {cell_df["mcherry_mean"].mean():.1f}')
    axes[1, 1].set_xlabel('mCherry Mean Intensity')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('mCherry Mean Intensity per Cell')
    axes[1, 1].legend()

    # Scatter: GFP vs mCherry
    axes[1, 2].scatter(cell_df['mcherry_mean'], cell_df['gfp_mean'], alpha=0.5, s=20, c='blue')
    axes[1, 2].set_xlabel('mCherry Mean Intensity')
    axes[1, 2].set_ylabel('GFP Mean Intensity')
    axes[1, 2].set_title('GFP vs mCherry (per cell)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_DPI)
    plt.close(fig)


def save_ratio_mask_visualization(masks, cell_df, output_path, ratio_column='gfp_mcherry_ratio', invert=True):
    """
    Save visualization of masks colored by mCherry/GFP rate ratio.

    Args:
        masks: Segmentation mask array
        cell_df: DataFrame with cell measurements (must have 'cell_label' and ratio data)
        output_path: Path to save the visualization
        ratio_column: Column name for ratio (default: 'gfp_mcherry_ratio')
        invert: If True, display mCherry/GFP instead of GFP/mCherry
    """
    if len(cell_df) == 0:
        print("    No cells to plot ratio mask")
        return

    # Create ratio map using rate-based ratio
    ratio_map = np.zeros_like(masks, dtype=np.float64)

    for _, row in cell_df.iterrows():
        label = int(row['cell_label'])
        if invert:
            # mCherry/GFP rate ratio (inverted from gfp_mcherry_ratio)
            gfp_mcherry = row[ratio_column] if not np.isnan(row[ratio_column]) else 0
            ratio = 1.0 / gfp_mcherry if gfp_mcherry > 0 else 0
        else:
            ratio = row[ratio_column] if not np.isnan(row[ratio_column]) else 0

        ratio_map[masks == label] = ratio

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Mask background (where masks == 0)
    ratio_map_masked = np.ma.masked_where(masks == 0, ratio_map)

    # Use a diverging colormap centered around 1 (equal expression)
    # Values < 1: more GFP (green-ish), Values > 1: more mCherry (red-ish)
    vmin = np.percentile(ratio_map[masks > 0], 5) if np.any(masks > 0) else 0
    vmax = np.percentile(ratio_map[masks > 0], 95) if np.any(masks > 0) else 2

    # Use RdYlGn_r: red for high mCherry/GFP, green for low
    im = ax.imshow(ratio_map_masked, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('mCherry / GFP rate ratio', fontsize=12)

    # Calculate stats
    ratios = ratio_map[masks > 0]
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)

    ax.set_title(f'Cells colored by mCherry/GFP rate ratio\n'
                 f'Mean: {mean_ratio:.3f}, Median: {median_ratio:.3f}, n={len(cell_df)}',
                 fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_best_focus_composite(data, focus_results, output_path, gfp_channel=1, mcherry_channel=2):
    """
    Save a composite image with GFP and mCherry each at their best focus z-slice.

    Args:
        data: Z-stack array with shape (Z, C, Y, X)
        focus_results: Dict containing z positions for each channel
        output_path: Path to save the composite image
        gfp_channel: GFP channel index
        mcherry_channel: mCherry channel index
    """
    # Get best z for each channel
    z_gfp = focus_results.get('z_gfp_measure', focus_results.get('gfp', {}).get('best_z', 0))
    z_mcherry = focus_results.get('z_mcherry_measure', focus_results.get('mcherry', {}).get('best_z', 0))

    # Extract images at best focus for each channel
    gfp_image = data[z_gfp, gfp_channel, :, :].astype(np.float64)
    mcherry_image = data[z_mcherry, mcherry_channel, :, :].astype(np.float64)

    # Normalize to 0-1 range using percentile scaling
    def normalize_image(img, low_pct=1, high_pct=99.5):
        low = np.percentile(img, low_pct)
        high = np.percentile(img, high_pct)
        img_norm = (img - low) / (high - low + 1e-10)
        return np.clip(img_norm, 0, 1)

    gfp_norm = normalize_image(gfp_image)
    mcherry_norm = normalize_image(mcherry_image)

    # Create RGB composite: GFP=green, mCherry=red
    composite = np.zeros((*gfp_image.shape, 3), dtype=np.float32)
    composite[:, :, 0] = mcherry_norm  # Red channel
    composite[:, :, 1] = gfp_norm      # Green channel
    composite[:, :, 2] = 0             # Blue channel

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(composite)
    ax.set_title(f'Best Focus Composite\nGFP (green) z={z_gfp}, mCherry (red) z={z_mcherry}',
                fontsize=14)
    ax.axis('off')

    # Add colorbar legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'GFP (z={z_gfp})'),
        Patch(facecolor='red', label=f'mCherry (z={z_mcherry})'),
        Patch(facecolor='yellow', label='Colocalization'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_saturation_visualization(gfp_image, mcherry_image, masks, output_path, saturation_value=65535):
    """
    Save visualization showing saturated pixels within cell masks.

    Args:
        gfp_image: GFP fluorescence image
        mcherry_image: mCherry fluorescence image
        masks: Segmentation masks
        output_path: Path to save the visualization
        saturation_value: Pixel value indicating saturation (default: 65535 for 16-bit)
    """
    from skimage.segmentation import find_boundaries

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Get mask boundaries
    boundaries = find_boundaries(masks, mode='outer')
    cell_region = masks > 0

    # GFP saturation visualization
    # Create grayscale background from GFP
    gfp_norm = (gfp_image - np.percentile(gfp_image, 1)) / (np.percentile(gfp_image, 99) - np.percentile(gfp_image, 1) + 1e-10)
    gfp_norm = np.clip(gfp_norm, 0, 1)

    gfp_rgb = np.stack([gfp_norm * 0.5] * 3, axis=-1)  # Dim grayscale background

    # Highlight cell boundaries in white
    gfp_rgb[boundaries] = [1, 1, 1]

    # Highlight saturated pixels in bright green
    gfp_saturated_mask = (gfp_image >= saturation_value) & cell_region
    gfp_rgb[gfp_saturated_mask] = [0, 1, 0]  # Bright green for saturated

    n_gfp_sat_pixels = np.sum(gfp_saturated_mask)
    n_gfp_sat_cells = len(np.unique(masks[gfp_saturated_mask])) if n_gfp_sat_pixels > 0 else 0

    axes[0].imshow(gfp_rgb)
    axes[0].set_title(f'GFP Saturation\n{n_gfp_sat_pixels} saturated pixels in {n_gfp_sat_cells} cells', fontsize=12)
    axes[0].axis('off')

    # mCherry saturation visualization
    mcherry_norm = (mcherry_image - np.percentile(mcherry_image, 1)) / (np.percentile(mcherry_image, 99) - np.percentile(mcherry_image, 1) + 1e-10)
    mcherry_norm = np.clip(mcherry_norm, 0, 1)

    mcherry_rgb = np.stack([mcherry_norm * 0.5] * 3, axis=-1)  # Dim grayscale background

    # Highlight cell boundaries in white
    mcherry_rgb[boundaries] = [1, 1, 1]

    # Highlight saturated pixels in bright red
    mcherry_saturated_mask = (mcherry_image >= saturation_value) & cell_region
    mcherry_rgb[mcherry_saturated_mask] = [1, 0, 0]  # Bright red for saturated

    n_mcherry_sat_pixels = np.sum(mcherry_saturated_mask)
    n_mcherry_sat_cells = len(np.unique(masks[mcherry_saturated_mask])) if n_mcherry_sat_pixels > 0 else 0

    axes[1].imshow(mcherry_rgb)
    axes[1].set_title(f'mCherry Saturation\n{n_mcherry_sat_pixels} saturated pixels in {n_mcherry_sat_cells} cells', fontsize=12)
    axes[1].axis('off')

    plt.suptitle('Saturated Pixels within Cell Masks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_channel_shift_visualization(gfp_image, mcherry_image, masks, shift_result, output_path):
    """
    Save visualization comparing 0-shift vs optimal-shift composites.

    Args:
        gfp_image: GFP fluorescence image
        mcherry_image: mCherry fluorescence image
        masks: Original segmentation masks
        shift_result: Result from find_optimal_channel_shift
        output_path: Path to save the visualization
    """
    from skimage.segmentation import find_boundaries

    dx = shift_result['optimal_shift_x']
    dy = shift_result['optimal_shift_y']
    pct_increase = shift_result['percent_increase']

    # Get shifted masks
    shifted_masks = apply_mask_shift(masks, dx, dy)

    # Normalize images for display
    def normalize_image(img, low_pct=1, high_pct=99.5):
        low = np.percentile(img, low_pct)
        high = np.percentile(img, high_pct)
        img_norm = (img - low) / (high - low + 1e-10)
        return np.clip(img_norm, 0, 1)

    gfp_norm = normalize_image(gfp_image)
    mcherry_norm = normalize_image(mcherry_image)

    # Create composites
    composite_base = np.zeros((*gfp_image.shape, 3), dtype=np.float32)
    composite_base[:, :, 0] = mcherry_norm  # Red
    composite_base[:, :, 1] = gfp_norm      # Green

    # Get boundaries
    boundaries_original = find_boundaries(masks, mode='outer')
    boundaries_shifted = find_boundaries(shifted_masks, mode='outer')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original (0 shift)
    composite_orig = composite_base.copy()
    composite_orig[boundaries_original, :] = [1, 1, 1]  # White outlines
    axes[0].imshow(composite_orig)
    axes[0].set_title('Original (shift = 0, 0)\nGFP=green, mCherry=red', fontsize=12)
    axes[0].axis('off')

    # Optimal shift
    composite_shifted = composite_base.copy()
    composite_shifted[boundaries_shifted, :] = [0, 1, 1]  # Cyan outlines for shifted
    axes[1].imshow(composite_shifted)
    axes[1].set_title(f'Optimal Shift (dx={dx}, dy={dy})\n+{pct_increase:.1f}% avg intensity', fontsize=12)
    axes[1].axis('off')

    # Shift heatmap
    if len(shift_result['all_shifts']) > 0:
        # Create 2D heatmap of percent increases
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
        axes[2].set_title('Avg % Intensity Change', fontsize=12)
        axes[2].legend(loc='upper right', fontsize=9)
        plt.colorbar(im, ax=axes[2], label='% change')

    plt.suptitle('Channel Alignment Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=VIS_DPI, bbox_inches='tight')
    plt.close(fig)


def save_ratio_analysis(cell_df, output_path, gfp_exposure_ms=None, mcherry_exposure_ms=None):
    """Save detailed GFP/mCherry ratio analysis."""
    if len(cell_df) == 0:
        print("    No cells to plot ratio analysis")
        return

    valid_df = cell_df.dropna(subset=['gfp_mcherry_ratio'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Ratio histogram with more detail
    axes[0, 0].hist(valid_df['gfp_mcherry_ratio'], bins=50, color='purple', alpha=0.7, edgecolor='darkviolet')
    mean_ratio = valid_df['gfp_mcherry_ratio'].mean()
    std_ratio = valid_df['gfp_mcherry_ratio'].std()
    median_ratio = valid_df['gfp_mcherry_ratio'].median()
    axes[0, 0].axvline(mean_ratio, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_ratio:.3f}')
    axes[0, 0].axvline(median_ratio, color='gray', linestyle=':', linewidth=2, label=f'Median: {median_ratio:.3f}')
    axes[0, 0].set_xlabel('GFP/mCherry Rate Ratio')
    axes[0, 0].set_ylabel('Count')
    exp_info = ""
    if gfp_exposure_ms and mcherry_exposure_ms:
        exp_info = f"\n(GFP: {gfp_exposure_ms}ms, mCherry: {mcherry_exposure_ms}ms)"
    axes[0, 0].set_title(f'GFP/mCherry Rate Ratio Distribution{exp_info}\n(n={len(valid_df)}, std={std_ratio:.3f})')
    axes[0, 0].legend()

    # Log-scale ratio histogram
    log_ratios = np.log10(valid_df['gfp_mcherry_ratio'][valid_df['gfp_mcherry_ratio'] > 0])
    axes[0, 1].hist(log_ratios, bins=50, color='orange', alpha=0.7, edgecolor='darkorange')
    axes[0, 1].axvline(log_ratios.mean(), color='black', linestyle='--', label=f'Mean: {log_ratios.mean():.2f}')
    axes[0, 1].set_xlabel('log10(GFP/mCherry Rate Ratio)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Log-scale Rate Ratio Distribution')
    axes[0, 1].legend()

    # Ratio vs cell area
    axes[1, 0].scatter(valid_df['area_pixels'], valid_df['gfp_mcherry_ratio'], alpha=0.5, s=20, c='purple')
    axes[1, 0].set_xlabel('Cell Area (pixels)')
    axes[1, 0].set_ylabel('GFP/mCherry Rate Ratio')
    axes[1, 0].set_title('Rate Ratio vs Cell Size')

    # Summary stats text
    gfp_rate_mean = cell_df['gfp_rate'].mean() if 'gfp_rate' in cell_df.columns else 0
    mcherry_rate_mean = cell_df['mcherry_rate'].mean() if 'mcherry_rate' in cell_df.columns else 0
    stats_text = f"""Summary Statistics
==================
Number of cells: {len(valid_df)}

Exposure times:
  GFP:     {gfp_exposure_ms} ms
  mCherry: {mcherry_exposure_ms} ms

GFP/mCherry Rate Ratio:
  Mean:   {mean_ratio:.4f}
  Std:    {std_ratio:.4f}
  Median: {median_ratio:.4f}
  Min:    {valid_df['gfp_mcherry_ratio'].min():.4f}
  Max:    {valid_df['gfp_mcherry_ratio'].max():.4f}

GFP Rate (bkg-sub/exposure):
  Mean:   {gfp_rate_mean:.1f}

mCherry Rate (bkg-sub/exposure):
  Mean:   {mcherry_rate_mean:.1f}

Cell Area (pixels):
  Mean:   {cell_df['area_pixels'].mean():.1f}
  Std:    {cell_df['area_pixels'].std():.1f}
"""
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=VIS_DPI)
    plt.close(fig)


# =============================================================================
# Main Processing
# =============================================================================

def process_cloning_image(input_file, output_dir, params=CELLPOSE_PARAMS, force_z=None, time_index=None, highpass_cutoff=12.0, focus_method='fluorescence', z_shift=2, shift_range=5, no_shift=False, bkg_sub_segment=False, position_index=None):
    """
    Process a single z-stack cloning image.

    Args:
        input_file: Path to nd2 file
        output_dir: Output directory
        params: Cellpose parameters
        force_z: Force specific z-slice (None for automatic)
        time_index: Time index for time series
        highpass_cutoff: High-pass filter cutoff for focus detection
        focus_method: Focus detection method
        z_shift: Force constant z-shift between channels
        shift_range: Range for XY channel alignment optimization (default: 5)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    data, sizes, channels = load_nd2_zstack(input_file, time_index=time_index, position_index=position_index)
    n_z = sizes['Z']

    # Detect channel indices from metadata
    gfp_ch, ph3_ch, mcherry_ch = detect_channel_indices(channels)
    # Fall back to hardcoded defaults for any channel not found
    if gfp_ch is None:
        gfp_ch = GFP_CHANNEL
        print(f"  WARNING: GFP channel not found by name, using default index {gfp_ch}")
    if ph3_ch is None:
        ph3_ch = PH3_CHANNEL
        print(f"  WARNING: Ph3 channel not found by name, using default index {ph3_ch}")
    if mcherry_ch is None:
        mcherry_ch = MCHERRY_CHANNEL
        print(f"  WARNING: mCherry channel not found by name, using default index {mcherry_ch}")
    print(f"  Channel indices: GFP={gfp_ch}, Ph3={ph3_ch}, mCherry={mcherry_ch}")

    # Extract exposure times
    exposure_times = extract_exposure_times(input_file)
    gfp_exposure_ms = exposure_times.get('GFP', None)
    mcherry_exposure_ms = exposure_times.get('mCherry', None)
    print(f"\nExposure times:")
    print(f"  GFP: {gfp_exposure_ms} ms" if gfp_exposure_ms else "  GFP: not found")
    print(f"  mCherry: {mcherry_exposure_ms} ms" if mcherry_exposure_ms else "  mCherry: not found")

    # Extract and save nd2 metadata
    nd2_metadata = extract_nd2_metadata(input_file, time_index=time_index, position_index=position_index)
    save_nd2_metadata(nd2_metadata, output_dir / 'nd2_metadata.csv')

    # Select best z-slice
    # Convert cutoff to sigma (sigma = cutoff / 2.5)
    highpass_sigma = highpass_cutoff / 2.5

    if n_z == 1:
        # No z-stack: skip autofocus, use the single slice
        best_z = 0
        best_z_gfp_measure = 0
        focus_results = {
            'method': 'none (single z)',
            'mcherry': {'best_z': 0, 'scores': [1.0]},
            'combined_z': 0,
            'z_gfp_measure': 0,
            'z_mcherry_measure': 0,
            'z_segmentation': 0,
        }
        print(f"\nNo z-stack (Z=1): skipping autofocus, using z=0")
    elif force_z is not None:
        best_z = force_z
        # Still compute focus metrics for visualization (use highpass for viz)
        _, focus_results = select_best_z_dual_highpass_gradient(
            data, ph3_ch, mcherry_ch, highpass_sigma=highpass_sigma)
        focus_results['combined_z'] = force_z
        print(f"\nUsing forced z={best_z}")
    elif focus_method == 'brenner':
        # Use Brenner gradient - compute for all 3 channels
        best_z_mcherry, mcherry_scores = select_best_z_brenner(data, mcherry_ch)
        best_z_gfp, gfp_scores = select_best_z_brenner(data, gfp_ch)
        best_z_ph3, ph3_scores = select_best_z_brenner(data, ph3_ch)

        # Use mCherry best z for segmentation
        best_z = best_z_mcherry

        focus_results = {
            'method': 'brenner',
            'ph3': {'best_z': best_z_ph3, 'scores': ph3_scores},
            'gfp': {'best_z': best_z_gfp, 'scores': gfp_scores},
            'mcherry': {'best_z': best_z_mcherry, 'scores': mcherry_scores},
            'combined_z': best_z,
        }
        print(f"\nFocus selection (Brenner gradient):")
        print(f"  Ph3 best z:     {best_z_ph3}")
        print(f"  GFP best z:     {best_z_gfp}")
        print(f"  mCherry best z: {best_z_mcherry}")
        print(f"  Segmentation z: {best_z} (mCherry)")
    elif focus_method == 'variance':
        # Use normalized variance on mCherry channel only
        best_z, mcherry_scores = select_best_z_normalized_variance(data, mcherry_ch)
        _, ph3_scores = select_best_z_normalized_variance(data, ph3_ch)
        focus_results = {
            'ph3': {'best_z': np.argmax(ph3_scores), 'scores': ph3_scores},
            'mcherry': {'best_z': best_z, 'scores': mcherry_scores},
            'combined_z': best_z,
        }
        print(f"\nFocus selection (normalized variance on mCherry):")
        print(f"  Ph3 best z:     {focus_results['ph3']['best_z']}")
        print(f"  mCherry best z: {best_z}")
        print(f"  Selected z:     {best_z}")
    elif focus_method == 'highpass':
        best_z, focus_results = select_best_z_dual_highpass_gradient(
            data, ph3_ch, mcherry_ch, highpass_sigma=highpass_sigma)
        print(f"\nFocus selection (high-pass gradient, cutoff={highpass_cutoff}px):")
        print(f"  Ph3 best z:     {focus_results['ph3']['best_z']}")
        print(f"  mCherry best z: {focus_results['mcherry']['best_z']}")
        print(f"  Median z:       {best_z}")
    else:
        # Default: background-subtracted Brenner on mCherry (same as segment_2channel.py)
        best_z, scores = select_best_z_fluorescence(
            data, mcherry_ch,
            background_percentile=25, metric='brenner')
        focus_results = {
            'method': 'fluorescence',
            'mcherry': {'best_z': best_z, 'scores': scores},
            'combined_z': best_z,
        }
        print(f"\nFocus selection (background-subtracted Brenner on mCherry):")
        print(f"  Z-scores: {['%.1f' % s for s in scores]}")
        print(f"  mCherry best z: {best_z}")

    # Determine z for each channel (only relevant when n_z > 1)
    if n_z > 1:
        if z_shift is not None:
            # Force constant shift: GFP_z = mCherry_z + shift
            best_z_gfp_measure = max(0, min(n_z - 1, best_z + z_shift))
            print(f"  Using forced z-shift: GFP at z={best_z_gfp_measure} (mCherry z + {z_shift})")
        elif focus_method == 'brenner' and 'gfp' in focus_results:
            # Use GFP's own best z for measurement
            best_z_gfp_measure = focus_results['gfp']['best_z']
            print(f"  GFP measurement at z={best_z_gfp_measure}, mCherry at z={best_z}")
        else:
            # Default: use same z for all
            best_z_gfp_measure = best_z

    # Store z positions in focus_results
    focus_results['z_gfp_measure'] = best_z_gfp_measure
    focus_results['z_mcherry_measure'] = best_z
    focus_results['z_segmentation'] = best_z

    # Get images: mCherry at segmentation z, GFP and Ph3 at shifted z
    mcherry_image = data[best_z, mcherry_ch, :, :]
    gfp_image = data[best_z_gfp_measure, gfp_ch, :, :]
    ph3_image = data[best_z_gfp_measure, ph3_ch, :, :]

    # Initialize model and segment
    print(f"\nInitializing Omnipose model...")
    model = omni_models.CellposeModel(model_type=params['model_type'], gpu=False)
    print(f"  Model: {params['model_type']}")

    # Select channel for segmentation
    segment_channel = params.get('segment_channel', 'ph3')
    if segment_channel == 'mcherry':
        segment_image = mcherry_image
        print(f"  Segmenting on mCherry channel...")
    elif segment_channel == 'gfp':
        segment_image = gfp_image
        print(f"  Segmenting on GFP channel...")
    else:
        segment_image = ph3_image
        print(f"  Segmenting on Ph3 channel...")

    if bkg_sub_segment:
        empty_masks = np.zeros(segment_image.shape, dtype=np.int32)
        seg_bkg = calculate_background(segment_image, empty_masks)['bkg_image']
        seg_input = np.clip(segment_image.astype(np.float32) - seg_bkg, 0, None)
        seg_min, seg_max = seg_input.min(), seg_input.max()
        seg_input = (seg_input - seg_min) / (seg_max - seg_min)
        print(f"  Background-subtracted + min-max normalized for segmentation")
    else:
        seg_input = segment_image

    masks = segment_frame(model, seg_input, params)
    n_cells = len(np.unique(masks)) - 1  # Exclude background
    print(f"  Detected {n_cells} cells")

    # mCherry intensity refocus: scan all z-slices for brightest mCherry within masks
    # (only when there is a z-stack to scan)
    if n_cells > 0 and n_z > 1:
        print(f"\nmCherry intensity refocus: scanning z-slices for brightest mCherry in masks...")
        mask_bool = masks > 0
        z_max_mcherry = []
        for z in range(n_z):
            ch_slice = data[z, mcherry_ch, :, :]
            z_max_mcherry.append(float(ch_slice[mask_bool].max()) if np.any(mask_bool) else 0.0)
        best_z_refocus = int(np.argmax(z_max_mcherry))
        z_jump = abs(best_z_refocus - best_z)
        print(f"  Z-scores (max mCherry in masks): {['%.0f' % s for s in z_max_mcherry]}")
        print(f"  mCherry refocus z: {best_z_refocus} (was {best_z}, jump={z_jump})")

        # If refocus moved by >1 slice and initial segmentation found <6 cells,
        # re-segment at the new z to get a better mask
        if z_jump >= 2 and n_cells < 6:
            print(f"  Large z-jump ({z_jump}) with few cells ({n_cells}) — re-segmenting at z={best_z_refocus}...")
            reseg_image = data[best_z_refocus, mcherry_ch, :, :]
            masks = segment_frame(model, reseg_image, params)
            n_cells = len(np.unique(masks)) - 1
            print(f"  Re-segmentation detected {n_cells} cells")

        best_z = best_z_refocus
        focus_results['z_mcherry_measure'] = best_z
        focus_results['z_mcherry_refocus_scores'] = z_max_mcherry

        # Recompute GFP z from updated mCherry z
        if z_shift is not None:
            best_z_gfp_measure = max(0, min(n_z - 1, best_z + z_shift))
            print(f"  GFP z updated: {best_z_gfp_measure} (mCherry z + {z_shift})")
        focus_results['z_gfp_measure'] = best_z_gfp_measure
        focus_results['z_segmentation'] = best_z

        # Re-extract images at updated z positions
        mcherry_image = data[best_z, mcherry_ch, :, :]
        gfp_image = data[best_z_gfp_measure, gfp_ch, :, :]
        ph3_image = data[best_z_gfp_measure, ph3_ch, :, :]

    # Calculate background for each fluorescent channel (2D quadratic model)
    print("\nCalculating background...")
    gfp_bkg_stats = calculate_background(gfp_image, masks, iqr_factor=1.5)
    mcherry_bkg_stats = calculate_background(mcherry_image, masks, iqr_factor=1.5)
    print(f"  GFP background: center={gfp_bkg_stats['median']:.1f}, std={gfp_bkg_stats['std']:.1f}")
    print(f"  mCherry background: center={mcherry_bkg_stats['median']:.1f}, std={mcherry_bkg_stats['std']:.1f}")

    # Find optimal channel shift
    # Initialize default (no shift)
    gfp_shift_result = {'optimal_shift_x': 0, 'optimal_shift_y': 0, 'percent_increase': 0, 'all_shifts': []}
    mcherry_shift_result = {'optimal_shift_x': 0, 'optimal_shift_y': 0, 'percent_increase': 0, 'all_shifts': []}
    gfp_shifted_masks = masks
    mcherry_shifted_masks = masks

    if no_shift:
        print(f"\nChannel alignment: skipped (--no-shift)")
    elif segment_channel == 'ph3':
        # Segmented on Ph3, optimize shift for both GFP and mCherry
        gfp_shift_result = find_optimal_channel_shift(masks, gfp_image, shift_range=shift_range)
        mcherry_shift_result = find_optimal_channel_shift(masks, mcherry_image, shift_range=shift_range)

        print(f"  GFP optimal shift: dx={gfp_shift_result['optimal_shift_x']}, "
              f"dy={gfp_shift_result['optimal_shift_y']} (+{gfp_shift_result['percent_increase']:.1f}%)")
        print(f"  mCherry optimal shift: dx={mcherry_shift_result['optimal_shift_x']}, "
              f"dy={mcherry_shift_result['optimal_shift_y']} (+{mcherry_shift_result['percent_increase']:.1f}%)")

        gfp_shifted_masks = apply_mask_shift(masks,
                                              gfp_shift_result['optimal_shift_x'],
                                              gfp_shift_result['optimal_shift_y'])
        mcherry_shifted_masks = apply_mask_shift(masks,
                                                  mcherry_shift_result['optimal_shift_x'],
                                                  mcherry_shift_result['optimal_shift_y'])

    elif segment_channel == 'mcherry':
        # Segmented on mCherry, only optimize shift for GFP
        gfp_shift_result = find_optimal_channel_shift(masks, gfp_image, shift_range=shift_range)

        print(f"  GFP optimal shift: dx={gfp_shift_result['optimal_shift_x']}, "
              f"dy={gfp_shift_result['optimal_shift_y']} (+{gfp_shift_result['percent_increase']:.1f}%)")
        print(f"  mCherry: no shift (segmentation channel)")

        gfp_shifted_masks = apply_mask_shift(masks,
                                              gfp_shift_result['optimal_shift_x'],
                                              gfp_shift_result['optimal_shift_y'])
        # mCherry uses original masks (no shift needed)

    elif segment_channel == 'gfp':
        # Segmented on GFP, only optimize shift for mCherry
        mcherry_shift_result = find_optimal_channel_shift(masks, mcherry_image, shift_range=shift_range)

        print(f"  GFP: no shift (segmentation channel)")
        print(f"  mCherry optimal shift: dx={mcherry_shift_result['optimal_shift_x']}, "
              f"dy={mcherry_shift_result['optimal_shift_y']} (+{mcherry_shift_result['percent_increase']:.1f}%)")

        mcherry_shifted_masks = apply_mask_shift(masks,
                                                  mcherry_shift_result['optimal_shift_x'],
                                                  mcherry_shift_result['optimal_shift_y'])
        # GFP uses original masks (no shift needed)

    # Store shift results
    shift_results = {
        'gfp': gfp_shift_result,
        'mcherry': mcherry_shift_result
    }

    # Measure cells with background subtraction, shifted masks, and exposure normalization
    print("\nMeasuring cell intensities (with background subtraction and rate calculation)...")
    cell_df = measure_cells(masks, gfp_image, mcherry_image,
                            gfp_shifted_masks=gfp_shifted_masks,
                            mcherry_shifted_masks=mcherry_shifted_masks,
                            gfp_exposure_ms=gfp_exposure_ms,
                            mcherry_exposure_ms=mcherry_exposure_ms,
                            gfp_bkg_image=gfp_bkg_stats['bkg_image'],
                            mcherry_bkg_image=mcherry_bkg_stats['bkg_image'])

    # Report saturation warnings
    if len(cell_df) > 0:
        n_gfp_sat = cell_df['gfp_saturated'].sum()
        n_mcherry_sat = cell_df['mcherry_saturated'].sum()
        if n_gfp_sat > 0 or n_mcherry_sat > 0:
            print(f"  WARNING: Saturated cells detected!")
            print(f"    GFP saturated: {n_gfp_sat}/{len(cell_df)} cells")
            print(f"    mCherry saturated: {n_mcherry_sat}/{len(cell_df)} cells")

    # Compute summary
    summary = compute_summary_stats(cell_df)

    # Save outputs
    print("\nSaving outputs...")

    # Masks
    masks_path = output_dir / 'masks.tif'
    tifffile.imwrite(masks_path, masks.astype(np.uint16), compression='zlib')
    print(f"  Masks: {masks_path}")

    # Cell measurements
    cell_csv_path = output_dir / 'cell_measurements.csv'
    cell_df.to_csv(cell_csv_path, index=False)
    print(f"  Cell measurements: {cell_csv_path} ({len(cell_df)} cells)")

    # Summary stats
    summary_path = output_dir / 'summary_stats.csv'
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"  Summary stats: {summary_path}")

    # Z-selection data
    if isinstance(focus_results, dict) and focus_results.get('method') == 'fluorescence':
        mcherry_scores = focus_results['mcherry']['scores']
        z_df = pd.DataFrame({
            'z': range(len(mcherry_scores)),
            'mcherry_brenner_score': mcherry_scores,
            'selected': [z == best_z for z in range(len(mcherry_scores))]
        })
    elif isinstance(focus_results, dict) and 'ph3' in focus_results and 'mcherry' in focus_results:
        # Dual channel high-pass gradient method
        ph3_scores = focus_results['ph3']['scores']
        mcherry_scores = focus_results['mcherry']['scores']
        z_df = pd.DataFrame({
            'z': range(len(ph3_scores)),
            'ph3_gradient_score': ph3_scores,
            'mcherry_gradient_score': mcherry_scores,
            'selected': [z == best_z for z in range(len(ph3_scores))]
        })
    elif isinstance(focus_results, dict) and 'gradient_scores' in focus_results:
        # Simple gradient-only method
        gradient_scores = focus_results['gradient_scores']
        z_df = pd.DataFrame({
            'z': range(len(gradient_scores)),
            'gradient_score': gradient_scores,
            'selected': [z == best_z for z in range(len(gradient_scores))]
        })
    elif isinstance(focus_results, dict) and 'gfp' in focus_results:
        # Dual channel method
        gfp_grad = focus_results['gfp']['gradient']['scores']
        gfp_areas = focus_results['gfp']['cell_size']['mean_areas']
        mcherry_grad = focus_results['mcherry']['gradient']['scores']
        mcherry_areas = focus_results['mcherry']['cell_size']['mean_areas']
        z_df = pd.DataFrame({
            'z': range(len(gfp_grad)),
            'gfp_gradient_score': gfp_grad,
            'gfp_mean_cell_area': gfp_areas,
            'mcherry_gradient_score': mcherry_grad,
            'mcherry_mean_cell_area': mcherry_areas,
            'selected': [z == best_z for z in range(len(gfp_grad))]
        })
    elif isinstance(focus_results, dict) and 'gradient' in focus_results:
        gradient_scores = focus_results['gradient']['scores']
        mean_areas = focus_results['cell_size']['mean_areas']
        z_df = pd.DataFrame({
            'z': range(len(gradient_scores)),
            'gradient_score': gradient_scores,
            'mean_cell_area': mean_areas,
            'selected': [z == best_z for z in range(len(gradient_scores))]
        })
    else:
        # Single-z or unknown method: just record which z was used
        z_df = pd.DataFrame({'z': [best_z], 'selected': [True]})
    z_path = output_dir / 'z_selection.csv'
    z_df.to_csv(z_path, index=False)
    print(f"  Z-selection: {z_path}")

    # Visualizations
    print("\nGenerating visualizations...")

    save_segmentation_visualization(
        ph3_image, gfp_image, mcherry_image, masks,
        n_cells, best_z, focus_results,
        output_dir / 'segmentation.png'
    )
    print(f"  Segmentation: segmentation.png")

    save_z_crop_visualization(
        data, masks, best_z,
        output_dir / 'z_crop.png',
        mcherry_channel=mcherry_ch
    )
    print(f"  Z-crop: z_crop.png")

    save_intensity_histograms(cell_df, output_dir / 'intensity_histograms.png')
    print(f"  Histograms: intensity_histograms.png")

    save_ratio_analysis(cell_df, output_dir / 'ratio_analysis.png',
                        gfp_exposure_ms=gfp_exposure_ms, mcherry_exposure_ms=mcherry_exposure_ms)
    print(f"  Ratio analysis: ratio_analysis.png")

    # Best-focus composite (GFP and mCherry each at their optimal z)
    save_best_focus_composite(data, focus_results, output_dir / 'best_focus_composite.png',
                              gfp_channel=gfp_ch, mcherry_channel=mcherry_ch)
    print(f"  Best-focus composite: best_focus_composite.png")

    # Masks colored by mCherry/GFP ratio
    save_ratio_mask_visualization(masks, cell_df, output_dir / 'ratio_masks.png')
    print(f"  Ratio masks: ratio_masks.png")

    # Channel shift visualization (only for channels where shift was optimized)
    if segment_channel != 'gfp' and len(gfp_shift_result['all_shifts']) > 0:
        save_channel_shift_visualization(gfp_image, mcherry_image, masks, gfp_shift_result,
                                          output_dir / 'channel_shift_gfp.png')
        print(f"  Channel shift (GFP): channel_shift_gfp.png")

    if segment_channel != 'mcherry' and len(mcherry_shift_result['all_shifts']) > 0:
        save_channel_shift_visualization(gfp_image, mcherry_image, masks, mcherry_shift_result,
                                          output_dir / 'channel_shift_mcherry.png')
        print(f"  Channel shift (mCherry): channel_shift_mcherry.png")

    # Saturation visualization
    save_saturation_visualization(gfp_image, mcherry_image, masks,
                                   output_dir / 'saturation_map.png')
    print(f"  Saturation map: saturation_map.png")

    # Background subtracted visualization
    save_background_visualization(gfp_image, mcherry_image, gfp_bkg_stats, mcherry_bkg_stats,
                                   output_dir / 'background_subtracted.png')
    print(f"  Background subtracted: background_subtracted.png")

    # Save background, shift, and exposure parameters
    params_df = pd.DataFrame([{
        'gfp_exposure_ms': gfp_exposure_ms,
        'mcherry_exposure_ms': mcherry_exposure_ms,
        'gfp_bkg_median': gfp_bkg_stats['median'],
        'gfp_bkg_std': gfp_bkg_stats['std'],
        'gfp_bkg_n_outliers': gfp_bkg_stats['n_outliers'],
        'mcherry_bkg_median': mcherry_bkg_stats['median'],
        'mcherry_bkg_std': mcherry_bkg_stats['std'],
        'mcherry_bkg_n_outliers': mcherry_bkg_stats['n_outliers'],
        'gfp_shift_x': gfp_shift_result['optimal_shift_x'],
        'gfp_shift_y': gfp_shift_result['optimal_shift_y'],
        'gfp_shift_pct_increase': gfp_shift_result['percent_increase'],
        'mcherry_shift_x': mcherry_shift_result['optimal_shift_x'],
        'mcherry_shift_y': mcherry_shift_result['optimal_shift_y'],
        'mcherry_shift_pct_increase': mcherry_shift_result['percent_increase'],
    }])
    params_df.to_csv(output_dir / 'background_shift_params.csv', index=False)
    print(f"  Background/shift/exposure params: background_shift_params.csv")

    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Number of cells: {n_cells}")
    print(f"Best z-slice: {best_z}")
    print(f"\nExposure times:")
    print(f"  GFP:     {gfp_exposure_ms} ms" if gfp_exposure_ms else "  GFP:     not found")
    print(f"  mCherry: {mcherry_exposure_ms} ms" if mcherry_exposure_ms else "  mCherry: not found")
    print(f"\nBackground (pixel values, center of 2D quadratic fit):")
    print(f"  GFP:     center={gfp_bkg_stats['median']:.1f}, std={gfp_bkg_stats['std']:.1f}")
    print(f"  mCherry: center={mcherry_bkg_stats['median']:.1f}, std={mcherry_bkg_stats['std']:.1f}")
    print(f"\nChannel shift (optimal):")
    print(f"  GFP:     dx={gfp_shift_result['optimal_shift_x']}, dy={gfp_shift_result['optimal_shift_y']} (+{gfp_shift_result['percent_increase']:.1f}%)")
    print(f"  mCherry: dx={mcherry_shift_result['optimal_shift_x']}, dy={mcherry_shift_result['optimal_shift_y']} (+{mcherry_shift_result['percent_increase']:.1f}%)")
    print(f"\nGFP rate (bkg-sub intensity / exposure):")
    print(f"  Mean: {summary.get('gfp_rate_mean', 0):.1f}")
    print(f"  Std:  {summary.get('gfp_rate_std', 0):.1f}")
    print(f"\nmCherry rate (bkg-sub intensity / exposure):")
    print(f"  Mean: {summary.get('mcherry_rate_mean', 0):.1f}")
    print(f"  Std:  {summary.get('mcherry_rate_std', 0):.1f}")
    print(f"\nGFP/mCherry Rate Ratio:")
    print(f"  Mean:   {summary['ratio_mean']:.4f}")
    print(f"  Std:    {summary['ratio_std']:.4f}")
    print(f"  Median: {summary['ratio_median']:.4f}")
    print("="*50)

    return cell_df, summary, masks


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment and quantify cloning z-stack images')
    parser.add_argument('--input', type=str, required=True,
                        help='Input nd2 file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: based on input filename)')
    parser.add_argument('--force-z', type=int, default=None,
                        help='Force specific z-slice (default: automatic selection)')
    parser.add_argument('--model', type=str, default='bact_fluor_omni',
                        help='Omnipose model type (default: bact_fluor_omni)')
    parser.add_argument('--diameter', type=int, default=15,
                        help='Cell diameter for segmentation (default: 15)')
    parser.add_argument('--segment-channel', type=str, default='mcherry',
                        choices=['ph3', 'mcherry', 'gfp'],
                        help='Channel to segment on (default: mcherry)')
    parser.add_argument('--time-index', type=int, default=None,
                        help='For time series: which frame to process (0-indexed, default: 0)')
    parser.add_argument('--position', type=int, default=None,
                        help='For multi-position files: which XY position to process (0-indexed, default: 0)')
    parser.add_argument('--highpass-cutoff', type=float, default=12.0,
                        help='High-pass filter cutoff in pixels for focus detection (default: 12)')
    parser.add_argument('--focus-method', type=str, default='fluorescence',
                        choices=['fluorescence', 'highpass', 'brenner', 'variance'],
                        help='Focus detection method: fluorescence (default, bkg-sub Brenner on mCherry), highpass, brenner, or variance')
    parser.add_argument('--z-shift', type=int, default=2,
                        help='Force constant z-shift between GFP and mCherry (GFP_z = mCherry_z + shift, default: 2)')
    parser.add_argument('--shift-range', type=int, default=5,
                        help='Range for XY channel alignment optimization in pixels (default: 5)')
    parser.add_argument('--no-shift', action='store_true', default=False,
                        help='Skip XY channel alignment optimization entirely')
    parser.add_argument('--bkg-sub-segment', action='store_true', default=False,
                        help='Subtract 2D quadratic background and min-max normalize before segmentation')
    args = parser.parse_args()

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        input_path = Path(args.input)
        output_dir = Path('./segmentation_output') / f'cloning_{input_path.stem}'

    # Update params if specified
    params = CELLPOSE_PARAMS.copy()
    params['model_type'] = args.model
    params['diameter'] = args.diameter
    params['segment_channel'] = args.segment_channel

    print(f"Configuration:")
    print(f"  Input: {args.input}")
    print(f"  Output: {output_dir}")
    print(f"  Model: {params['model_type']}")
    print(f"  Diameter: {params['diameter']}")
    print(f"  Segment on: {params['segment_channel']}")
    if args.force_z is not None:
        print(f"  Forced z: {args.force_z}")

    # Process
    cell_df, summary, masks = process_cloning_image(
        args.input,
        output_dir,
        params=params,
        force_z=args.force_z,
        time_index=args.time_index,
        highpass_cutoff=args.highpass_cutoff,
        focus_method=args.focus_method,
        z_shift=args.z_shift,
        shift_range=args.shift_range,
        no_shift=args.no_shift,
        bkg_sub_segment=args.bkg_sub_segment,
        position_index=args.position,
    )

    print("\nDone!")
