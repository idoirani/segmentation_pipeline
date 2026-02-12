"""
Autofocus utilities for z-stack microscopy images.

Provides multiple focus metric functions and z-selection strategies.
All select_best_z_* functions expect z_stack with shape (Z, C, Y, X).
"""

import numpy as np
import cv2
from scipy import ndimage


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

    grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

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

    if mask_outliers:
        valid_mask = mask_outlier_pixels(img_float, iqr_factor)
    else:
        valid_mask = np.ones_like(img_float, dtype=bool)

    low_pass = ndimage.gaussian_filter(img_float, sigma=highpass_sigma)
    high_pass = img_float - low_pass

    grad_x = cv2.Sobel(high_pass, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(high_pass, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    return np.mean(grad_mag[valid_mask])


# =============================================================================
# Z-selection Functions
# =============================================================================

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
    best_z_ph3, ph3_scores = select_best_z_highpass_gradient(z_stack, ph3_channel_idx, highpass_sigma, mask_outliers, iqr_factor)
    best_z_mcherry, mcherry_scores = select_best_z_highpass_gradient(z_stack, mcherry_channel_idx, highpass_sigma, mask_outliers, iqr_factor)

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

        try:
            thresh = threshold_otsu(image)
        except:
            mean_areas.append(np.inf)
            continue

        binary = image > thresh

        labeled, n_objects = ndimage.label(binary)

        if n_objects == 0:
            mean_areas.append(np.inf)
            continue

        areas = ndimage.sum(binary, labeled, range(1, n_objects + 1))

        valid_areas = [a for a in areas if 50 < a < 5000]

        if len(valid_areas) == 0:
            mean_areas.append(np.inf)
        else:
            mean_areas.append(np.mean(valid_areas))

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
    best_z_gradient, gradient_scores = select_best_z_by_gradient(z_stack, channel_idx)
    best_z_size, mean_areas = select_best_z_by_cell_size(z_stack, channel_idx)

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
    best_z_gfp_gradient, gfp_gradient_scores = select_best_z_by_gradient(z_stack, gfp_channel_idx)
    best_z_gfp_size, gfp_mean_areas = select_best_z_by_cell_size(z_stack, gfp_channel_idx)

    best_z_mcherry_gradient, mcherry_gradient_scores = select_best_z_by_gradient(z_stack, mcherry_channel_idx)
    best_z_mcherry_size, mcherry_mean_areas = select_best_z_by_cell_size(z_stack, mcherry_channel_idx)

    all_z_indices = [
        best_z_gfp_gradient,
        best_z_gfp_size,
        best_z_mcherry_gradient,
        best_z_mcherry_size
    ]

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


# =============================================================================
# Background-subtracted autofocus (for fluorescence channels)
# =============================================================================

def select_best_z_fluorescence(z_stack, channel_idx, background_percentile=25,
                                metric='brenner'):
    """
    Select best z-slice for a fluorescence channel after subtracting
    the background (estimated as a percentile of pixel values).

    This improves autofocus by removing the uniform background floor
    so the metric responds only to the fluorescent signal.

    Args:
        z_stack: Array with shape (Z, C, Y, X)
        channel_idx: Which channel to use
        background_percentile: Percentile to use as background estimate (default: 25)
        metric: Focus metric to use ('brenner', 'gradient', 'laplacian', 'normalized_variance')

    Returns:
        best_z: Index of best-focused z-slice
        scores: Focus scores for all z-slices
    """
    n_z = z_stack.shape[0]
    scores = []

    metric_funcs = {
        'laplacian': variance_of_laplacian,
        'gradient': normalized_gradient_magnitude,
    }

    for z in range(n_z):
        image = z_stack[z, channel_idx, :, :].astype(np.float64)

        # Subtract background
        bkg = np.percentile(image, background_percentile)
        image = np.clip(image - bkg, 0, None)

        if metric == 'brenner':
            score = np.sum((image[:, 2:] - image[:, :-2]) ** 2)
        elif metric == 'normalized_variance':
            score = image.std() / image.mean() if image.mean() > 0 else 0
        elif metric in metric_funcs:
            score = metric_funcs[metric](image)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        scores.append(score)

    best_z = np.argmax(scores)
    return best_z, scores
