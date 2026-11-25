# Bacterial segmentation and fluorescence-ratio pipeline

Segments bacteria in Nikon `.nd2` z-stacks with [Omnipose](https://github.com/kevinjohncutler/omnipose)
and measures, per cell, a **measurement channel** (GFP, YFP, …) against an **mCherry/mKate reference
channel**. The main output is a per-cell, exposure-normalized ratio of the two.

Main script: **`segment_cloning.py`**. It handles one field (one nd2 file, or one position or time
point of a multi-position or time-lapse file) per run.

## What it does

1. **Loads** the nd2 file and selects a time point (`--time-index`) or XY position (`--position`) if
   the file has more than one. Files without a z-stack are also supported.
2. **Identifies channels** by name: Phase (`Ph3`/`Phase`), the measurement channel (`GFP` or `YFP`,
   or the one you name with `--measure-channel`), and the reference (`mCherry`/`mKate`/`RFP`).
3. **Autofocus.** Picks the best z-slice on mCherry (default: background-subtracted Brenner score).
   The measurement channel is read `--z-shift` slices away from that (default +2) to correct for
   chromatic focal offset.
4. **Segments** cells with Omnipose (default model `bact_fluor_omni`, run on mCherry).
5. **Refocus.** Rescans the z-stack for the slice where mCherry inside the masks is brightest. If that
   moves the focus by more than one slice and fewer than 6 cells were found, it re-segments there.
6. **Background.** Fits a 2D quadratic background per channel from cell-free pixels and subtracts it.
7. **XY channel alignment.** Finds the integer pixel shift (within `--shift-range`) that best aligns the
   masks to each non-segmentation channel, and measures through the shifted masks.
8. **Measures** per cell: area, sums and means, background-subtracted sums, exposure-normalized
   rates (`sum_bkg_sub / exposure_ms`) and the ratio `gfp_rate / mcherry_rate`. Exposure times are read
   from the nd2 metadata.

## Installation

See [Dependencies](#dependencies). In short:

```bash
conda create -n cellpose python=3.10
conda activate cellpose
pip install -r requirements.txt
```

Omnipose downloads its model weights to `~/.cellpose/models/` on first use, so the first run needs
internet access.

## Usage

```bash
conda activate cellpose
python segment_cloning.py --input path/to/image.nd2 --output path/to/out_dir
```

### Examples

```bash
# GFP + mCherry, all defaults (measurement channel auto-detected)
python segment_cloning.py --input sample.nd2

# YFP as the measurement channel, with a custom label for plots
python segment_cloning.py --input "lasi 37.nd2" --measure-channel YFP --measure-label "LasI-YFP"

# Pick the measurement channel by index instead of by name
python segment_cloning.py --input sample.nd2 --measure-channel 1

# Multi-position file: process each XY position separately
for p in 0 1 2; do
  python segment_cloning.py --input LasI27.nd2 --position $p --output out/LasI27_p$((p+1))_out
done

# Measure both channels in the same z-slice (no chromatic z offset)
python segment_cloning.py --input sample.nd2 --z-shift 0

# Force a z-slice (e.g. if autofocus picks the wrong one)
python segment_cloning.py --input sample.nd2 --force-z 2
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Input `.nd2` file |
| `--output` | `./segmentation_output/cloning_<name>` | Output directory |
| `--measure-channel` | auto (GFP or YFP) | Channel to quantify against mCherry: a name (case-insensitive, e.g. `YFP`) or an index (e.g. `1`) |
| `--measure-label` | channel name in the file | Label for the measurement channel in plots and printed output |
| `--segment-channel` | `mcherry` | `mcherry`, `ph3`, or `gfp` (the measurement channel) |
| `--model` | `bact_fluor_omni` | Omnipose model (`bact_fluor_omni` for fluorescence, `bact_phase_omni` for phase) |
| `--diameter` | `15` | Expected cell diameter in pixels |
| `--position` | `0` | XY position to process in multi-position files (0-indexed) |
| `--time-index` | `0` | Time point to process in time-lapse files (0-indexed) |
| `--focus-method` | `fluorescence` | `fluorescence`, `highpass`, `brenner`, or `variance` |
| `--force-z` | auto | Use this z-slice instead of autofocus |
| `--z-shift` | `2` | Measurement-channel z = mCherry z + this value |
| `--highpass-cutoff` | `12` | High-pass cutoff (px) for the `highpass` focus method |
| `--shift-range` | `5` | Maximum XY shift (px) searched when aligning channels |
| `--no-shift` | off | Skip XY channel alignment |
| `--bkg-sub-segment` | off | Subtract background and normalize the image before segmentation |

#### Measurement channel and label

`--measure-channel` and `--measure-label` let the same code handle GFP, YFP or any other channel.

- With neither flag, the script looks for a channel named `GFP` or `YFP`, and labels plots with that
  channel's name from the nd2 file.
- `--measure-channel` overrides the detection. It accepts an exact or partial name, or an index. An
  ambiguous name (one that matches several channels) is an error.
- `--measure-label` only changes how the channel is shown in plots and printed output.

**Column names do not change.** CSV columns keep the `gfp_*` prefix (`gfp_rate`, `gfp_mcherry_ratio`,
…) whatever the channel, so the `collect_results*.py` scripts keep working. Each run records which
channel was measured and its label in `background_shift_params.csv` (`measure_channel`,
`measure_label`).

## Output files (per run)

| File | Content |
|---|---|
| `cell_measurements.csv` | One row per cell: area, centroid, `gfp_*` / `mcherry_*` sums, means, backgrounds, background-subtracted sums, rates, `gfp_mcherry_ratio`, saturation flags |
| `summary_stats.csv` | Per-image cell count and mean/median/std of the main quantities |
| `background_shift_params.csv` | Measurement channel and label, exposure times, background levels, XY shifts |
| `z_selection.csv` | Focus scores per z-slice and the chosen slices |
| `nd2_metadata.csv` | Flattened nd2 metadata (exposures, objective, stage position, …) |
| `masks.tif` | Label image (0 = background, 1..N = cells) |
| `segmentation.png`, `z_crop.png`, `best_focus_composite.png` | QC: masks, focus across z, composite |
| `background_subtracted.png`, `channel_shift_*.png`, `saturation_map.png` | QC: background, alignment, saturation |
| `intensity_histograms.png`, `ratio_analysis.png`, `ratio_masks.png` | Per-cell distributions |

## Collecting results across runs

| Script | Input layout |
|---|---|
| `collect_results_mkate.py` | `<input>/{gene}_T{time}_{rep}/{field}_out/cell_measurements.csv` (time courses) |
| `collect_results_mkate_ss.py` | `<input>/{gene}/{gene}_ss_{temp}_{rep}/{field}_out/…` (steady state) |
| `collect_results_timeseries.py` | `<input>/t{N}/cell_measurements.csv` (CellASIC time series) |
| `collect_results.py`, `collect_results_2ch.py` | Older folder conventions |

Run any of them with `--help` for options.

## Other scripts

- `autofocus.py`: focus metrics, imported by `segment_cloning.py`.
- `segment_2channel.py`: 2-channel (GFP + Phase) variant, imports helpers from `segment_cloning.py`.
- `parse_onix.py`: parses CellASIC ONIX experiment files.
- `compare_cloning_ratios.py`: compares ratios across cloning samples.

## Troubleshooting

- **`running_mean should contain 2 elements not 1` / `failed to load model`.** The model isn't an
  Omnipose model, for example `cyto3`. Use `bact_fluor_omni` or `bact_phase_omni`.
- **`shape mismatch … (2,) … (2,1)` in `omnipose/core.py follow_flows`.** This is an Omnipose bug.
  It shows up on fields with very few or very dim cells, or with large bright artifacts. Check the
  field. Usually it should be excluded.
- **Measurement exposure "not found".** The channel name isn't in the nd2 metadata description, so
  rates are not normalized. Check `nd2_metadata.csv`.
- **Autofocus picks the first or last slice.** Noisy background can inflate the focus score. Check
  `z_crop.png`, and use `--force-z` if needed.

## Dependencies

Tested with Python 3.10.18 on macOS (Apple Silicon, CPU only), in conda env `cellpose`.
Exact versions are in [`requirements.txt`](requirements.txt).

| Package | Tested version | Used for |
|---|---|---|
| `omnipose` | 1.0.6 | Segmentation. Provides `cellpose_omni` (models) and `omnipose` |
| `torch` / `torchvision` | 2.2.2 / 0.17.2 | Omnipose backend (CPU; a GPU is not required) |
| `nd2` | 0.11.1 | Reading Nikon `.nd2` files and metadata |
| `tifffile` | 2025.5.10 | Writing `masks.tif` |
| `opencv-python-headless` | 4.12.0.88 | Image filtering in focus metrics |
| `numpy` | 1.26.4 | Arrays (only numpy 1.x has been tested) |
| `scipy` | 1.15.3 | Filtering, background fits, statistics |
| `pandas` | 2.3.3 | Tables and CSV output |
| `matplotlib` | 3.10.8 | QC figures |
| `scikit-image` | 0.25.2 | Image utilities (Omnipose dependency) |

**Model weights.** These are downloaded automatically to `~/.cellpose/models/`.
`bact_fluor_omnitorch_0` is used by default, and `bact_phase_omnitorch_0` when segmenting on phase
with `--model bact_phase_omni`.

**Notes**

- The pipeline has only been tested with numpy 1.26 and torch 2.2. If a fresh install pulls numpy ≥ 2
  and Omnipose errors, pin `numpy<2`.
- Plain Cellpose (`cellpose` package) models such as `cyto3` and `cpsam` are **not** compatible with
  this script.
