# Code and derived data: Graph-Based Structural Damage Localization through Modal Relative Motion

Reproduction guide for the SDHM R2 manuscript. The package contains the
analysis code, the physical sensor graph, the derived modal features, and the
plotting scripts for every data figure of the revised manuscript.

## Layout

```
code_repository/hbta_physics_informed_frf/
    src/hbta_frf/        analysis package (FRF estimation, modal fitting, scores)
    topology/            physical sensor graph: 58 nodes, 122 candidate connections
    cache/               derived modal features (produced from the raw data, step A)
    outputs/             result files used by the figure scripts
    tests/               unit tests
latex/sdhm_revision_R2/
    scripts/             plotting scripts for the revised figures
    figures/             output folder of the plotting scripts
```

The relative layout must be kept, because the plotting scripts locate the
analysis package and the result files relative to their own position.

## Environment

Python 3.10 or later. Install the dependencies with

```
python -m pip install -r code_repository/hbta_physics_informed_frf/requirements.txt
```

The install command above is run from the repository root. Commands in steps A and B are run from `code_repository/hbta_physics_informed_frf`, with `src` on the Python path.

PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
```

Linux/macOS:

```bash
export PYTHONPATH="$PWD/src"
```

Write new outputs to a separate folder to preserve the included reference results.

## Raw data

The raw HBTA benchmark data (`data_100Hz.h5`, 2.5 GB) are not redistributed here.
They are openly available from Zenodo (doi:10.5281/zenodo.10507956, CC BY 4.0).
Only step A and Figure 8a need the raw file. Pass its path with `--source`, or,
for the Figure 8 script, set `HBTA_RAW_HDF5` to the absolute path of the downloaded raw file. Otherwise it looks for `data_100Hz.h5` in the current directory.

## Step A: derived features from the raw data (optional)

The files in `cache/` were produced from the raw data with the following modules:

```
python -m hbta_frf.prepare_frf_dataset --source RAW.h5 --output cache/p2_noise_frf_dataset.npz
python -m hbta_frf.extract_modal_features --dataset cache/p2_noise_frf_dataset.npz --output cache/p2_noise_modal_features.npz
python -m hbta_frf.run_frf_audit --source RAW.h5 --cache cache/healthy_p2_noise_frf_audit.npz --output-json outputs/frf_audit.json --output-figure outputs/frf_audit.png
python -m hbta_frf.run_ds1_sweep_validation --source RAW.h5 --noise-features cache/p2_noise_modal_features.npz --noise-audit cache/healthy_p2_noise_frf_audit.npz --edge-list topology/hbta_physical_edge_list.csv --output-cache cache/ds1_sweep_modal_features.npz --output-json outputs/ds1_sweep_validation.json --output-figure outputs/ds1_sweep_validation.png
python -m hbta_frf.run_all_damage_sweep_region --source RAW.h5 --noise-features cache/p2_noise_modal_features.npz --frf-dataset cache/p2_noise_frf_dataset.npz --edge-list topology/hbta_physical_edge_list.csv --output-cache cache/all_damage_sweep_modal_features.npz --output-json outputs/all_damage_sweep_region.json --output-figure outputs/all_damage_sweep_region.png
```

`run_ds1_sweep_validation` also runs the topology-permutation control of
Section 5.1.1 (1000 permutations) and writes it to `ds1_sweep_validation.json`.

## Step B: results from the derived features (minutes)

```
python -m hbta_frf.run_topk_spatial_localization --sweep-features cache/all_damage_sweep_modal_features.npz --noise-features cache/p2_noise_modal_features.npz --frf-dataset cache/p2_noise_frf_dataset.npz --edge-list topology/hbta_physical_edge_list.csv --output-json outputs/topk_spatial_localization.json --output-figure outputs/topk_spatial_localization.png
python -m hbta_frf.run_bandwidth_logscale_sweep --sweep-features cache/all_damage_sweep_modal_features.npz --noise-features cache/p2_noise_modal_features.npz --frf-dataset cache/p2_noise_frf_dataset.npz --edge-list topology/hbta_physical_edge_list.csv --output-json outputs/revision_r1/bandwidth_logscale_sweep.json
python -m hbta_frf.run_revision_r1_checks --sweep-features cache/all_damage_sweep_modal_features.npz --frf-dataset cache/p2_noise_frf_dataset.npz --edge-list topology/hbta_physical_edge_list.csv --ds1-sweep-json outputs/ds1_sweep_validation.json --output-json outputs/revision_r1/revision_r1_checks.json
```

## Step C: figures

Run from `latex/sdhm_revision_R2/scripts`:

| Figure | Content | Script |
|---|---|---|
| 2 | Edge-gradient concept | `make_fig_edge_gradient_R1.py` |
| 3 | Multi-reference fusion concept | `make_fig_modal_fusion_R1.py` |
| 4 | Spatial smoothing concept | `make_fig_smoothing_R1.py` |
| 6 | Sensor layout with physical bridge outlines | `make_fig_layout_R2.py` |
| 7 | Damage states | `make_fig_layout_R1.py` (also generates the unused R1 Figure 6) |
| 8 | FRF fit and fused mode shapes (needs raw data) | `make_fig_modal_quality_R1.py` |
| 9 | Fused DS1 spectrum | `make_fig_ds1_fused_R1.py` |
| 10–13 | Discrete and continuous benchmark, permutation, bandwidth | `make_fig_results_R1.py` |
| 14 | Pool-size sensitivity | `make_fig_poolsize_R1.py` |

Figures 1 (workflow diagram) and 5 (photograph) contain no data.

## Where each reported number comes from

| Result | Source |
|---|---|
| DS1 fused spectrum (Rank 1, s = 0.164) | `make_fig_ds1_fused_R1.py` via `scripts/r1_data.py` |
| Permutation p-values (Section 5.1.1) | `outputs/ds1_sweep_validation.json` |
| Discrete benchmark (Figure 10, Table A1) | `scripts/r1_data.py::discrete_benchmark`, written to `scripts/r1_discrete_benchmark.json` |
| Continuous benchmark (Figure 11, Table A1), K sensitivity | `outputs/topk_spatial_localization.json` |
| Bandwidth sensitivity (Figure 13) | `outputs/revision_r1/bandwidth_logscale_sweep.json` |
| Grid-resolution check, auto-MAC (Sections 2.2, 4.3) | `outputs/revision_r1/revision_r1_checks.json` |
| Fit quality (Section 4.1) | `weighted_nmse` and `median_coherence` fields of `cache/all_damage_sweep_modal_features.npz` |

## Tests

Install pytest separately for development checks. From `code_repository/hbta_physics_informed_frf`, with the Python path set as above:

```text
python -m pip install pytest
python -m pytest tests -q
```

## Manuscript compilation

PDF manuscripts and PDF journal-template assets are intentionally excluded from Git. To compile locally, obtain the complete journal template from the publisher and provide its PDF assets in `Definitions/`. With the required LaTeX packages and those local assets available, run twice from `latex/sdhm_revision_R2`:

```text
pdflatex -interaction=nonstopmode -halt-on-error graph_based_damage_localization_hbta_sdhm_R2.tex
```

The optional `tools/build_submission_R2.py` creates the clean PDF, a Track PDF versus the included R1 baseline source, and a clean-source ZIP. It requires `pdflatex`, `latexdiff` (or standalone `latexdiff-so`) and Perl. Its outputs stay inside the R2 directory.

## Version and checksum notes

[reproducibility_manifest_R2.json](reproducibility_manifest_R2.json) lists the selected R2 release files. Checksums use LF-normalized text and exact binary bytes, so Windows checkout line endings do not affect verification. The manifest omits its own checksum. Historical files elsewhere in this repository are outside this release manifest.
