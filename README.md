# Graph-Based Structural Damage Localization through Modal Relative Motion

Physics-informed, FEM-free graph localization of structural damage from measured
modal data, validated on the full-scale **Hell Bridge Test Arena (HBTA)**
benchmark.

**Authors:** Truong Thanh Chung¹², Tran Ngoc Hoa¹ (corresponding: `ngochoa@utc.edu.vn`), Bui Tien Thanh¹
¹ University of Transport and Communications, Hanoi, Vietnam
² Nha Trang University, Khanh Hoa, Vietnam

## Abstract

Vibration-based damage localization in civil infrastructure typically requires
high-fidelity calibrated finite-element models (FEMs) or extensive damaged
training data, both of which are rarely available in practice. This paper
develops a graph-based, model-free framework for structural damage
localization by mapping measured modal alterations directly onto a physical
sensor graph. Formulated from classical modal perturbation mechanics,
localized stiffness reduction along active load paths is shown to alter the
relative modal motion across connected sensor nodes, yielding a training-free
normalized edge-gradient damage score ($s_e$). When vibration data from
multiple excitation setups are available, an iterative multi-reference modal
vector fusion scheme synthesizes a consensus, high-signal-to-noise modal
representation without separate setup-specific calibrations.

The framework is experimentally validated on HBTA across two diagnostic
tiers: **(i) discrete single-connection diagnosis** — on an isolated bolted
joint disconnection (DS1), the fused edge score pinpoints the documented
connection as the Rank-1 candidate among all 122 load paths (AUROC = 1.00,
$p = 0.013$ under 1000 topology permutations); and **(ii) top-$K$ restricted
continuous spatial smoothing** — across the full DS1–DS8 benchmark sequence,
a Gaussian-pooled endpoint node-energy profile $R(x)$, restricted to the
$K=8$ highest-scoring candidate connections, localizes 7/9 physical targets
(77.8%) across 7/8 damage states (87.5%, $p=0.003$); the one multi-site
scenario (DS7) is not recovered and is reported as an open limitation.

## Repository layout

```
latex/                          Manuscript source (compiles to the final PDF)
  graph_based_damage_localization_hbta.tex   Main paper
  graph_based_damage_localization_hbta.pdf   Compiled paper
  figures/                      Figures included in the paper
  scripts/                      Python scripts that generate those figures

code_repository/
  hbta_physics_informed_frf/    Full reproduction pipeline for the experiments
    src/hbta_frf/               FRF/modal identification, graph scoring, spatial smoothing
    tests/                      Unit tests
    cache/                      Cached intermediate modal features (.npz)
    topology/                   Frozen 122-edge physical sensor-graph list (hbta_physical_edge_list.csv)
    outputs/                    Generated figures/JSON results, incl. the 3 bridge/sensor images used in the paper
    external_reference/         Minimal snapshots of prior-paper code, kept for provenance only
    README.md                   Study log: hypotheses, go/no-go gates, and findings
    PLAN.md, PRELIMINARY_RESULTS.md, GRAPH_DAMAGE_GP_RESULTS.md,
    DS1_LOCALIZATION_AUDIT.md, DS1_SWEEP_VALIDATION.md, ...   Research audit trail

papers/                         Reference literature (not this paper's own output)
un_use/                         Superseded drafts, QA renders, and an unrelated
                                 exploratory project — kept, not deleted, in case
                                 anything is needed later
```

## Reproducing the results

The experiments run against the HBTA acceleration dataset. Per
[`code_repository/hbta_physics_informed_frf/README.md`](code_repository/hbta_physics_informed_frf/README.md),
the raw data stay outside this repository (read-only, `data_100Hz.h5`) and are
never copied in; the `cache/*.npz` files here hold the modal features already
extracted from it, and `topology/hbta_physical_edge_list.csv` holds the frozen
122-edge sensor graph, so the downstream localization/scoring/plotting steps
run end-to-end without needing the raw recordings.

```bash
cd code_repository/hbta_physics_informed_frf
pip install -r requirements.txt
pip install -e .          # installs the hbta_frf package (see src/hbta_frf)
pytest tests/             # unit tests

# main multi-reference spatial-localization pipeline (writes JSON + figure)
python -m hbta_frf.run_multireference_spatial_localization \
  --sweep-features cache/all_damage_sweep_modal_features.npz \
  --noise-features cache/p2_noise_modal_features.npz \
  --frf-dataset cache/p2_noise_frf_dataset.npz \
  --edge-list topology/hbta_physical_edge_list.csv \
  --output-json outputs/multireference_spatial_localization.json \
  --output-figure outputs/multireference_spatial_localization.png \
  --permutations 1000
```

This reproduces `outputs/multireference_spatial_localization.json` byte-for-byte
against the version already committed here — the pipeline is fully
deterministic given the committed cache and edge list. Every other
`run_*.py` script's `--edge-list` argument takes the same
`topology/hbta_physical_edge_list.csv` file.

Table 2 and Figure 11 in the manuscript are generated by
`hbta_frf.run_topk_spatial_localization` (same argument pattern as above),
which reproduces `outputs/topk_spatial_localization.json`.

See the individual `run_*.py` modules under `src/hbta_frf/` for each
experiment (DS1 confirmatory audit, full DS1–DS8 benchmark sweep, robustness
and sensitivity audits) and the corresponding `*.md` files for what each run
was testing and its outcome.

To recompile the manuscript and its figures:

```bash
cd latex/scripts
python make_fig_modal_quality_master.py   # and the other make_fig_*.py scripts
cd ..
latexmk -pdf graph_based_damage_localization_hbta.tex
```

## Data availability

Raw HBTA acceleration records are not distributed in this repository; see
[`code_repository/hbta_physics_informed_frf/README.md`](code_repository/hbta_physics_informed_frf/README.md)
for the data boundary and provenance notes.
