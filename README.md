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
$p = 0.013$ under 1000 topology permutations); and **(ii) generalized
continuous spatial smoothing** — across the full DS1–DS8 benchmark sequence,
Gaussian kernel smoothing $R(x)$ reliably localizes 8/9 physical targets
(88.9%) across 7/8 damage states (87.5%).

## Repository layout

```
latex/                          Manuscript source (compiles to the final PDF)
  spatial_intelligence_connection_localization.tex   Main paper
  spatial_intelligence_connection_localization.pdf   Compiled paper
  figures/                      Figures included in the paper
  scripts/                      Python scripts that generate those figures
  backups/                      Snapshot of the manuscript before the last section restructure

code_repository/
  hbta_physics_informed_frf/    Full reproduction pipeline for the experiments
    src/hbta_frf/               FRF/modal identification, graph scoring, spatial smoothing
    tests/                      Unit tests
    cache/                      Cached intermediate modal features (.npz)
    outputs/                    Generated figures/JSON results, incl. the 3 bridge/sensor images used in the paper
    external_reference/         Minimal snapshots of prior-paper code, kept for provenance only
    README.md                   Study log: hypotheses, go/no-go gates, and findings
    PLAN.md, PRELIMINARY_RESULTS.md, GRAPH_DAMAGE_GP_RESULTS.md,
    DS1_LOCALIZATION_AUDIT.md, DS1_SWEEP_VALIDATION.md, ...   Research audit trail

output/pdf/                     Standalone copy of the final compiled paper
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
extracted from it, so the downstream localization/scoring/plotting steps run
without needing the raw recordings.

```bash
cd code_repository/hbta_physics_informed_frf
pip install -r requirements.txt
pip install -e .          # installs the hbta_frf package (see src/hbta_frf)
pytest tests/             # unit tests
python -m hbta_frf.run_multireference_spatial_localization   # main localization pipeline
```

See the individual `run_*.py` modules under `src/hbta_frf/` for each
experiment (DS1 confirmatory audit, full DS1–DS8 benchmark sweep, robustness
and sensitivity audits) and the corresponding `*.md` files for what each run
was testing and its outcome.

To recompile the manuscript and its figures:

```bash
cd latex/scripts
python make_fig_modal_quality_master.py   # and the other make_fig_*.py scripts
cd ..
latexmk -pdf spatial_intelligence_connection_localization.tex
```

## Data availability

Raw HBTA acceleration records are not distributed in this repository; see
[`code_repository/hbta_physics_informed_frf/README.md`](code_repository/hbta_physics_informed_frf/README.md)
for the data boundary and provenance notes.
