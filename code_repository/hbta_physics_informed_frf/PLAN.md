# Concrete experiment plan

## Research question

Does a modal-physics prior improve estimation of the healthy complex FRF
distribution, and consequently one-class damage detection, when healthy bridge
recordings are limited?

## Phase 0 - immutable-source audit

- Inspect HDF5 recording names, attributes, durations, sensor channels, and
  shaker channels without writing to the source directory.
- Freeze recording roles using healthy data only.
- Save source metadata and file timestamps in `outputs/source_audit.json`.

## Phase 1 - FRF feasibility gate

- Estimate H1 acceleration FRFs from the shaker acceleration proxy to all 58
  vertical response channels.
- Use non-overlapping 60 s blocks after a fixed edge trim.
- Quantify coherence, usable frequency coverage, between-window repeatability,
  and between-recording healthy repeatability.
- Select the frequency band and candidate modal peaks using healthy recordings
  only.

Go if the dominant resonances have useful shaker-response coherence and stable
locations across healthy recordings. Otherwise stop or reformulate the study.

## Phase 2 - numerical verification

- Simulate a small linear MDOF system with measured input, response noise, and
  controlled stiffness reductions.
- Verify modal-FRF fitting, white-box bias under misspecification, and recovery
  by a GP discrepancy term.

## Phase 3 - three models

1. White-box: modal FRF plus observation noise.
2. Black-box: zero-mean GP over stacked real/imaginary FRFs.
3. Grey-box: modal FRF prior mean plus GP discrepancy.

The modal poles are shared across sensors. Model selection uses healthy
reference/development recordings only.

## Phase 4 - locked evaluation

- Fit with 1, 2, and 3 healthy reference recordings.
- Set the operational threshold on a separate healthy calibration recording.
- Evaluate false alarms on a held-out healthy recording.
- Open DS1-DS8 only for final per-state and macro detection evaluation.
- Report held-out healthy negative log predictive likelihood, macro-AUROC,
  false-alarm rate, TPR at the calibrated threshold, and data-scarcity curves.

## Stop criteria

- No stable coherent FRF band.
- Dominant healthy modal frequencies are not repeatable across recordings.
- GP discrepancy dominates the entire healthy signal so the modal mean has no
  measurable explanatory role.
- Grey-box gains are negligible at every healthy-data level.

## Gate outcome after the first experiment

The original HBTA-only modal+GP claim is stopped. Grey-box gains are negligible
within one recording, and all models fail under the separate healthy recording
domain shift. The retained next experiment must vary the number of independent
healthy environmental sessions, not merely the number of blocks from one
session. Details are recorded in `PRELIMINARY_RESULTS.md`.

The follow-up rank-one graph-GP experiment is also complete. Its scale-invariant
alignment score controls the external healthy false alarm and improves ranking,
but the physical graph does not beat shuffled-graph controls at the 5% level.
The result does not support a topology-caused detection claim. A localization
continuation requires a different benchmark with multiple known single-link
damages; see `GRAPH_DAMAGE_GP_RESULTS.md`.

The stricter modal physics-consistency extension is preserved in
`FUTURE_PHYSICS_CONSISTENCY_TEST.md`. It should be developed only as a separate
follow-up: the candidate edge must explain the measured modal-residue and
frequency changes through the first-order perturbation model, and must beat the
simple endpoint baseline under a fixed blind gate.

## Future extension: multi-edge or region localization

DS2--DS8 should not be forced into the current single-edge localization paper.
Their documented interventions are multi-link, member-level, or not mapped
one-to-one to a represented sensor edge. Preserve the proposed extension in
`FUTURE_MULTI_EDGE_LOCALIZATION.md`: model a damage set or region with

```text
Delta K = -sum_{e in S} rho_e k_e b_e b_e.T
```

and evaluate localization at the region, member, or multi-edge-set level after
establishing a new ground-truth mapping.
