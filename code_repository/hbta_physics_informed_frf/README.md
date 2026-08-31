# Physics-Informed FRF GP for HBTA

Isolated research workspace for testing a single hypothesis:

> Does a modal-physics prior improve healthy frequency-response modelling and
> one-class damage detection when only limited healthy bridge recordings are
> available?

## Data boundary

- Raw HBTA data remain read-only at
  `D:\BIG DATA\Hell Bridge Test Arena HBTA\data_100Hz.h5`.
- Raw data are never copied into this repository.
- Original code under the prior paper repositories is never executed or edited.
- Minimal source snapshots are copied into `external_reference/` for provenance
  only. Production code must not import from that directory.

## Study boundary

This first study covers measured shaker-to-response frequency-response
functions, a modal white-box model, a black-box GP, and a grey-box GP with a
modal prior mean. It does not study topology, localization, environmental
normalization, sensor outages, or restoration.

## Go/no-go gates

1. The shaker and bridge responses must yield repeatable, coherent FRFs in a
   defensible frequency band.
2. Dominant healthy modal frequencies must be sufficiently stable across
   independently acquired healthy recordings.
3. The grey-box model must improve held-out healthy predictive likelihood or
   damage discrimination under limited healthy data relative to both white-box
   and black-box baselines.

All generated artifacts belong under `outputs/` or `cache/` in this directory.

## Current outcome

The HBTA feasibility gate passed, but the first modal+GP paper claim did not.
Same-session damage detection is saturated for every baseline, whereas the
separately acquired healthy noise recording causes 100% false alarms for every
baseline. See `PRELIMINARY_RESULTS.md` for the evidence and go/no-go decision.

A subsequent rank-one graph-GP prototype found a robust scale-invariant damage
ranking signal, but the physical graph did not significantly outperform 100
sensor-shuffled controls. See `GRAPH_DAMAGE_GP_RESULTS.md`. The current decision
is no-go for a general detection claim and conditional go for localization only
if several uniquely mapped damage cases can be obtained from another benchmark.

A locked DS1-only follow-up now uses 1,000 whole-record graph-label
permutations. The rank-one resolvent localizes the known AL14--AL18 edge with
`p=0.012`, but a much simpler endpoint-contrast baseline performs better
(14/14 blocks, `p=0.010`). See `DS1_LOCALIZATION_AUDIT.md`. This is a no-go for
claiming incremental value from the resolvent graph-GP, and a conditional go
only for a transparently labeled single-case, FEM-free modal edge-contrast
proof of concept.

The cross-excitation gate is now complete: the frozen endpoint-contrast score
passes P1 and P2 vertical sweeps, with aggregate ranks 4 and 1 and permutation
`p=0.03497` and `p=0.01299`, respectively. Four spectral/pass-window variants
retain both positions within the top 5%. See `DS1_SWEEP_VALIDATION.md`. The
paper is therefore a defensible single-case full-scale proof of concept, while
remaining limited to one damage location.
