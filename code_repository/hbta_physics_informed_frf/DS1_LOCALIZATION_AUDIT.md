# DS1 single-case localization audit

## Question

Can the known DS1 damaged connection, AL14--AL18, be identified blindly among
all 122 physical graph edges using only healthy-fitted modal information and no
finite-element model?

This audit separates two claims:

1. the rank-one resolvent field contains DS1 localization information;
2. the resolvent/graph-GP machinery adds information beyond a simple local edge
   contrast.

## Locked protocol

- Fit the healthy modal template and tune graph hyperparameters using only the
  fourteen UDS01 blocks.
- Keep all fourteen UDS02 blocks as an external healthy negative control.
- Keep all fourteen DS1 blocks unseen until scoring.
- Search blindly over all 122 candidate edges. The known AL14--AL18 label is
  used only to compute its post-scoring rank.
- Treat the fourteen blocks as repeated estimates from one record, not as
  fourteen independent damage experiments.
- Use one record-level mean reciprocal rank as the primary statistic.
- Compare the complete DS1 record with 1,000 sensor-label permutations within
  the AG and AL families. Every permutation preserves topology, relation rows,
  edge weights, degree sequence, and the full temporal record.

DS1 was selected after the broader DS1--DS8 exploration because it is the only
state with a unique ground-truth edge represented in the graph. Consequently,
this is a locked follow-up audit, not a preregistered confirmatory experiment.

## Physics and simple baseline

For candidate edge `e`, let `b_e` be its signed incidence vector and

```text
R = (L + alpha I)^(-1).
```

The rank-one stiffness-loss model predicts the spatial damage direction

```text
g_e = R b_e.
```

Its scale-invariant score is the graph-GP-whitened alignment between the modal
residual and `g_e`, averaged over the three fitted modes.

The simple baseline replaces `g_e` by `b_e` and removes graph-GP whitening. It
therefore measures only the normalized response contrast between the two edge
endpoints. This is not entirely physics-free: it still uses the physical edge
set and a local compatibility/deformation assumption. It does not use the
resolvent propagation or the probabilistic graph model.

## Results

| Quantity | Rank-one resolvent | Endpoint contrast |
|---|---:|---:|
| Aggregate rank of AL14--AL18 among 122 edges | **1** | **1** |
| Blocks ranking AL14--AL18 first | 11/14 | **14/14** |
| Record mean reciprocal rank | 0.893 | **1.000** |
| Graph-label permutation p-value | 0.0120 | **0.0100** |
| AL14--AL18 score AUROC, DS1 versus UDS02 | 1.000 | 1.000 |
| Aggregate AL14--AL18 rank in healthy UDS02 | 6 | 2 |
| Leave-one-DS1-block-out aggregate rank | 1 in 14/14 runs | 1 in 14/14 runs |

For the rank-one model, every modal subset places AL14--AL18 within the top
seven edges. The second and third modes carry most of the top-1 localization
signal. Aggregating consecutive decisions from 60 s to 120, 420, and 840 s
keeps AL14--AL18 at aggregate rank one.

For the endpoint contrast, AL14--AL18 remains aggregate rank one for all seven
nonempty subsets of the three modes. Its DS1-versus-UDS02 target-edge AUROC is
0.995 or higher for every subset. Thus the result is not caused by one selected
mode.

## Interpretation

The DS1 record contains a strong, topology-specific local signature at the
known damaged connection. Both graph-label permutation tests pass, so the
correct mapping is very unlikely to arise from an arbitrary relabeling of the
same graph.

However, the rank-one resolvent is not the supported source of improvement. A
strictly simpler endpoint-contrast score is stronger on DS1, while both obtain
perfect separation of the target-edge score from external healthy UDS02. The
rank-one graph-GP therefore fails its incremental-value gate.

The healthy negative control also warns against overinterpretation. Even
without DS1, AL14--AL18 ranks sixth under the resolvent and second under endpoint
contrast, although it never becomes the aggregate top edge and its absolute
score remains below every DS1 block. The location is intrinsically responsive
to recording-domain change; damage increases an already sensitive contrast.

## Cross-excitation validation

The follow-up sweep audit is reported in `DS1_SWEEP_VALIDATION.md`. The score
was frozen from the P2-noise development configuration and then applied to P1
and P2 vertical sine sweeps. AL14--AL18 ranked 4 and 3 in the two P1 passes and
1 and 1 in the two P2 passes. The target-edge score completely separated DS1
from leave-one-recording-out healthy sweeps in both positions; graph-label
permutation p-values were 0.03497 (P1) and 0.01299 (P2). Four spectral-window
and pass-window perturbations retained both positions within the top 5%.

Thus the P2-noise result is not adequately explained by one excitation
configuration. It remains a same-damage, cross-excitation validation rather
than validation across damage locations.

## Decision

**No-go for the claim that the multi-scale rank-one resolvent graph-GP is needed
or that it is the mechanism producing DS1 localization.**

**Go for a narrow single-case statement:** a FEM-free, modal edge-contrast
indicator identifies AL14--AL18 blindly across P1 and P2 excitation
configurations in the DS1 full-scale record, survives 1,000 topology-label
controls in each configuration, and separates DS1 from leave-one-recording-out
healthy sweeps at that edge.

This supports a simple local structural prior, not a general-purpose damage
localizer. The result does not establish generalization to other damage
locations, and it does not support incremental value from the rank-one
resolvent graph-GP. A paper based only on HBTA must be framed explicitly as a
full-scale single-case proof of concept, not as general validation of
physics-informed graph-GP localization.

## Artifacts

- Implementation: `src/hbta_frf/run_ds1_localization_audit.py`
- Core scores: `src/hbta_frf/graph_damage_gp.py`
- Machine-readable result: `outputs/ds1_localization_audit.json`
- Figure: `outputs/ds1_localization_audit.png`
- Unit tests: five passed
