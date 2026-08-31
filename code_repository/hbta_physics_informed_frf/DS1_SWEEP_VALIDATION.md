# DS1 cross-excitation validation

## Purpose

The P2-noise experiment showed that the known DS1 connection AL1ranked first by a modal edge-contrast indicator. The decisive question was
4--AL18 was
whether that result was merely an artifact of one excitation configuration.
This audit applies the formula unchanged to two vertical swept-sine
configurations already present in HBTA:

- P1 vertical sweep;
- P2 vertical sweep.

No new data were copied, no FEM was introduced, and no parameter was tuned on
DS1 sweep responses.

## Frozen protocol

- Development data: P2-noise modal features and selected pole-fit sensors from
  the preceding audit.
- Healthy template for each position: both healthy sweep records, with two
  passes per record.
- Test data: the two DS1 sweep passes at the same position.
- Sweep passes: 45--285 s (up-sweep) and 380--615 s (down-sweep).
- H1-FRF: 2,048-point Welch segments, 75% overlap, 5--26 Hz modal fitting
  band.
- Candidate set: all 122 physical edges.
- Ground truth: AL14--AL18, used only after blind scoring.
- Healthy control: leave-one-recording-out residuals, holding out both passes
  of each healthy record.
- Primary null: 1,000 whole-record sensor-label permutations within AG/AL
  families, preserving the graph structure and edge weights.

The sweep windows were chosen from the measured shaker frequency ridge and
modal-fit quality before inspecting localization ranks.

## FRF/modal quality

- Median coherence over all sweep passes: **0.9961**.
- Minimum fraction of coherent sensor-frequency bins: **0.9759**.
- Weighted modal-fit NMSE: median **0.0239**, range **0.0074--0.0476**.
- All 12 modal optimizations converged.
- Fitted poles remain close to the three development modes near 6.8, 17.1 and
  24.1 Hz.

The sweep measurements are therefore sufficiently coherent for a localization
test; the gate is not being decided by a failed FRF extraction.

## Primary results

| Configuration | DS1 pass ranks for AL14--AL18 | Aggregate rank | Healthy LORO aggregate rank | Target-edge AUROC | Permutation p |
|---|---:|---:|---:|---:|---:|
| P1 vertical sweep | 4, 3 | **4** | 30 | **1.000** | **0.03497** |
| P2 vertical sweep | 1, 1 | **1** | 2 | **1.000** | **0.01299** |

Both configurations satisfy the locked position-level criteria:

- aggregate rank within the top 5% of 122 edges (top seven);
- both sweep passes within the top 5%;
- topology-label permutation (pleq0.05);
- every DS1 target-edge score exceeds every leave-one-recording-out healthy
  target-edge score.

The edge is more sensitive in the P2 configuration: it is exactly top-1 in both
passes. P1 still identifies the correct connection in the top four, despite a
different excitation position.

## Robustness

Four prespecified extraction variants were evaluated without changing the
indicator:

| Variant | P1 aggregate rank | P2 aggregate rank | Target-edge AUROC |
|---|---:|---:|---:|
| 1,024-point spectral window | 4 | 1 | 1.000 |
| 4,096-point spectral window | 3 | 1 | 1.000 |
| Pass windows moved inward by 5 s | 4 | 1 | 1.000 |
| Pass windows moved outward by 5 s | 4 | 1 | 1.000 |

The result therefore does not depend on one Welch resolution or an exact
start/stop choice.

## Gate decision

The combined gate is **GO**:

- P1 and P2 both pass the locked localization test;
- the P2-noise, P1-sweep and P2-sweep permutation tests are all significant;
- every robustness variant retains both positions in the top 5%;
- modal fits are successful and coherent.

This means the P2-noise result is not adequately explained as an artifact of a
single excitation position or a single spectral-window choice.

## What this does and does not prove

It supports the narrow statement:

> A FEM-free modal edge-contrast indicator can localize the known DS1 connection
> AL14--AL18 across P1 and P2 vertical excitation configurations in the HBTA
> full-scale experiment.

It does **not** prove localization at unseen damage locations. P1 and P2 are
independent excitation configurations for the same damage case, not independent
damage sites. It also does not restore the failed claim that the rank-one
resolvent graph-GP provides incremental value; the simple endpoint contrast is
the indicator validated here.

The healthy P2 sweep still ranks AL14--AL18 second, showing that this connection
is intrinsically sensitive to the recording domain. The evidence is therefore
strongest for topology-specific localization after an alarm, not for a
generalized healthy/damaged detector.

## Manuscript decision

The paper is now reasonable as a **single-case full-scale proof of concept** if
it is explicitly framed as a mechanics-informed, FEM-free edge-contrast study.
It should not claim a general physics-informed GP or a general damage localizer.
The remaining limitation is scientific scope, not a failed cross-excitation
gate.

## Artifacts

- Sweep extraction and validation: `src/hbta_frf/run_ds1_sweep_validation.py`.
- Derived sweep cache: `cache/ds1_sweep_modal_features.npz`.
- Machine-readable result: `outputs/ds1_sweep_validation.json`.
- Figure: `outputs/ds1_sweep_validation.png`.
- Raw HBTA file remained read-only and unchanged.

