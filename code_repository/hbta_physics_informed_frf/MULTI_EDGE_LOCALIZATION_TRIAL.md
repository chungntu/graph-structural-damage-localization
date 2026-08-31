# DS1--DS8 multi-edge and physics-consistency trial

## Question

Can the current three-mode, sensor-graph formulation be extended from the
single represented DS1 edge to the full HBTA DS1--DS8 set without a calibrated
finite-element model?

## Ground-truth level used for the exploratory gate

The public HBTA description does not name an exact sensor-graph edge set for
DS2--DS8.  The most defensible common target currently available is therefore
the documented structural subsystem:

| Damage states | Documented intervention | Sensor-graph family tested |
|---|---|---|
| DS1--DS2 | stringer-to-floor-beam connection(s) | deck longitudinal |
| DS3--DS4 | stringer cross beam(s) | deck transverse |
| DS5--DS7 | lateral-bracing connection(s) | lateral bracing |
| DS8 | floor-beam-to-main-member connection | deck--truss interface |

This is subsystem localization, not exact localization of an uninstrumented
joint.  Six graph families were kept as candidates, including the truss chord
and diagonal families, so the test was not restricted to the known damage type.

## Experiments

### 1. Edge-score aggregation on P2 random input

Single-edge scores were aggregated within each structural family by the mean
of its largest 20 percent of scores.  The endpoint score identified the
documented family for five of eight states (DS1, DS2, and DS5--DS7).  The
graph-propagated score identified four of eight because DS1 was displaced by
the lateral-bracing family.  DS3, DS4, and DS8 were not recovered.

### 2. Cross-excitation swept-sine test

All 40 vertical swept-sine passages were read from the raw HDF5 file in place:
two shaker positions, four healthy passages per position, and two passages for
each of DS1--DS8 per position.  All optimizations converged.  Median coherence
was 0.996, the minimum coherent-point fraction was 0.970, and median weighted
NMSE was 0.029.  DS2 at P1 had the largest NMSE (0.106) and placed the second
mode at the upper search bound, so it requires special caution.

For the endpoint score, five of eight states were assigned to the documented
family at both P1 and P2.  The mean reciprocal rank over the 16 state-position
tests was 0.726.  A joint 1000-permutation edge-family-label control gave
one-sided p=0.009 for mean reciprocal rank and p=0.002 for the fraction correct
at both positions.  The signal is therefore topology-specific but incomplete:
DS3, DS4, and DS8 repeatedly favored the lateral-bracing family.

The graph-propagated score was weaker: four of eight states were correct at
both positions, with cross-position mean reciprocal rank 0.690.

### 3. First-order multi-edge physics consistency

For each candidate family, up to three non-negative edge amplitudes were fitted
using first-order eigenvector-perturbation directions constructed from the
three healthy measured modes.  An in-sample fit appeared strong but was not
diagnostic because several multi-edge subspaces could explain the same
three-mode residual.  Under the stricter gate---fit amplitudes on two modes and
predict the held-out third mode---none of the eight states ranked its documented
family first.  The cross-mode mean reciprocal rank was 0.317.  The strict
physics-consistency gate therefore failed.

### 4. Eigenvalue-shift consistency

The first-order signature
`delta lambda proportional to -w_e |b_e^T phi_m|^2` recovered only one of eight
states with the primary aggregation rule.  Its state-level mean reciprocal rank
was 0.448 and was not better than permuted family labels (p=0.370).

## Decision

The current formulation cannot honestly claim localization of all DS1--DS8.
It contains reproducible subsystem information for DS1--DS2 and DS5--DS7, but
the three retained vertical modes do not identify DS3--DS4 or DS8.  Adding the
failed extensions to the main manuscript as if they were a successful
multi-damage method would weaken the paper.

The next defensible experiment is not a larger multi-edge fit on the same three
modes.  It requires (i) a drawing-locked spatial region or edge-set truth for
every damage state and (ii) a richer observation model, most plausibly a
full-frequency edge field or a larger tracked modal set.  DS8 is especially
important because its fitted poles reach the current modal search bounds and
the benchmark literature reports modal reorganization for this global damage
case.

## Spatial-region follow-up (2026-08-29)

The family-level result was re-audited because a response may move onto a
neighboring member while remaining near the actual damage. Red intervention
marks were digitized from the published HBTA schematic and projected onto the
longitudinal bridge axis. The deck projection was calibrated by the independently
known DS1 coordinate at x=-3.5 m. The resulting approximate region centers were
DS3=-8.62 m, DS4=-5.58 m, DS5=-7.44 m, DS6=-1.56 m, and DS8=2.21 m; DS7 uses
the two DS5/DS6 regions.

A Gaussian region scan pooled all edge types within one 3.5-m panel. Separate
P1/P2 vertical estimates localized DS1, DS2, and DS8 within one panel at both
positions. DS4--DS6 were often close under P2 but shifted toward approximately
x=+4.5 m under P1. The primary endpoint scan achieved mean absolute error
4.36 m and was only marginal against coordinate permutations (p=0.060).

Enforcing input-location invariance by estimating one multi-reference mode shape
from P1 and P2 improved the graph scan to five of eight correct states (DS1,
DS2, DS4, DS6, and DS8), but DS3, DS5, and DS7 remained outside the one-panel
gate. Its mean absolute longitudinal error was 3.64 m. Against 1000 coordinate-
label permutations, the result was close to, but did not cross, the conventional
5 percent threshold (one-sided p=0.053 for mean error and p=0.058 for the
fraction of states within one panel). It is therefore a useful lead rather than
a validated all-damage localization result. Dedicated Y-direction swept-sine
modes were then tested for DS5--DS7.
Only DS6 passed; DS5 and the DS5 component of DS7 remained displaced. Lateral
modal fits were also weak (median NMSE 0.224, maximum 0.445), and the spatial
result did not beat coordinate permutations (p=0.229 for graph propagation).

This follow-up refines rather than reverses the decision: DS8 has genuine
near-location evidence that was hidden by structural-family aggregation, but
the available modal representation still does not support all-state spatial
localization.

## Reproducible artifacts

- `outputs/multiedge_region_experiment.json`: P2 random-input family test.
- `outputs/all_damage_sweep_region.json`: P1/P2 cross-excitation family test.
- `outputs/physics_consistency_multiedge.json`: held-out-mode mechanics test.
- `outputs/frequency_shift_region_experiment.json`: eigenvalue-shift test.
- `outputs/spatial_region_localization.json`: drawing-locked longitudinal scan.
- `outputs/multireference_spatial_localization.json`: P1/P2-fused longitudinal
  scan and coordinate-label permutation control.
- `outputs/spatial_region_sensor_graph.png`: sensor-graph overlay of drawing-
  locked and inferred DS1--DS8 longitudinal locations.
- `outputs/lateral_sweep_localization.json`: Y-direction DS5--DS7 audit.
- `cache/all_damage_sweep_modal_features.npz`: derived 40-passage cache; raw
  HDF5 data remain at their original location.
- `cache/all_damage_lateral_sweep_modal_features.npz`: derived lateral cache;
  raw HDF5 data remain at their original location.
