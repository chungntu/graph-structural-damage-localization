# Future extension: multi-edge and region localization

This note preserves a possible follow-up paper beyond the current DS1-only
single-edge study.

## Why this is needed

The current model tests one candidate graph edge at a time. That is appropriate
for DS1 because the documented intervention can be mapped to the drawing-derived
AL14--AL18 path. DS2--DS8 do not have the same one-to-one mapping: some affect
multiple connections, some remove a member spanning several sensor paths, and
some damage joints that are not directly instrumented. Their inferred dominant
edges are therefore hypotheses, not validated localization labels.

## Proposed physical model

Represent a damage set or region \(\mathcal S\) by a sum of local stiffness
reductions:

\[
\Delta K=-\sum_{e\in\mathcal S}\rho_e k_e b_e b_e^{\mathsf T}.
\]

The single-edge score can then be extended to score a set, member, or region.
Possible outputs are:

- a ranked multi-edge set;
- a member or subsystem score obtained by aggregating its candidate edges;
- a region posterior or confidence set rather than one exact edge.

## Required new ground truth

Before running a confirmatory study, define a documented mapping from every
damage state to one of the following:

1. a unique candidate edge;
2. a validated set of candidate edges;
3. a member or region label with an explicit aggregation rule.

The mapping must be fixed before inspecting localization scores. If the damaged
joint is uninstrumented, the claim should be limited to the nearest represented
member/path or region and should not be called exact edge localization.

## Candidate follow-up paper

**Working title:** *FEM-Free Multi-Edge Localization of Distributed Connection
Losses from Modal-Residue Graph Contrasts.*

**Question:** Can a set-valued graph perturbation model localize distributed or
member-level damage more faithfully than a single-edge model when the damage
truth is represented as a known edge set or region?

Use DS2--DS8 only after the physical mapping is established. Keep DS1 as a
single-edge sanity check, and evaluate region/member recovery rather than forcing
all states into an exact-edge rank.

## Reminder

This extension is intentionally deferred. The current paper should remain the
simple DS1 demonstration with one known edge and no additional model complexity.

## Exploratory result (2026-08-29)

The proposed subsystem aggregation was implemented and tested on P2 random
input and on all P1/P2 vertical swept-sine records.  It consistently recovered
the documented subsystem for DS1--DS2 and DS5--DS7, but not for DS3--DS4 or
DS8.  See `MULTI_EDGE_LOCALIZATION_TRIAL.md` and
`outputs/all_damage_sweep_region.json`.  This partial result is not sufficient
for an all-state localization claim.  Exact multi-edge validation remains
blocked by the absence of a drawing-locked sensor-edge set for DS2--DS8.

A subsequent drawing-digitized spatial-region scan confirmed that DS8 is close
in physical space even when its winning structural family is incorrect. A
P1/P2 multi-reference estimate recovered DS1, DS2, DS4, DS6, and DS8 within one
panel, but not DS3, DS5, or DS7. Dedicated lateral excitation did not recover
DS5/DS7 and had weak modal-fit quality. See
`outputs/spatial_region_localization.json` and
`outputs/multireference_spatial_localization.json` and
`outputs/lateral_sweep_localization.json`.
