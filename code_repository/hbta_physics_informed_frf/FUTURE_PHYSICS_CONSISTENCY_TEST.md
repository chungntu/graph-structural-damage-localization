# Future extension: modal physics-consistency localization

This note preserves a stricter physics-informed extension for a later paper.
The current DS1 paper remains intentionally simple and keeps the endpoint
contrast as its primary indicator.

## Core idea

For a candidate edge (e), use the measured healthy modal poles and mode
shapes to predict the first-order residual caused by a rank-one stiffness loss:

\[
g_{e,m}=w_e\sum_{n\ne m}
\frac{(b_e^\mathsf{T}\phi_n)(b_e^\mathsf{T}\phi_m)}{\lambda_n-\lambda_m}
\phi_n .
\]

Estimate a non-negative damage amplitude from the observed modal residual:

\[
\widehat{\gamma}_e=\arg\min_{\gamma\ge0}
\sum_m\left\|r_m-\gamma g_{e,m}\right\|_{\Sigma_m^{-1}}^2 .
\]

The candidate edge is judged by how well it explains the residual, rather than
only by how large an endpoint contrast happens to be.

## Stronger joint test

Add the first-order eigenvalue prediction

\[
\delta\lambda_{e,m}\approx-\gamma w_e
\left|b_e^\mathsf{T}\phi_m\right|^2
\]

and score each edge using a joint likelihood for modal-residue changes and
frequency shifts. Estimate \(\Sigma_m\) from healthy records only. The test
must be blind to the DS1 label until ranking is complete.

## Required assumptions and gates

- modal poles are sufficiently separated for first-order perturbation;
- phase and scale alignment of measured residues is fixed before testing;
- the graph edge weights are declared as reduced-order stiffness surrogates,
  not calibrated FEM stiffnesses;
- the observed sensors provide enough information for the candidate edge to be
  identifiable;
- the physics-consistency score must beat the endpoint baseline under a fixed
  cross-excitation and topology-permutation protocol.

If these gates fail, report the result as a mechanics-informed feature study,
not as a strict physics-informed localization method.

## Reminder

Revisit this extension only after the current DS1 manuscript is complete. It is
the preferred route for making the physics enter the estimator itself rather
than only motivating the feature choice.

## Exploratory result (2026-08-29)

A sparse family-level version was tested by fitting at most three non-negative
edge amplitudes.  In-sample residual explanation was misleadingly strong.  The
predeclared stricter check---fit on two measured modes and predict the third---
failed for every DS1--DS8 state.  The separate eigenvalue-shift signature also
failed its topology-permutation gate.  See `MULTI_EDGE_LOCALIZATION_TRIAL.md`,
`outputs/physics_consistency_multiedge.json`, and
`outputs/frequency_shift_region_experiment.json`.  The present three measured
vertical modes are therefore insufficient for this extension.
