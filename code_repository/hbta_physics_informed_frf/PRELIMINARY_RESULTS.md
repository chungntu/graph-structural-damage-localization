# Preliminary HBTA result and go/no-go decision

## Question tested

Does a three-mode physical prior improve complex-FRF modelling and one-class
damage detection over a frequency-only GP when only a few healthy HBTA blocks
are available?

This is deliberately one question. The study does not include topology,
localization, sensor failure, restoration, or a learned environmental model.

## First-principles model

For a linear structure, the acceleration FRF measured at sensor `s` is

```text
H_s(w) = sum_r [-w^2 R_sr / (w_r^2 - w^2 + 2 i zeta_r w_r w)] + discrepancy.
```

The modal frequencies `w_r` and damping ratios `zeta_r` are shared by all
sensors. Each sensor has its own complex residue `R_sr`. The grey-box model uses
the modal expression as its mean and a Matérn-3/2 GP for the remaining smooth
frequency discrepancy. The black-box model uses the same GP with zero physical
mean.

## Data and immutable-source rule

- Raw source: `D:\BIG DATA\Hell Bridge Test Arena HBTA\data_100Hz.h5`.
- The raw source is read in place and was not copied or modified.
- No original script from the raw-data folder or either previous paper was run.
- Derived FRFs and modal features are stored only under this project.
- Primary data: P2 vertical noise excitation, two UDS recordings and DS1--DS8.
- Each recording yields fourteen non-overlapping 60 s blocks after a 30 s edge
  trim. There are 140 blocks, 58 response sensors, and 297 bins over 1--30 Hz.

## Feasibility gate: passed

- Median shaker-response coherence is 0.977 and 0.970 in the two healthy
  recordings.
- The median usable-sensor fraction is 1.0; 90.2% of frequency bins satisfy the
  fixed coherence-coverage rule.
- Healthy log-magnitude correlation is 0.979.
- Median healthy difference is 0.885 dB and 4.54 degrees.
- Repeatable peaks include approximately 6.8, 12.2, and 24.1 Hz.

Therefore, failure later in the experiment is not caused by unusable FRFs.

## Physical-mean audit: passed, but not sufficient

The fitted vertical modes are 6.821, 17.245, and 24.141 Hz, with damping ratios
near 2%. The three-mode model has median weighted NMSE 0.0207 on its training
recording and 0.0876 on the second healthy recording. A direct empirical healthy
mean obtains 0.0793 on that recording. Thus the modal model explains most of the
healthy FRF, but is about 10.5% worse than the empirical mean out of sample. This
is acceptable for a prior mean, but does not itself establish a GP advantage.

## Primary baseline result: current claim is not supported

Within the first healthy recording, damage discrimination saturates: with four
or eight healthy training blocks, the modal-only, GP-only, and modal+GP models
all obtain macro DS1--DS8 AUROC 1.0 and zero false alarms on three held-out
same-session healthy blocks. There is no measurable room for a grey-box gain.

On the second healthy recording, all three models produce a 100% false-alarm
rate for every tested training size. When that recording is used as the healthy
negative class, macro damage AUROC at eight blocks is only 0.149 for modal-only,
0.151 for GP-only, and 0.159 for modal+GP. The small numerical grey-box increase
is scientifically irrelevant because all methods fail the false-alarm test.

The second UDS recording is a genuine recording-domain shift: it was acquired
five days earlier, at 10 C rather than 13 C, and is marked as containing a train
passage. Most damage records are closer to the first UDS recording in date and
conditions. Treating the two UDS recordings as exchangeable would therefore be
incorrect.

## Physics diagnostics

Two physically motivated reductions were tested without adding a GP:

1. Shared modal poles with sensor residues fitted freely for every block.
2. Modal residue shapes compared through complex MAC, which removes an unknown
   complex input-participation scalar.

Both separate damage perfectly inside the first recording domain. Neither
controls the second-recording false alarms at the calibration threshold. The
mode-shape representation does preserve useful ranking: using one healthy block,
its macro AUROC against the external healthy recording is 0.952, but this falls
to 0.722 with eight blocks as the reference distribution becomes narrowly tied
to the first session. Ranking without false-alarm control is not enough for an
operational SHM claim.

## Go/no-go decision

**No-go for the original paper claim on HBTA alone.**

The evidence currently supports three statements:

1. The physical modal mean is valid and compact.
2. The within-session HBTA task is too easy to demonstrate the value of a GP
   prior.
3. The cross-session HBTA task is dominated by unmodelled environmental or
   acquisition shift, and only two comparable healthy noise sessions are
   available. A GP cannot learn a defensible environmental distribution from
   two sessions.

Proceeding directly to a paper would likely produce either a trivial result or
an over-fitted nuisance correction. That would violate the intended principle
of a simple and genuinely physics-informed paper.

## Recommended next experiment before committing a paper

Retain the same three-mode equation, but test it on a dataset with many healthy
sessions spanning environmental conditions. The single hypothesis becomes:

> A modal-physics GP requires fewer healthy environmental states than a
> black-box GP to control false alarms while retaining sensitivity to stiffness
> loss.

Use HBTA only as a strict external stress test, not as the sole source for
learning environmental variation. The experiment should have three models only
(modal, GP, modal+GP), a fixed recording-level split, a synthetic MDOF check,
and data-scarcity curves by number of healthy sessions rather than number of
blocks from one session.

The next go gate is explicit: continue only if modal+GP reduces held-out healthy
negative log predictive likelihood and false alarms, and improves or preserves
damage AUROC, over both baselines for at least two low-data regimes. Otherwise
stop this paper direction.
