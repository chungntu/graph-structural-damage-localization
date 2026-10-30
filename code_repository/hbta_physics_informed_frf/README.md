# HBTA graph damage localization — SDHM R2

Start with the [R2 reproduction guide](../../REPRODUCIBILITY_R2.md) for installation, derived-feature pipelines, tests, and figure generation.

- `src/hbta_frf/`: analysis implementation.
- `cache/`: derived modal features and cached spectral estimates; raw benchmark HDF5 is external.
- `topology/hbta_physical_edge_list.csv`: 58-channel, 122-connection physical graph.
- `outputs/`: included reference results; the release manifest identifies the outputs used by R2.
- `tests/`: numerical unit tests.

The analysis and reference data used by R2 match the submitted R1 supplementary package. R2 updates Figure 6 and the manuscript explanations without changing the reported benchmark results. Older research notes and experimental modules are retained for history; follow the R2 guide for the reported configuration.
