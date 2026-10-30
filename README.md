# Graph-Based Structural Damage Localization through Modal Relative Motion

Code, derived modal features, and reproduction scripts for the **SDHM R2 revision**.

**Authors:** Chung Truong Thanh and Thanh Bui Tien.  
**Corresponding author:** Chung Truong Thanh, chungtt@ntu.edu.vn.

The framework represents the Hell Bridge Test Arena (HBTA) sensor layout as a physical graph with 58 sensor channels and 122 candidate connections. The fused discrete score ranks the DS1 joint first. Continuous regional localization recovers all seven single-site states within one bay; the two-site state DS7 remains unresolved.

## Current version

- [Reproduction instructions and figure-to-script mapping](REPRODUCIBILITY_R2.md)
- [Analysis package](code_repository/hbta_physics_informed_frf/src/hbta_frf)
- [R2 figure scripts](reproduction/sdhm_R2/scripts)
- [Release file checksums](reproducibility_manifest_R2.json)

The analysis files and cached features were checked against the supplementary code archive submitted with R1. R2 adds the physical bridge background in Figure 6 and explanatory manuscript changes; the reported numerical results are unchanged. Script names ending in `_R1` are retained where their calculations or figures are unchanged in R2. Use the R2 script directory for this revision.

Manuscript sources, journal submission files, PDFs, and unused research materials are not distributed in this repository.

## Quick start

Python 3.10 or later is required. From the repository root:

```text
python -m pip install -r code_repository/hbta_physics_informed_frf/requirements.txt
python reproduction/sdhm_R2/scripts/make_fig_layout_R2.py
```

See [REPRODUCIBILITY_R2.md](REPRODUCIBILITY_R2.md) for the numerical pipelines, optional tests, other figures, and LaTeX compilation.

## Data availability

The raw benchmark measurements are available from [Zenodo](https://doi.org/10.5281/zenodo.10507956) and are not redistributed here. The code and derived modal features are publicly available in this repository. The raw HDF5 file is needed to repeat modal extraction and the measured-FRF panel of Figure 8; downstream localization can be reproduced from the included derived features.

The Figure 6 structural background was checked against the benchmark [sensor-layout drawing](https://doi.org/10.5281/zenodo.14028239). Sensor IDs, coordinates, and candidate connections remain unchanged from R1.
