# Introduction
Many years ago Silvia Venegas and I wrote a manual to assist students in our department (and elsewhere) in 
doing EOF analysis and SVD for combined fields (see docs). We used matlab for the calculations and gave the recipes. 
As a result this was used widely and is still being cited 30 years on. After playing around with ipython and jupyter notebooks
I realised that if I was doing this again python would be the way. There are many packages that can do this now a days, in
my experience what matters is being able to do the analysis and visualize it at the same time. This is what interactive data
analysis is about. The translation is done using gemini 3.8 flash. It sometimes adds in terms (like names of eof fields) that 
may be strange. 

# Empirical Orthogonal Function (EOF) Analysis of Sea Surface Temperatures

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A repository of Python programs and interactive Jupyter notebooks implementing **Empirical Orthogonal Function (EOF)** and **Principal Component Analysis (PCA)** on global and regional sea surface temperature anomalies.

The mathematical formulations and computational recipes in this repository are based on:
> **Björnsson, H. and Venegas, S. A. (1997):**  
> *A Manual for EOF and SVD analyses of Climatic Data.*  
> Department of Atmospheric and Oceanic Sciences and Centre for Climate and Global Change Research (C²GCR), McGill University.  
> Report No. 97-1. (Available in [`docs/eof.pdf`](docs/eof.pdf)).

---

## Repository Structure

```text
EOFexamples/
├── data/
│   ├── ERSST_AnoAnn.nc              # Global annual SST anomalies (1880–2020)
│   └── ERSST_AtlanticAnoAnn.nc      # North Atlantic annual SST anomalies (10°N–82°N, 1880–2020)
├── docs/
│   └── eof.pdf                      # "A Manual for EOF and SVD analyses of Climatic Data"
├── figures/                         # Output figures for global EOF analysis
├── figures_atlantic/                # Output figures for North Atlantic EOF analysis (1880–2020)
├── figures_atlantic_detrended/      # Output figures for detrended North Atlantic EOF analysis (1910–2020)
├── notebooks/
│   ├── EOF_Analysis_ERSST.ipynb             # Global EOF analysis tutorial
│   └── EOF_Analysis_ERSST_Atlantic.ipynb    # North Atlantic regional EOF analysis tutorial
├── src/
│   ├── eof_analysis_ersst.py                # Standalone CLI for global SST anomalies
│   ├── eof_analysis_atlantic.py             # Standalone CLI for North Atlantic anomalies
│   └── eof_analysis_atlantic_detrended.py   # Standalone CLI for detrended North Atlantic (post-1910)
├── environment.yml                  # Conda environment definition
├── requirements.txt                 # Pip requirements file
├── .gitignore
└── README.md
```

---

## Key Methodologies Implemented

1. **$S$-Mode Matrix Decomposition (§1.2, §2.1):**  
   Data are formatted into an $n \times p$ matrix ($n$ observation times in rows, $p$ active ocean grid points in columns) and centered by removing column-wise temporal means (`detrend(M, 0)`).
2. **Spherical Area Weighting:**  
   Grid cells are weighted by $\sqrt{\cos(\phi)}$ to correct for meridian convergence toward the poles.
3. **Direct SVD & Dual Covariance Equivalence (§2.3.1, §4.2.1):**  
   Decomposition is computed via direct Singular Value Decomposition ($F_w = U \Sigma V^T$) and verified against the dual temporal covariance method ($L = F_w F_w^T$ for $p \gg n$), avoiding large $p \times p$ spatial covariance memory overhead.
4. **North's Rule of Thumb (§4.3.2):**  
   Eigenvalue sampling error bounds are evaluated as $\Delta \lambda_i \approx \lambda_i \sqrt{2/n}$ to assess mode degeneracy and statistical separability.
5. **Physical Pattern Reconstruction (§4.1):**  
   Anomalies are projected in physical units ($^\circ\mathrm{C}$ per $1\sigma$ excursion of the standardized PC time series) alongside 11-year running means.
6. **Homogeneous Correlation & Local Variance Maps (§4.1.1, §4.1.2):**  
   Mapping correlation coefficients ($r$) and local variance explained ($r^2 \times 100\%$) across the ocean to reveal centers of action.
7. **Rotated EOFs via Kaiser's Varimax (§2.4, Appendix A):**  
   Direct translation of the MATLAB `VARIMAX` routine from Appendix A to isolate compact, regional modes of climate variability.

---

## Datasets

The analyses utilize annual Sea Surface Temperature anomalies from NOAA Extended Reconstructed Sea Surface Temperature version 5 (**ERSSTv5**):
- **`data/ERSST_AnoAnn.nc`:** Global ocean domain (89 latitudes $\times$ 180 longitudes, $2^\circ \times 2^\circ$ resolution, 141 years from 1880 to 2020).
- **`data/ERSST_AtlanticAnoAnn.nc`:** North Atlantic basin ($10^\circ\mathrm{N} - 82^\circ\mathrm{N}$, $68^\circ\mathrm{W} - 14^\circ\mathrm{E}$, 37 latitudes $\times$ 42 longitudes, 1880–2020).

---

## Installation & Environment Setup

### Using Conda (Recommended)

```bash
git clone https://github.com/gagnagaman/EOFexamples.git
cd EOFexamples
conda env create -f environment.yml
conda activate eof-analysis
```

### Using Pip

```bash
git clone https://github.com/gagnagaman/EOFexamples.git
cd EOFexamples
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive Notebooks

Launch Jupyter and open either notebook in the [`notebooks/`](notebooks/) directory:

```bash
jupyter notebook notebooks/
```

- **[`notebooks/EOF_Analysis_ERSST.ipynb`](notebooks/EOF_Analysis_ERSST.ipynb):** Global SST EOF decomposition, identifying the global warming trend (Mode 1, 36.4%), Pacific Decadal / ENSO mode (Mode 2, 13.7%), and multi-basin variability.
- **[`notebooks/EOF_Analysis_ERSST_Atlantic.ipynb`](notebooks/EOF_Analysis_ERSST_Atlantic.ipynb):** North Atlantic domain decomposition, identifying the Atlantic Multidecadal Variability / AMV (Mode 1, 45.5%) and the North Atlantic SST Tripole (Mode 2, 13.4%).

### 2. Standalone CLI Programs

The scripts in [`src/`](src/) can be executed directly from either the root directory or `src/`:

```bash
# Global EOF analysis (exports to figures/):
python src/eof_analysis_ersst.py

# North Atlantic EOF analysis (exports to figures_atlantic/):
python src/eof_analysis_atlantic.py

# Detrended North Atlantic EOF analysis post-1910 (exports to figures_atlantic_detrended/):
python src/eof_analysis_atlantic_detrended.py --start_year 1910
```

---

## Key Results Summary

### Global Ocean (ERSSTv5, 1880–2020)
- **EOF 1 (36.4%):** Global ocean warming trend with accelerated upward trajectory post-1970.
- **EOF 2 (13.7%):** Pacific Decadal Oscillation (PDO) / ENSO-like decadal pattern.
- **EOF 3 & 4 (~4% each):** Atlantic Multidecadal Variability and Southern Ocean modes.

### North Atlantic Basin (ERSSTv5, 10°N–82°N, 1880–2020)
- **EOF 1 (45.5%):** Basin-wide warming and multidecadal variability (AMV/AMO), characterized by warm periods in the 1930s–1950s, a cold period in the 1970s–1980s, and strong warming from the mid-1990s onward.
- **EOF 2 (13.4%):** North Atlantic SST tripole pattern, featuring opposing anomalies between the subpolar gyre and the subtropical/eastern Atlantic, characteristic of oceanic response to the North Atlantic Oscillation (NAO).
- **EOF 3 (9.5%):** Meridional gradient mode between the subpolar gyre and lower latitudes.
- **EOF 4 (7.1%):** High-latitude mode focused on the Nordic Seas and Arctic inflow regions.
- **Separation:** According to North's rule of thumb, Modes 1, 2, 3, and 4 are all statistically well-separated with non-overlapping error bars. Together, the top 4 modes account for **75.5%** of the total variance in the basin.

---

## References

- Björnsson, H. and S. A. Venegas, 1997: *A Manual for EOF and SVD analyses of Climatic Data.* CCGCR Report No. 97-1, McGill University, 52 pp.
- North, G. R., T. L. Bell, R. F. Cahalan, and F. J. Moeng, 1982: Sampling errors in the estimation of empirical orthogonal functions. *Mon. Wea. Rev.*, 110, 699–706.
- Huang, B., et al., 2017: Extended Reconstructed Sea Surface Temperature version 5 (ERSSTv5), Upgrades, validations, and intercomparisons. *J. Climate*, 30, 8179–8205.
