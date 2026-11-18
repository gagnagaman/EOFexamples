#!/usr/bin/env python3
"""
EOF Analysis of Detrended North Atlantic SST Anomalies (Post-1910)
------------------------------------------------------------------
Based on the methodologies presented in:
"A Manual for EOF and SVD analyses of Climatic Data"
H. Björnsson and S. A. Venegas
Department of Atmospheric and Oceanic Sciences & C2GCR, McGill University

This standalone script:
1. Loads subpolar North Atlantic annual SST anomalies from ERSST_AtlanticAnoAnn.nc
   (10°N - 82°N, 68°W - 14°E).
2. Filters the time series to years after 1910 (1910–2020 by default).
3. Fits and removes the linear trend at each ocean gridpoint (scipy.signal.detrend).
4. Plots and exports the spatial map of the removed linear trend (°C / decade).
5. Prepares the weighted residual data matrix F_w using cos(lat) area weighting.
6. Decomposes the residuals into EOFs via SVD (F_w = U Sigma V^T) and validates
   against the Dual Covariance method (FF^T for p >> n).
7. Evaluates the variance spectrum and North et al. (1982) sampling error bounds.
8. Reconstructs spatial regression patterns (°C per 1-sigma PC) and standardized
   PC time series (1910–2020).
9. Computes homogeneous correlation maps (r) and local variance explained maps (r^2).
10. Computes Rotated EOFs using Kaiser's Varimax rotation (Appendix A).
11. Saves all figures into 'figures_atlantic_detrended/'.
"""

import os
import argparse
import numpy as np
import scipy.signal
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings("ignore")


def running_mean(x, n=11):
    """Computes a centered moving average of length n."""
    return np.convolve(x, np.ones(n) / n, mode="same")


def vector_to_map(vec, mask, n_lat, n_lon):
    """Reshapes a 1D vector of ocean points back to a 2D (lat, lon) array."""
    grid = np.full((n_lat, n_lon), np.nan)
    grid[mask] = vec
    return grid


def varimax(phi, gamma=1.0, max_iter=500, tol=1e-6):
    """
    Performs Varimax rotation on matrix phi (truncated eigenvector basis).
    Translated from the MATLAB M-script VARIMAX in Appendix A of Björnsson & Venegas.
    """
    p, k = phi.shape
    r = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        lam = np.dot(phi, r)
        u, s, vh = np.linalg.svd(
            np.dot(
                phi.T,
                np.asarray(lam) ** 3
                - (gamma / p) * np.dot(lam, np.diag(np.diag(np.dot(lam.T, lam)))),
            )
        )
        r = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1.0 + tol:
            break
    return np.dot(phi, r), r


def setup_atlantic_map(ax, lons, lats):
    """Configures regional North Atlantic map bounds, features, and gridlines."""
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", color="#222222", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="#d0d0d0", zorder=1)
    gl = ax.gridlines(
        draw_labels=True, linestyle=":", color="gray", alpha=0.5,
        xlocs=np.arange(-60, 20, 20), ylocs=np.arange(20, 90, 20)
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


def resolve_path(filepath, search_dirs):
    """Finds filepath directly or within candidate search directories."""
    if os.path.exists(filepath):
        return os.path.abspath(filepath)
    for sdir in search_dirs:
        candidate = os.path.join(sdir, filepath)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return filepath


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_data_dir = os.path.join(project_root, "data")
    default_fig_dir = os.path.join(project_root, "figures_atlantic_detrended")

    parser = argparse.ArgumentParser(
        description="EOF Analysis of Linearly Detrended North Atlantic ERSST Data (Post-1910)"
    )
    parser.add_argument(
        "--ncfile",
        type=str,
        default=os.path.join(default_data_dir, "ERSST_AtlanticAnoAnn.nc"),
        help="Path to the Atlantic NetCDF file (default: data/ERSST_AtlanticAnoAnn.nc)",
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=1910,
        help="Start year for analysis (default: 1910)",
    )
    parser.add_argument(
        "--figdir",
        type=str,
        default=default_fig_dir,
        help="Directory to save generated figures (default: figures_atlantic_detrended/)",
    )
    parser.add_argument(
        "--n_modes",
        type=int,
        default=4,
        help="Number of leading EOF modes to plot in detail (default: 4)",
    )
    args = parser.parse_args()

    nc_path = resolve_path(args.ncfile, [default_data_dir, project_root, "."])
    fig_dir = args.figdir if os.path.isabs(args.figdir) else os.path.join(project_root, args.figdir)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 78)
    print(" EOF Analysis of Linearly Detrended North Atlantic SST Anomalies")
    print(f" Subset Period: {args.start_year} - 2020 | Removing Linear Trend at Each Grid Point")
    print(" Based on: Björnsson & Venegas, 'A Manual for EOF and SVD analyses...'")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. Load Data & Filter Period
    # -------------------------------------------------------------------------
    print(f"\n[1/7] Ingesting dataset: {nc_path}")
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"Dataset file '{nc_path}' not found.")

    ds = xr.open_dataset(nc_path)
    lats = ds["lat"].values
    lons = ds["lon"].values
    time_coords = ds["time"].values
    years_all = np.array([t.year for t in time_coords])
    n_lat = len(lats)
    n_lon = len(lons)

    ssta_all = ds["ssta"].values.squeeze()

    # Temporal filter: post-1910 (years >= start_year)
    year_mask = years_all >= args.start_year
    years = years_all[year_mask]
    ssta_sub = ssta_all[year_mask]
    n_years = len(years)

    print(
        f"      Full record: {years_all[0]} - {years_all[-1]} ({len(years_all)} years)"
    )
    print(
        f"      Filtered period: {years[0]} - {years[-1]} ({n_years} years)"
    )
    print(
        f"      Domain: {lats.min():.0f}°N - {lats.max():.0f}°N, {lons.min():.0f}°E - {lons.max():.0f}°E ({n_lat} lats x {n_lon} lons)"
    )

    # Ocean mask
    std_spatial = np.nanstd(ssta_sub, axis=0)
    ocean_mask = (~np.isnan(ssta_sub[0])) & (std_spatial > 1e-4)
    n_ocean = int(np.sum(ocean_mask))
    print(
        f"      Valid ocean points: {n_ocean} / {n_lat * n_lon} ({n_ocean / (n_lat * n_lon):.1%})"
    )

    M_raw = ssta_sub[:, ocean_mask]

    # -------------------------------------------------------------------------
    # 2. Linear Detrending & Trend Map
    # -------------------------------------------------------------------------
    print("\n[2/7] Computing and removing linear trend at each gridpoint...")
    # Calculate slope and intercept: M(t, x) = slope(x)*t + intercept(x) + residual(t, x)
    t = years - np.mean(years)
    var_t = np.sum(t**2)
    slopes = np.dot(t, M_raw) / var_t  # deg C / year
    decadal_trend = slopes * 10.0      # deg C / decade

    # Residual data matrix F (detrended): scipy.signal.detrend removes linear fit
    F_residuals = scipy.signal.detrend(M_raw, axis=0, type="linear")

    print(
        f"      Decadal trend range: {decadal_trend.min():+.3f} to {decadal_trend.max():+.3f} °C/decade"
    )
    print(
        f"      Basin-mean trend: {np.mean(decadal_trend):+.3f} °C/decade"
    )

    # Plot Linear Trend Map
    trend_map = vector_to_map(decadal_trend, ocean_mask, n_lat, n_lon)
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    setup_atlantic_map(ax, lons, lats)
    tlim = max(np.nanpercentile(np.abs(trend_map), 99), 0.15)
    mesh = ax.pcolormesh(
        lons,
        lats,
        trend_map,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-tlim,
        vmax=tlim,
        shading="auto",
    )
    cbar = plt.colorbar(mesh, orientation="horizontal", pad=0.08, shrink=0.75, aspect=20)
    cbar.set_label("Linear Trend ($^\\circ$C per decade, 1910–2020)", fontsize=10)
    ax.set_title(
        f"North Atlantic SST Linear Trend ({years[0]}–{years[-1]})\n(Removed Prior to EOF Calculation)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fpath = os.path.join(fig_dir, "01_linear_trend_map.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")

    # -------------------------------------------------------------------------
    # 3. Area Weighting & EOF Decomposition of Residuals
    # -------------------------------------------------------------------------
    print("\n[3/7] Weighting residuals and computing EOFs via SVD...")
    lat_2d, _ = np.meshgrid(lats, lons, indexing="ij")
    w_2d = np.sqrt(np.cos(np.deg2rad(lat_2d)))
    w_ocean = w_2d[ocean_mask]
    F_w = F_residuals * w_ocean

    # Method A: Direct SVD
    U, S, Vt = np.linalg.svd(F_w, full_matrices=False)
    eigenvalues = S**2
    PCs = U * S
    EOFs = Vt

    # Method B: Dual Covariance Validation (FF^T)
    L = np.dot(F_w, F_w.T)
    w_dual, _ = np.linalg.eigh(L)
    eigenvalues_dual = np.sort(w_dual)[::-1]
    rel_err = np.max(
        np.abs(eigenvalues[:20] - eigenvalues_dual[:20]) / eigenvalues[:20]
    )
    print(
        f"      Max relative discrepancy in top 20 eigenvalues (SVD vs Dual): {rel_err:.2e}"
    )

    # -------------------------------------------------------------------------
    # 4. Variance Explained & North's Rule of Thumb
    # -------------------------------------------------------------------------
    print("\n[4/7] Evaluating variance spectrum and North's error bars...")
    total_var = np.sum(eigenvalues)
    var_exp = (eigenvalues / total_var) * 100.0
    cum_var = np.cumsum(var_exp)
    delta_lambda = var_exp * np.sqrt(2.0 / n_years)

    print("\n" + "-" * 76)
    print(
        f"{'Mode':<6} | {'Eigenvalue':<14} | {'Var Expl (%)':<14} | {'North Error (+/-)':<18} | {'Cumulative (%)':<14}"
    )
    print("-" * 76)
    for i in range(10):
        print(
            f"EOF {i+1:<3} | {eigenvalues[i]:<14.2f} | {var_exp[i]:<14.2f} | {delta_lambda[i]:<18.2f} | {cum_var[i]:<14.2f}"
        )
    print("-" * 76)

    # Scree Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    modes = np.arange(1, 16)
    ax1.errorbar(
        modes,
        var_exp[:15],
        yerr=delta_lambda[:15],
        fmt="-o",
        color="#1f77b4",
        ecolor="#d62728",
        elinewidth=1.8,
        capsize=4,
        capthick=1.5,
        markersize=6,
        label=r"Variance Explained $\pm \Delta\lambda$",
    )
    ax1.set_xlabel("EOF Mode Number", fontsize=11)
    ax1.set_ylabel("Variance Explained (%)", fontsize=11)
    ax1.set_title(
        f"Detrended Eigenvalue Spectrum & North's Error ({years[0]}–{years[-1]})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xticks(modes)
    ax1.legend(loc="upper right")

    ax2.plot(
        modes, cum_var[:15], "-s", color="#2ca02c", linewidth=2, markersize=6
    )
    ax2.axhline(50, color="gray", linestyle="--", alpha=0.7, label="50% Threshold")
    ax2.set_xlabel("Number of Retained Modes", fontsize=11)
    ax2.set_ylabel("Cumulative Variance Explained (%)", fontsize=11)
    ax2.set_title("Cumulative Variance Explained", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xticks(modes)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fpath = os.path.join(fig_dir, "02_scree_plot_north_rule.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")

    # -------------------------------------------------------------------------
    # 5. Spatial Patterns and PC Time Series
    # -------------------------------------------------------------------------
    print(
        f"\n[5/7] Reconstructing leading {args.n_modes} detrended EOF modes and PCs..."
    )
    mode_names = [
        "EOF 1: Atlantic Multidecadal Variability (AMV / AMO)",
        "EOF 2: North Atlantic Tripole Mode (NAO Imprint)",
        "EOF 3: Subpolar Gyre vs. Subtropics Gradient Mode",
        "EOF 4: High-Latitude / Nordic Seas Mode",
    ]

    pc_norm_all = np.zeros((n_years, args.n_modes))
    spatial_regs = np.zeros((args.n_modes, n_lat, n_lon))

    for k in range(args.n_modes):
        pc = PCs[:, k].copy()
        pc_std = np.std(pc)
        pc_norm = pc / pc_std

        # Regress unweighted detrended data F onto standardized PC
        reg = np.dot(F_residuals.T, pc_norm) / n_years

        # Set convention: for Mode 1 (AMV), positive in the warm subpolar/subtropical Atlantic
        if k == 0 and np.mean(reg) < 0:
            pc_norm = -pc_norm
            reg = -reg

        pc_norm_all[:, k] = pc_norm
        spatial_regs[k] = vector_to_map(reg, ocean_mask, n_lat, n_lon)

        # Plot individual mode
        fig = plt.figure(figsize=(13, 5))
        ax_map = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        setup_atlantic_map(ax_map, lons, lats)
        vlim = max(np.nanpercentile(np.abs(spatial_regs[k]), 99), 0.25)

        mesh = ax_map.pcolormesh(
            lons,
            lats,
            spatial_regs[k],
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        mname = mode_names[k] if k < len(mode_names) else f"EOF Mode {k+1}"
        ax_map.set_title(
            f"{mname} (Detrended)\nSpatial Pattern ({var_exp[k]:.1f}% Variance)",
            fontsize=11,
            fontweight="bold",
        )
        cbar = plt.colorbar(
            mesh,
            ax=ax_map,
            orientation="horizontal",
            pad=0.08,
            shrink=0.75,
            aspect=20,
        )
        cbar.set_label("SST Anomaly ($^\\circ$C per $1\\sigma$ PC)", fontsize=9)

        ax_ts = fig.add_subplot(1, 2, 2)
        ax_ts.plot(
            years,
            pc_norm,
            color="#1f77b4",
            linewidth=1.2,
            alpha=0.6,
            label="Annual Detrended PC",
        )
        ax_ts.plot(
            years[5:-5],
            running_mean(pc_norm, 11)[5:-5],
            color="#d62728",
            linewidth=2.4,
            label="11-Year Running Mean",
        )
        ax_ts.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_ts.set_xlim(years[0], years[-1])
        ax_ts.set_ylim(-3.2, 3.2)
        ax_ts.set_xlabel("Year", fontsize=11)
        ax_ts.set_ylabel(r"Normalized Amplitude ($\sigma$)", fontsize=11)
        ax_ts.set_title(
            f"Principal Component {k+1} Time Series ({years[0]}–{years[-1]})",
            fontsize=11,
            fontweight="bold",
        )
        ax_ts.grid(True, linestyle=":", alpha=0.6)
        ax_ts.legend(loc="upper left", framealpha=0.9, fontsize=9)

        plt.tight_layout()
        fname = f"0{k+3}_eof{k+1}_pattern_and_timeseries.png"
        fpath = os.path.join(fig_dir, fname)
        fig.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"      Saved: {fpath}")

    # Combined summary figure (using add_gridspec, without underlying 0-to-1 axes)
    print("\n      Generating comprehensive 4-mode summary panel...")
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1.2])

    for k in range(min(4, args.n_modes)):
        ax_m = fig.add_subplot(gs[k, 0], projection=ccrs.PlateCarree())
        setup_atlantic_map(ax_m, lons, lats)
        vlim = max(np.nanpercentile(np.abs(spatial_regs[k]), 99), 0.25)
        m = ax_m.pcolormesh(
            lons,
            lats,
            spatial_regs[k],
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
        )
        ax_m.set_title(
            f"{mode_names[k]} ({var_exp[k]:.1f}%)",
            fontsize=10,
            fontweight="bold",
        )
        plt.colorbar(
            m, ax=ax_m, orientation="horizontal", pad=0.08, shrink=0.75, aspect=20
        )

        ax_t = fig.add_subplot(gs[k, 1])
        ax_t.plot(years, pc_norm_all[:, k], color="#1f77b4", alpha=0.5, lw=1.0)
        ax_t.plot(
            years[5:-5],
            running_mean(pc_norm_all[:, k], 11)[5:-5],
            color="#d62728",
            lw=2.0,
        )
        ax_t.axhline(0, color="k", ls="--", lw=0.7, alpha=0.6)
        ax_t.set_xlim(years[0], years[-1])
        ax_t.set_ylim(-3.2, 3.2)
        ax_t.set_title(f"Detrended PC {k+1} Time Series", fontsize=10, fontweight="bold")
        ax_t.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fpath = os.path.join(fig_dir, "07_leading_4_eofs_summary.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")

    # -------------------------------------------------------------------------
    # 6. Homogeneous Correlation & Local Variance Maps (§4.1)
    # -------------------------------------------------------------------------
    print(
        "\n[6/7] Computing Homogeneous Correlation and Local Variance Explained maps (§4.1)..."
    )
    corr_maps = np.zeros((2, n_lat, n_lon))
    var_loc_maps = np.zeros((2, n_lat, n_lon))

    F_std = F_residuals / np.std(F_residuals, axis=0)
    for k in range(2):
        pc = pc_norm_all[:, k]
        corr = np.mean(F_std * pc[:, None], axis=0)
        corr_maps[k] = vector_to_map(corr, ocean_mask, n_lat, n_lon)
        var_loc_maps[k] = vector_to_map((corr**2) * 100.0, ocean_mask, n_lat, n_lon)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    c_titles = [
        (
            "Detrended EOF 1: Homogeneous Correlation ($r$)",
            "Detrended EOF 1: Local Variance Explained ($r^2 \\times 100\\%$)",
        ),
        (
            "Detrended EOF 2: Homogeneous Correlation ($r$)",
            "Detrended EOF 2: Local Variance Explained ($r^2 \\times 100\\%$)",
        ),
    ]

    for row in range(2):
        ax_c = axes[row, 0]
        setup_atlantic_map(ax_c, lons, lats)
        mesh_c = ax_c.pcolormesh(
            lons,
            lats,
            corr_maps[row],
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
        )
        ax_c.set_title(c_titles[row][0], fontsize=11, fontweight="bold")
        cb_c = plt.colorbar(
            mesh_c, ax=ax_c, orientation="horizontal", pad=0.08, shrink=0.75
        )
        cb_c.set_label("Correlation ($r$)", fontsize=9)

        ax_v = axes[row, 1]
        setup_atlantic_map(ax_v, lons, lats)
        mesh_v = ax_v.pcolormesh(
            lons,
            lats,
            var_loc_maps[row],
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            vmin=0,
            vmax=80,
        )
        ax_v.set_title(c_titles[row][1], fontsize=11, fontweight="bold")
        cb_v = plt.colorbar(
            mesh_v, ax=ax_v, orientation="horizontal", pad=0.08, shrink=0.75
        )
        cb_v.set_label("Local Variance Explained (%)", fontsize=9)

    plt.tight_layout()
    fpath = os.path.join(fig_dir, "08_homogeneous_correlation_local_variance.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")

    # -------------------------------------------------------------------------
    # 7. Varimax Rotation (Appendix A)
    # -------------------------------------------------------------------------
    print("\n[7/7] Applying Varimax rotation to leading 3 detrended EOFs (Appendix A)...")
    k_rot = 3
    E_trunc = EOFs[:k_rot, :].T
    E_rot, _ = varimax(E_trunc)
    PCs_rot = np.dot(F_w, E_rot)

    fig = plt.figure(figsize=(13, 3.8 * k_rot))
    gs = fig.add_gridspec(k_rot, 2, width_ratios=[1, 1.2])

    for k in range(k_rot):
        ax_map = fig.add_subplot(gs[k, 0], projection=ccrs.PlateCarree())
        setup_atlantic_map(ax_map, lons, lats)
        pattern_rot = vector_to_map(
            E_rot[:, k] / w_ocean, ocean_mask, n_lat, n_lon
        )
        vlim = np.nanpercentile(np.abs(pattern_rot), 99)

        mesh = ax_map.pcolormesh(
            lons,
            lats,
            pattern_rot,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
        )
        ax_map.set_title(
            f"Rotated Detrended EOF {k+1} Pattern (Varimax)",
            fontsize=11,
            fontweight="bold",
        )
        plt.colorbar(mesh, ax=ax_map, orientation="horizontal", pad=0.08, shrink=0.75)

        ax_ts = fig.add_subplot(gs[k, 1])
        ts_rot = PCs_rot[:, k]
        ts_rot_norm = (ts_rot - np.mean(ts_rot)) / np.std(ts_rot)
        ax_ts.plot(
            years,
            ts_rot_norm,
            color="#2ca02c",
            alpha=0.6,
            label="Annual Rotated PC",
        )
        ax_ts.plot(
            years[5:-5],
            running_mean(ts_rot_norm, 11)[5:-5],
            color="black",
            linewidth=2.0,
            label="11-Yr Mean",
        )
        ax_ts.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax_ts.set_xlim(years[0], years[-1])
        ax_ts.set_title(
            f"Rotated Detrended PC {k+1} Time Series",
            fontsize=11,
            fontweight="bold",
        )
        ax_ts.grid(True, linestyle=":", alpha=0.6)
        ax_ts.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fpath = os.path.join(fig_dir, "09_rotated_eofs_varimax.png")
    fig.savefig(fpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")

    print("\n" + "=" * 78)
    print(f" Analysis complete! All figures successfully saved to '{fig_dir}/'.")
    print("=" * 78)


if __name__ == "__main__":
    main()
