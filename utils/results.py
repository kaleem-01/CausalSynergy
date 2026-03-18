import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

# -----------------------------
# 1) Reshaping / combining reps
# -----------------------------
def stack_replication_metrics(
    rep_results,
    metrics,
    rep_id_attr=None,   # optional: if your rep object already stores a replication id
):
    """
    Convert a list of replication results (each with .metrics shaped:
      index = metric names, columns = i)
    into one long DataFrame with columns:
      i, rep, p_noise, samples, <metrics...>

    Parameters
    ----------
    rep_results : list
        List like [results_rep0, results_rep1, ...] where each has .metrics (pd.DataFrame).
    mapping_csv_path : str
        Path to grid_mapping.csv with columns: i, p_noise, samples
    metrics : list[str]
        Metric row-names to extract (must match df.index exactly).
    rep_id_attr : str | None
        If each replication object has an attribute with replication id, pass its name.
        Otherwise uses enumerate index.
    """
    long_parts = []
    for r_idx, res in enumerate(rep_results):
        rep_id = getattr(res, rep_id_attr) if rep_id_attr else r_idx

        df = res.eval_metrics.T.copy()

        # Ensure columns are dataset ids i (ints)
        try:
            df.columns = df.columns.astype(int)
        except Exception:
            # if columns are strings like "0","1",...
            df.columns = [int(c) for c in df.columns]

        # Transpose: rows become i
        t = df.T.reset_index().rename(columns={"index": "i"})
        # t["idx"] = t["idx"].astype(int)
        t["rep"] = rep_id

        # Keep only requested metrics (ignore missing gracefully)
        keep = ["i", "rep"] + [m for m in metrics if m in t.columns]
        t = t[keep]

        long_parts.append(t)

    long = pd.concat(long_parts, ignore_index=True)
    # If "BIC" column exists and contains list/array, extract last element
    if "BIC" in long.columns:
        long["BIC"] = long["BIC"].apply(
            lambda x: float(x[-1]) if isinstance(x, (list, tuple, np.ndarray, pd.Series)) else float(x)
        )
    # long = long.merge(mapping, on="i", how="left")

    # sanity
    # if long["n_roots"].isna().any() or long["n_syn"].isna().any():
    #     bad = long.loc[long["n_roots"].isna() | long["n_syn"].isna(), "i"].unique().tolist()
    #     raise ValueError(f"Some i values not found in grid_mapping.csv: {bad[:20]}{'...' if len(bad)>20 else ''}")

    return long

# # -----------------------------
# # 2) Plot helpers
# # -----------------------------
def pivot_grid(long_df, index, column, value_col, aggfunc="max"):
    """Return 2D grid (rows=p_noise, cols=samples) for a metric."""
    grid = (long_df
            .pivot_table(index=index, columns=column, values=value_col, aggfunc=aggfunc)
            .sort_index()
            .sort_index(axis=1))
    return grid




# def plot_heatmap_from_grid(grid, title, xlabel=None, ylabel=None, cbar_label=None, ax=None):
#     """Matplotlib heatmap from a pivoted grid DataFrame."""
#     if ax is None:
#         fig, ax = plt.subplots()
#     else:
#         fig = ax.figure

#     im = ax.imshow(grid.values, aspect="auto")
#     cb = fig.colorbar(im, ax=ax)
#     cb.set_label(cbar_label if cbar_label else "")

#     ax.set_xticks(range(len(grid.columns)))
#     ax.set_xticklabels(grid.columns)
#     ax.set_yticks(range(len(grid.index)))
#     ax.set_yticklabels(grid.index)

#     try:
#         ax.ticklabel_format(axis="x", style="sci")
#     except Exception:
#         pass

#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)

#     return fig, ax


def plot_heatmap_from_grid(
    grid,
    title,
    xlabel=None,
    ylabel=None,
    cbar_label=None,
    ax=None,
    vmin=None,
    vmax=None,
    cmap=None,
    norm=None,
    add_colorbar=True,
):
    """Matplotlib heatmap from a pivoted grid DataFrame, with optional shared scaling."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    data = grid.to_numpy(dtype=float)

    # Decide normalization: explicit norm overrides vmin/vmax
    if norm is None:
        # Only set vmin/vmax if provided; otherwise matplotlib autoscales per-plot
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
    else:
        im = ax.imshow(
            data,
            aspect="auto",
            origin="upper",
            norm=norm,
            cmap=cmap,
        )

    if add_colorbar:
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(cbar_label or "")

    # Tick positions are indices
    x_pos = np.arange(len(grid.columns))
    y_pos = np.arange(len(grid.index))
    ax.set_xticks(x_pos)
    ax.set_yticks(y_pos)

    def _is_numeric_seq(seq):
        try:
            _ = np.asarray(seq, dtype=float)
            return True
        except Exception:
            return False

    cols = list(grid.columns)
    rows = list(grid.index)

    cols_numeric = _is_numeric_seq(cols)
    rows_numeric = _is_numeric_seq(rows)

    if cols_numeric:
        col_vals = np.asarray(cols, dtype=float)
        use_sci = (np.nanmax(np.abs(col_vals)) >= 1e5) or (
            np.nanmin(np.abs(col_vals[np.nonzero(col_vals)])) < 1e-2
            if np.any(col_vals != 0) else False
        )

        if use_sci:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda i, _: f"{col_vals[int(round(i))]:.0e}"
                    if 0 <= int(round(i)) < len(col_vals) else ""
                )
            )
        else:
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda i, _: f"{col_vals[int(round(i))]:g}"
                    if 0 <= int(round(i)) < len(col_vals) else ""
                )
            )
    else:
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda i, _: str(cols[int(round(i))])
                if 0 <= int(round(i)) < len(cols) else ""
            )
        )

    if rows_numeric:
        row_vals = np.asarray(rows, dtype=float)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda i, _: f"{row_vals[int(round(i))]:g}"
                if 0 <= int(round(i)) < len(row_vals) else ""
            )
        )
    else:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda i, _: str(rows[int(round(i))])
                if 0 <= int(round(i)) < len(rows) else ""
            )
        )

    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.set_title(title)

    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="both", which="major", labelsize=9)

    return fig, ax, im


def _global_minmax(arrays, robust=False, q=(2, 98), symmetric=False):
    """Compute global vmin/vmax across a list of arrays, ignoring NaNs."""
    vals = []
    for a in arrays:
        if a is None:
            continue
        flat = np.asarray(a, dtype=float).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            vals.append(flat)
    if not vals:
        return None, None

    allv = np.concatenate(vals)

    if robust:
        lo, hi = np.nanpercentile(allv, [q[0], q[1]])
    else:
        lo, hi = np.nanmin(allv), np.nanmax(allv)

    if symmetric:
        m = np.nanmax(np.abs([lo, hi]))
        return -m, m
    return lo, hi


def plot_final_heatmap(
    results_by_label,
    subset_metrics,
    plot_metric,
    x_col="samples",
    y_col="p_noise",
    ncols=None,
    filter=None,
    aggfunc="mean",
    figsize_per_col=(5, 4),
    sharey=True,
    standardize_colors=True,
    robust=False,
    robust_q=(2, 98),
    symmetric=False,
    fixed_vmin=None,
    fixed_vmax=None,
    cmap=None,
    single_colorbar=True,
):
    """
    Plot heatmaps for an arbitrary number of algorithms/results, with standardized colors.

    New args
    --------
    standardize_colors : bool
        If True, all panels share the same vmin/vmax (unless fixed_vmin/vmax given).
    robust : bool
        If True, use percentiles robust_q for vmin/vmax instead of min/max (helps with outliers).
    symmetric : bool
        If True, force symmetric bounds around 0 (useful for signed metrics).
    fixed_vmin, fixed_vmax : float | None
        If provided, overrides computed bounds.
    cmap : str | None
        Optional colormap to pass through.
    single_colorbar : bool
        If True, draw one shared colorbar for the whole figure (recommended when standardized).
    """
    if not isinstance(results_by_label, dict) or len(results_by_label) == 0:
        raise ValueError("results_by_label must be a non-empty dict like {'PC': results_pc, ...}.")

    labels = list(results_by_label.keys())
    k = len(labels)

    # Choose subplot layout
    if ncols is None:
        ncols = min(4, k)
    nrows = (k + ncols - 1) // ncols

    fig_w = figsize_per_col[0] * ncols
    fig_h = figsize_per_col[1] * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        sharey=sharey
    )

    axes_list = [axes] if k == 1 else list(np.ravel(axes))

    # 1) Build all grids first
    grids_by_label = {}
    grid_arrays = []
    for label in labels:
        results = results_by_label[label]
        df = stack_replication_metrics(results, metrics=subset_metrics)
        if filter is not None:
            filter_col, filter_val = filter
            df = df[df[filter_col] == filter_val]
        grid = pivot_grid(df, index=y_col, column=x_col, value_col=plot_metric, aggfunc=aggfunc)
        grids_by_label[label] = grid
        grid_arrays.append(grid.to_numpy(dtype=float))

    # 2) Compute shared bounds / norm
    vmin = vmax = None
    norm = None
    if standardize_colors:
        if fixed_vmin is not None or fixed_vmax is not None:
            vmin = fixed_vmin
            vmax = fixed_vmax
        else:
            vmin, vmax = _global_minmax(
                grid_arrays,
                robust=robust,
                q=robust_q,
                symmetric=symmetric
            )
        norm = Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else None

    # 3) Plot panels (optionally without per-panel colorbars)
    last_im = None
    for i, label in enumerate(labels):
        ax = axes_list[i]
        grid = grids_by_label[label]

        _, _, im = plot_heatmap_from_grid(
            grid,
            title=f"{label} – {plot_metric}",
            cbar_label=plot_metric,
            ax=ax,
            cmap=cmap,
            norm=norm if standardize_colors else None,
            vmin=None if standardize_colors else fixed_vmin,
            vmax=None if standardize_colors else fixed_vmax,
            add_colorbar=(not single_colorbar) or (not standardize_colors),
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        last_im = im

    # Hide unused axes
    for j in range(k, len(axes_list)):
        axes_list[j].set_visible(False)

    # 4) One shared colorbar
    if standardize_colors and single_colorbar and last_im is not None:
        cb = fig.colorbar(last_im, ax=[ax for ax in axes_list[:k]], shrink=0.9)
        cb.set_label(plot_metric)

    return fig, axes

def plot_metric_vs_x(long_df, metric_col, x_col, title, ylabel=None, by=None, band="std", ax=None):
    """
    Line plot of metric vs x_col with mean ± std (or sem) bands.
    If `by` is set, draws one line per group.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    group_cols = [x_col] if by is None else [by, x_col]

    stats = (long_df
             .groupby(group_cols, as_index=False)[metric_col]
             .agg(mean="mean", std="std", count="count")
             .sort_values(group_cols))

    if by is None:
        x = stats[x_col].to_numpy()
        y = stats["mean"].to_numpy()
        ax.plot(x, y, marker="o")

        if band is not None:
            if band == "std":
                s = stats["std"].to_numpy()
            elif band == "sem":
                s = stats["std"].to_numpy() / np.sqrt(np.maximum(stats["count"].to_numpy(), 1))
            else:
                raise ValueError("band must be one of: None, 'std', 'sem'")
            ax.fill_between(x, y - s, y + s, alpha=0.2)
    else:
        for key, g in stats.groupby(by):
            x = g[x_col].to_numpy()
            y = g["mean"].to_numpy()

            ax.plot(x, y, marker="o", label=f"{by}={key}")

            if band is not None:
                if band == "std":
                    s = g["std"].to_numpy()
                elif band == "sem":
                    s = g["std"].to_numpy() / np.sqrt(np.maximum(g["count"].to_numpy(), 1))
                else:
                    raise ValueError("band must be one of: None, 'std', 'sem'")
                ax.fill_between(x, y - s, y + s, alpha=0.2)


    # ax.legend()

    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel if ylabel else metric_col)
    ax.set_title(title)

    return fig, ax


def plot_final_metric_vs_x(results_by_label, subset_metrics,
                           metric_col, x_col, title_suffix=None, ylabel=None,
                           by=None, band="std",
                           ncols=None, figsize_per_col=(5, 4),
                           sharey=True, sharex=False):
    """
    Plot metric-vs-x panels for an arbitrary number of algorithms/results,
    taking the SAME kind of input as `plot_final_heatmap` (label -> results),
    and stacking internally via `stack_replication_metrics`.

    Parameters
    ----------
    results_by_label : dict[str, list]
        Dict like {"PC": results_pc, "GES": results_ges, ...}
        Values are whatever `stack_replication_metrics` expects.
    subset_metrics : list[str]
        Metrics to stack so `metric_col` is available.
    metric_col : str
        Metric column to plot on y-axis (must exist after stacking).
    x_col : str
        Column to plot on x-axis (must exist in stacked df).
    title_suffix : str | None
        Optional extra text appended to each panel title.
    ylabel : str | None
        Y-axis label; defaults to metric_col.
    by : str | None
        If provided, one line per group (legend per panel).
    band : {"std","sem",None}
        Uncertainty band type.
    ncols : int | None
        Number of columns in subplot grid. If None: uses up to 4 columns.
    figsize_per_col : tuple[float, float]
        Base (width, height) per column for automatic figsize scaling.
    sharey : bool
        Share y-axis across panels.
    sharex : bool
        Share x-axis across panels.

    Returns
    -------
    fig, axes
    """
    if not isinstance(results_by_label, dict) or len(results_by_label) == 0:
        raise ValueError("results_by_label must be a non-empty dict like {'PC': results_pc, ...}.")

    labels = list(results_by_label.keys())
    k = len(labels)

    # Choose subplot layout
    if ncols is None:
        ncols = min(4, k)
    nrows = (k + ncols - 1) // ncols

    fig_w = figsize_per_col[0] * ncols
    fig_h = figsize_per_col[1] * nrows
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        sharey=sharey,
        sharex=sharex
    )

    # Make axes always iterable
    if k == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    # Build and plot each panel
    for i, label in enumerate(labels):
        results = results_by_label[label]

        long_df = stack_replication_metrics(results, metrics=subset_metrics)

        # Helpful early failure if user forgets to include needed columns
        missing = [c for c in [metric_col, x_col] + ([by] if by else []) if c not in long_df.columns]
        if missing:
            raise KeyError(
                f"[{label}] stacked df is missing columns: {missing}. "
                f"Available columns: {list(long_df.columns)}"
            )

        suffix = f" ({title_suffix})" if title_suffix else ""
        plot_metric_vs_x(
            long_df,
            metric_col=metric_col,
            x_col=x_col,
            title=f"{label} – {metric_col} vs {x_col}{suffix}",
            ylabel=ylabel if ylabel else metric_col,
            by=by,
            band=band,
            ax=axes_list[i]
        )

    # Hide any unused axes
    for j in range(k, len(axes_list)):
        axes_list[j].set_visible(False)
        

    return fig, axes