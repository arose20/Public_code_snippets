import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

import scipy.sparse as sp
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import KDTree

from shapely.geometry import Polygon, MultiPoint, Point
from shapely.vectorized import contains

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mticker
from skimage.color import lab2rgb

from collections import defaultdict
import json

from typing import Optional, Tuple, List, Dict, Union


def restore_shapes_from_json(adata):
    """
    Convert JSON-stringified 'shapes' in ROI_info back to list-of-dicts.
    """
    for roi_dict in adata.uns['ROI_info'].values():
        if "rois" in roi_dict and "shapes" in roi_dict["rois"]:
            shapes = roi_dict["rois"]["shapes"]
            if isinstance(shapes, str):
                roi_dict["rois"]["shapes"] = json.loads(shapes)
    return adata



def kde_category_spatial(
    adata: ad.AnnData,
    library_id: str,
    category: str,
    group: str,
    point_size: int = 2,
    background_alpha: float = 0.2,
    scatter: bool = True,
    img: Optional[str] = "hires",
    factor: int = 5,
    figsize: Tuple[int, int] = (10, 10),
    spatial_key: str = "spatial",
    show_axes: bool = True,
    show_legend: bool = True,
    save: Optional[str] = None,
    cmap: str = "turbo",
    bw_adjust: float = 0.15,
    levels: int = 50,
    thresh: float = 0.05,
    gridsize: int = 200,
    
    plot_scalebar: bool = True,
    scalebar_um: Optional[float] = None,
    scalebar_color: str = "black",
    scalebar_loc: str = "bottom_right",
    
    
) -> None:
    """
    Plot a spatial kernel density estimate (KDE) for a specific category group
    over a spatial transcriptomics image.
    """
    # Check category exists
    if category not in adata.obs.columns:
        raise ValueError(f"Category '{category}' not found in adata.obs.")

    data = adata.obs[category]
    coords = adata.obsm["spatial"]
    mask = data == group

    if img in adata.uns[spatial_key][library_id]["images"]:
        microns_per_pixel = adata.uns[spatial_key][library_id]["scalefactors"][
            "pixel_size"
        ]
        img_plot = adata.uns[spatial_key][library_id]["images"][img]

        x_plot = coords[:, 0] / microns_per_pixel / factor
        y_plot = coords[:, 1] / microns_per_pixel / factor

        group_x = coords[mask, 0] / microns_per_pixel / factor
        group_y = coords[mask, 1] / microns_per_pixel / factor

        img_plot = img_plot[::factor, ::factor, :]
    else:
        x_plot = coords[:, 0]
        y_plot = coords[:, 1]
        group_x = coords[mask, 0]
        group_y = coords[mask, 1]

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=figsize)

    if img is not None:
        ax.imshow(img_plot, origin="lower", alpha=0.9)

    if scatter:
        ax.scatter(
            x_plot,
            y_plot,
            s=point_size,
            color="grey",
            alpha=background_alpha,
            label="All spots",
        )

    sns.kdeplot(
        x=group_x,
        y=group_y,
        cmap=cmap,
        fill=True,
        alpha=0.5,
        bw_adjust=bw_adjust,
        levels=levels,
        thresh=thresh,
        gridsize=gridsize,
        ax=ax,
    )

    ax.set_aspect("equal")

    # ---------------- Axes ----------------
    if not show_axes:
        ax.axis("off")
    else:
        ax.set_xlabel("X (spatial)")
        ax.set_ylabel("Y (spatial)")

    ax.set_title(f"KDE of '{group}' in {category}")

    # ---------------- Legend ----------------
    if show_legend and scatter:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="grey",
                alpha=background_alpha,
                label="All spots",
                markersize=6,
            )
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )

    # ---------------- Colorbar ----------------
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("KDE density")
    
    
    # ---------------- Scale bar ----------------
    if plot_scalebar:
        # If user didn't provide, choose automatically
        if scalebar_um is None:
            # Determine the width in microns of the plotted image

            # width = full image in microns
            img_width_um = img_plot.shape[1] * microns_per_pixel * factor

            # Choose a nice round number ~1/5 of image width
            nice_numbers = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
            scalebar_um = nice_numbers[(np.abs(nice_numbers - img_width_um / 5)).argmin()]

        # Convert to pixels in the plotted image
        bar_length_px = scalebar_um / microns_per_pixel / factor
        bar_length_px = int(round(bar_length_px))

        img_height, img_width = img_plot.shape[:2]
        bar_height = img_height * 0.01  # 1% of image height

        # Location
        if scalebar_loc == "bottom_right":
            bar_x = img_width * 0.95 - bar_length_px
        elif scalebar_loc == "bottom_left":
            bar_x = img_width * 0.05
        else:
            raise ValueError("`scalebar_loc` must be 'bottom_right' or 'bottom_left'")
        bar_y = img_height * 0.05  # 5% from bottom

        rect = patches.Rectangle(
            (bar_x, bar_y),
            width=bar_length_px,
            height=bar_height,
            color=scalebar_color,
            edgecolor=scalebar_color
        )
        ax.add_patch(rect)

        # Text label above the bar
        ax.text(
            bar_x + bar_length_px / 2,
            bar_y + bar_height * 1.5,  # just above the bar
            f"{int(scalebar_um)} µm",
            color=scalebar_color,
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )
    
    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=300)

    plt.show()


def generate_random_colors(num_colors: int) -> List[str]:
    """
    Generate a list of random hex colors.

    Parameters
    ----------
    num_colors : int
        Number of random colors to generate.

    Returns
    -------
    List[str]
        List of hex color strings.
    """
    return [
        "#"
        + "".join(
            [np.random.choice(list("0123456789ABCDEF")) for j in range(6)]
        )
        for i in range(num_colors)
    ]


#def generate_random_colors(n):
#    """Helper: generate n visually distinct colors."""
#    import matplotlib.cm as cm
#    return cm.get_cmap("tab20", n).colors

def spatial_plot(
    adata: ad.AnnData,
    library_id: str,
    category: Optional[str] = None,
    groups: Optional[List[str]] = None,
    roi_names_to_plot: Optional[List[str]] = None,
    roi_colors: Optional[Union[dict, list]] = None,
    roi_linewidth: Optional[float] = 2,
    img: str = "hires",
    scatter: bool = False,
    point_size: int = 2,
    factor: int = 5,
    figsize: Tuple[int, int] = (10, 10),
    spatial_key: str = "spatial",
    cmap: Optional[Union[str, dict]] = None,
    spot_alpha: float = 0.9,
    plot_others: str = "hide",
    other_alpha: float = 0.3,
    plot_rectangle: bool = False,
    subset: bool = False,
    box_dims: Tuple[float, float, float, float] = (0, 0, 0, 0),
    show_axes: bool = False,
    gene_vmin: Optional[float] = None,
    gene_vmax: Optional[float] = None,
    
    plot_scalebar: bool = True,
    scalebar_um: Optional[float] = None,
    scalebar_color: str = "black",
    scalebar_loc: str = "bottom_right",
    
    title: Optional[Union[str, bool]] = None,
    
    save: Optional[str] = None,
    dpi: int = 300,
    close: bool = True,
) -> None:

    coords = adata.obsm["spatial"]
    microns_per_pixel = adata.uns[spatial_key][library_id]["scalefactors"]["pixel_size"]
    img_plot = adata.uns[spatial_key][library_id]["images"][img]

    x_micron = coords[:, 0]
    y_micron = coords[:, 1]
    x_plot = (x_micron / microns_per_pixel) / factor
    y_plot = (y_micron / microns_per_pixel) / factor
    img_plot = img_plot[::factor, ::factor, :]

    row_start_um, row_end_um, col_start_um, col_end_um = box_dims
    x0 = col_start_um / microns_per_pixel / factor
    x1 = col_end_um / microns_per_pixel / factor
    y0 = row_start_um / microns_per_pixel / factor
    y1 = row_end_um / microns_per_pixel / factor

    if subset:
        mask = (
            (x_micron >= col_start_um)
            & (x_micron <= col_end_um)
            & (y_micron >= row_start_um)
            & (y_micron <= row_end_um)
        )
        adata = adata[mask].copy()
        x_plot = x_plot[mask]
        y_plot = y_plot[mask]
        x0i, x1i = int(x0), int(x1)
        y0i, y1i = int(y0), int(y1)
        img_plot = img_plot[y0i:y1i, x0i:x1i, :]
        x_plot -= x0i
        y_plot -= y0i

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(img_plot, origin="lower")

    # ---------------- Collect legend handles ----------------
    scatter_handles = []
    roi_handles = []

    # ===================== SCATTER =====================
    if category is not None:
        scatter = True
    
    if scatter:
        if category is None:
            raise ValueError("`category` must be provided when scatter=True")

        in_var = category in adata.var_names
        in_obs = category in adata.obs.columns

        # ---------- DISPATCH ----------
        if in_var and in_obs:
            raise ValueError(
                f"`{category}` found in both adata.var_names and adata.obs.columns. "
                "Please disambiguate."
            )

        if not (in_var or in_obs):
            raise KeyError(
                f"`{category}` not found in adata.var_names or adata.obs.columns."
            )

        # ======================================================
        # GENE (always continuous)
        # ======================================================
        if in_var:
            expr = adata[:, category].X
            if sp.issparse(expr):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()

            sc = ax.scatter(
                x_plot,
                y_plot,
                c=expr,
                cmap=cmap or "viridis",
                s=point_size,
                alpha=spot_alpha,
                rasterized=True,
                vmin=gene_vmin,
                vmax=gene_vmax,
            )

            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(category)

        # ======================================================
        # OBS COLUMN
        # ======================================================
        else:
            values = adata.obs[category].to_numpy()

            # ---------- CONTINUOUS OBS ----------
            if pd.api.types.is_numeric_dtype(values):
                sc = ax.scatter(
                    x_plot,
                    y_plot,
                    c=values,
                    cmap=cmap or "viridis",
                    s=point_size,
                    alpha=spot_alpha,
                    rasterized=True,
                    vmin=gene_vmin,
                    vmax=gene_vmax,
                )

                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(category)

            # ---------- CATEGORICAL OBS ----------
            else:
                obs_cat = adata.obs[category].astype("category")

                cats = list(obs_cat.cat.categories)
                if groups is not None:
                    cats = [c for c in cats if c in groups]

                if len(cats) == 0:
                    raise ValueError("No categories left to plot after applying `groups`.")

                n_cat = len(cats)

                colors = (
                    [cmap[c] for c in cats]
                    if isinstance(cmap, dict)
                    else plt.get_cmap("tab20")(np.linspace(0, 1, n_cat))
                )

                cmap_sel = ListedColormap(colors)
                norm = BoundaryNorm(np.arange(n_cat + 1) - 0.5, ncolors=n_cat)

                mask_sel = obs_cat.isin(cats)
                codes = obs_cat.map({cat: i for i, cat in enumerate(cats)})

                ax.scatter(
                    x_plot[mask_sel],
                    y_plot[mask_sel],
                    c=codes[mask_sel],
                    cmap=cmap_sel,
                    norm=norm,
                    s=point_size,
                    alpha=spot_alpha,
                    rasterized=True,
                )

                scatter_handles = [
                    patches.Patch(color=colors[i], label=cats[i])
                    for i in range(n_cat)
                ]



    # ===================== ROI overlay =====================
    if roi_names_to_plot is not None and "ROI_names" in adata.uns:

        # ---- Normalize to list if single string ----
        if isinstance(roi_names_to_plot, str):
            roi_names_to_plot = [roi_names_to_plot]
            
        if roi_colors is None:
            cmap_roi = plt.get_cmap("tab20")
            roi_colors = {name: cmap_roi(i % cmap_roi.N) for i, name in enumerate(roi_names_to_plot)}
        elif not isinstance(roi_colors, dict):
            roi_colors = dict(zip(roi_names_to_plot, roi_colors))

        for roi_name in roi_names_to_plot:
            if roi_name not in adata.uns["ROI_names"]:
                continue
            color = roi_colors[roi_name]
            for roi_key in adata.uns["ROI_names"][roi_name]:
                roi = adata.uns["ROI_info"][roi_key]["rois"]

                if subset:
                    roi_coords_all = np.vstack([np.array(s["coords"]) for s in roi["shapes"]])
                    roi_xmin, roi_xmax = roi_coords_all[:,0].min(), roi_coords_all[:,0].max()
                    roi_ymin, roi_ymax = roi_coords_all[:,1].min(), roi_coords_all[:,1].max()
                    if roi_xmax < col_start_um / microns_per_pixel or roi_xmin > col_end_um / microns_per_pixel \
                       or roi_ymax < row_start_um / microns_per_pixel or roi_ymin > row_end_um / microns_per_pixel:
                        continue

                for shape in roi["shapes"]:
                    coords_array = np.array(shape["coords"])
                    roi_x = coords_array[:,0]/factor
                    roi_y = coords_array[:,1]/factor
                    if subset:
                        roi_x -= col_start_um / microns_per_pixel / factor
                        roi_y -= row_start_um / microns_per_pixel / factor
                    ax.plot(roi_x, roi_y, color=color, linewidth=roi_linewidth, alpha=0.9, zorder=15)

            roi_handles.append(patches.Patch(facecolor=color, edgecolor=color, label=roi_name))

    # ---------------- Add legends ----------------
    if scatter_handles and roi_handles:
        # Both legends
        scatter_legend = ax.legend(
            handles=scatter_handles,
            title=category,
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        ax.add_artist(scatter_legend)

        roi_legend = ax.legend(
            handles=roi_handles,
            title="ROIs",
            bbox_to_anchor=(1.45, 1),  # shift to the right of category legend
            loc="upper left"
        )
    elif scatter_handles:
        # Only category legend
        ax.legend(
            handles=scatter_handles,
            title=category,
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
    elif roi_handles:
        # Only ROI legend
        ax.legend(
            handles=roi_handles,
            title="ROIs",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

    
    # ---------------- Scale bar ----------------
    if plot_scalebar:
        # If user didn't provide, choose automatically
        if scalebar_um is None:
            # Determine the width in microns of the plotted image
            if subset:
                # width = x range of subset in microns
                img_width_um = col_end_um - col_start_um
            else:
                # width = full image in microns
                img_width_um = img_plot.shape[1] * microns_per_pixel * factor

            # Choose a nice round number ~1/5 of image width
            nice_numbers = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
            scalebar_um = nice_numbers[(np.abs(nice_numbers - img_width_um / 5)).argmin()]

        # Convert to pixels in the plotted image
        bar_length_px = scalebar_um / microns_per_pixel / factor
        bar_length_px = int(round(bar_length_px))

        img_height, img_width = img_plot.shape[:2]
        bar_height = img_height * 0.01  # 1% of image height

        # Location
        if scalebar_loc == "bottom_right":
            bar_x = img_width * 0.95 - bar_length_px
        elif scalebar_loc == "bottom_left":
            bar_x = img_width * 0.05
        else:
            raise ValueError("`scalebar_loc` must be 'bottom_right' or 'bottom_left'")
        bar_y = img_height * 0.05  # 5% from bottom

        rect = patches.Rectangle(
            (bar_x, bar_y),
            width=bar_length_px,
            height=bar_height,
            color=scalebar_color,
            edgecolor=scalebar_color
        )
        ax.add_patch(rect)

        # Text label above the bar
        ax.text(
            bar_x + bar_length_px / 2,
            bar_y + bar_height * 1.5,  # just above the bar
            f"{int(scalebar_um)} µm",
            color=scalebar_color,
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold",
        )

        
    # ---------------- Title ----------------
    if title is not False:
        if title is None:
            if category is not None:
                title_str = f"Plotting {category}"
            else:
                title_str = ""
        else:
            title_str = str(title)

        if title_str:
            ax.set_title(title_str, fontsize=14) #weight="bold")
    
        
    # ---------------- Rectangle ----------------
    if plot_rectangle and not subset:
        rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)

    # ---------------- Axes ----------------
    if not show_axes:
        ax.axis("off")
    else:
        # ---- Determine micron extents ----
        if subset:
            x_min_um = col_start_um
            x_max_um = col_end_um
            y_min_um = row_start_um
            y_max_um = row_end_um
        else:
            x_min_um = 0
            x_max_um = img_plot.shape[1] * microns_per_pixel * factor
            y_min_um = 0
            y_max_um = img_plot.shape[0] * microns_per_pixel * factor

        img_width_um = x_max_um - x_min_um

        # ---- Choose nice tick spacing (~1/5 width) ----
        nice_numbers = np.array([
            50, 100, 200, 500,
            1000, 2000, 5000,
            10000, 20000, 50000
        ])

        tick_um = nice_numbers[
            (np.abs(nice_numbers - img_width_um / 5)).argmin()
        ]

        # ---- Convert micron spacing to plot pixel spacing ----
        tick_px = tick_um / (microns_per_pixel * factor)

        # ---- Align first tick to nearest multiple in micron space ----
        first_tick_um = np.ceil(x_min_um / tick_um) * tick_um
        first_tick_px = (first_tick_um - x_min_um) / (microns_per_pixel * factor)

        # ---- Set locators with proper offset ----
        ax.xaxis.set_major_locator(
            mticker.FixedLocator(
                np.arange(first_tick_px,
                          img_plot.shape[1],
                          tick_px)
            )
        )

        ax.yaxis.set_major_locator(
            mticker.FixedLocator(
                np.arange(first_tick_px,
                          img_plot.shape[0],
                          tick_px)
            )
        )

        # ---- Formatter (convert plotted px → microns) ----
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, pos: int(round(
                    x * microns_per_pixel * factor + x_min_um
                ))
            )
        )

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda y, pos: int(round(
                    y * microns_per_pixel * factor + y_min_um
                ))
            )
        )

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")

    if save is not None:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    plt.show()
    if close:
        plt.close(fig)



def add_multi_radius_density(
    adata,
    radii_um=(30, 60, 120),
    spatial_key="spatial",
    density_prefix="cell_density",
):
    """
    Compute per-cell total density at multiple spatial scales (fixed-radius).

    Adds:
        {density_prefix}_r{radius}_um2
        {density_prefix}_r{radius}_mm2
    """
    coords = adata.obsm[spatial_key]
    tree = KDTree(coords)

    for r in radii_um:
        counts = tree.query_radius(coords, r=r, count_only=True)
        area_um2 = np.pi * r**2

        density_um2 = counts / area_um2
        density_mm2 = density_um2 * 1e6

        adata.obs[f"{density_prefix}_r{r}_um2"] = density_um2
        adata.obs[f"{density_prefix}_r{r}_mm2"] = density_mm2


def add_xenium_image_density(
    adata,
    library_id,
    spatial_key="spatial",
    density_key="cell_density_image",
    bin_size_px=25,
    sigma_bins=2.0,
):
    """
    Compute image-style binned + Gaussian-smoothed cell density for visualization.
    """
    coords = adata.obsm[spatial_key]
    pixel_size = adata.uns[spatial_key][library_id]["scalefactors"]["pixel_size"]

    bin_size_um = bin_size_px * pixel_size

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)

    nx = int((xmax - xmin) / bin_size_um) + 1
    ny = int((ymax - ymin) / bin_size_um) + 1

    H, xedges, yedges = np.histogram2d(
        coords[:, 0],
        coords[:, 1],
        bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]],
    )

    H_smooth = gaussian_filter(H, sigma=sigma_bins)

    ix = np.clip(np.searchsorted(xedges, coords[:, 0]) - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(yedges, coords[:, 1]) - 1, 0, ny - 1)

    adata.obs[density_key] = H_smooth[ix, iy]


def add_multiradius_celltype_density(
    adata,
    celltype_key,
    target_celltypes,
    radii_um=(30, 60, 120),
    spatial_key="spatial",
    density_prefix="ct_density",
    restrict_to_celltype=None,
    edge_correction=True,
    normalize_global=True,
):
    """
    Multi-radius per-cell-type density with vectorization, edge correction,
    and global abundance normalization.

    Adds:
        {density_prefix}_{celltype}_r{radius}
        {density_prefix}_{celltype}_r{radius}_norm (optional)
    """
    coords = adata.obsm[spatial_key]
    celltypes = adata.obs[celltype_key].values
    n_cells = coords.shape[0]

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)

    tree = KDTree(coords)

    tissue_area = (xmax - xmin) * (ymax - ymin)
    global_density = {
        ct: np.sum(celltypes == ct) / tissue_area
        for ct in target_celltypes
    }

    for r in radii_um:
        neighbors = tree.query_radius(coords, r=r)
        base_area = np.pi * r**2
        area = np.full(n_cells, base_area)

        if edge_correction:
            dx_left = coords[:, 0] - xmin
            dx_right = xmax - coords[:, 0]
            dy_bottom = coords[:, 1] - ymin
            dy_top = ymax - coords[:, 1]

            frac_x = np.minimum(1.0, (dx_left + dx_right) / (2 * r))
            frac_y = np.minimum(1.0, (dy_bottom + dy_top) / (2 * r))
            area = base_area * frac_x * frac_y
            area = np.maximum(area, base_area * 0.25)

        for ct in target_celltypes:
            ct_mask = (celltypes == ct).astype(int)

            counts = np.fromiter(
                (ct_mask[nbrs].sum() for nbrs in neighbors),
                dtype=float,
                count=n_cells,
            )

            density = counts / area

            if restrict_to_celltype is not None:
                density[celltypes != restrict_to_celltype] = np.nan

            col = f"{density_prefix}_{ct}_r{r}"
            adata.obs[col] = density

            if normalize_global:
                adata.obs[f"{col}_norm"] = density / global_density[ct]

                
            

def complete_roi_shapes_in_place(
    adata,
    rois_to_update,
    use_convex_hull=False,
    smooth=False,
    sigma=2.0
):
    """
    Completes ROI shapes in-place in adata.uns["ROI_info"].
    Coordinates remain in **image pixel space** for proper alignment with images.
    """
    if isinstance(rois_to_update, str):
        rois_to_update = [rois_to_update]

    for roi_name in rois_to_update:
        all_coords = []

        # Gather all coordinates
        for roi_key in adata.uns["ROI_names"][roi_name]:
            roi_entry = adata.uns["ROI_info"][roi_key]["rois"]
            for shape in roi_entry["shapes"]:
                coords = np.array(shape["coords"], dtype=float)
                all_coords.append(coords)

        all_coords = np.vstack(all_coords)

        # Build polygon
        if use_convex_hull:
            polygon = MultiPoint(all_coords).convex_hull
            new_coords = np.array(polygon.exterior.coords)
        else:
            polygon = Polygon(all_coords)
            new_coords = np.array(polygon.exterior.coords)
            if not np.allclose(new_coords[0], new_coords[-1]):
                new_coords = np.vstack([new_coords, new_coords[0]])

            if smooth:
                x, y = new_coords[:-1, 0], new_coords[:-1, 1]
                x_smooth = gaussian_filter1d(x, sigma=sigma)
                y_smooth = gaussian_filter1d(y, sigma=sigma)
                new_coords = np.column_stack([x_smooth, y_smooth])
                new_coords = np.vstack([new_coords, new_coords[0]])
                polygon = Polygon(new_coords)

        # Update in-place
        for roi_key in adata.uns["ROI_names"][roi_name]:
            roi_entry = adata.uns["ROI_info"][roi_key]["rois"]
            for shape in roi_entry["shapes"]:
                shape["coords"] = new_coords.copy()

        print(f"Completed ROI '{roi_name}' in place. Convex hull: {use_convex_hull}, smooth: {smooth}, points: {len(new_coords)}")

        
        
def create_ROI_mask(
    adata,
    library_id: str,
    name: str,
    ROI: str,
    force_recalculate: bool = False,
):
    
    # Convert cell coordinates from microns to pixels
    microns_per_pixel = adata.uns["spatial"][library_id]["scalefactors"]["pixel_size"]
    coords_micron = adata.obsm["spatial"]
    coords_px = coords_micron / microns_per_pixel
    cell_points = [Point(c) for c in coords_px]
    cell_barcodes = adata.obs_names.to_numpy()
    
    roi_polygons = {}
    roi_key = adata.uns['ROI_names'][ROI][0]
    roi_polygons[ROI] = Polygon(adata.uns['ROI_info'][roi_key]['rois']['shapes'][0]['coords'])
    
    poly = roi_polygons[ROI]

    inside_mask = np.array([poly.contains(p) for p in cell_points])
    barcodes_filtered = cell_barcodes[inside_mask]
    coords_filtered = coords_px[inside_mask]
    
    adata.uns[name] = {
        "barcodes": barcodes_filtered,
        "coords_filtered": coords_filtered,
         "roi_polygons" : roi_polygons
    }

    print(f"ROI mask '{name}' computed. {inside_mask.sum()} cells inside {ROI}.")
        
        
        

def create_vessel_mask(
    adata,
    library_id: str,
    name: str,
    ROIs: list,
    force_recalculate: bool = False,
):
    """
    Create vessel mask using **pixel coordinates** aligned with the image.
    Converts cell micron coordinates to pixel space to match ROIs.
    """
    uns_key = f"{name}_vessel_mask_information"
    expected_keys = {"barcodes", "coords_filtered", "roi_polygons"}

    if uns_key in adata.uns and expected_keys.issubset(adata.uns[uns_key].keys()):
        if not force_recalculate:
            print(f"Vessel mask '{uns_key}' exists. Skipping.")
            return
        else:
            print(f"Overwriting vessel mask '{uns_key}' due to force_recalculate=True.")

    # Convert cell coordinates from microns to pixels
    microns_per_pixel = adata.uns["spatial"][library_id]["scalefactors"]["pixel_size"]
    coords_micron = adata.obsm["spatial"]
    coords_px = coords_micron / microns_per_pixel
    cell_points = [Point(c) for c in coords_px]
    cell_barcodes = adata.obs_names.to_numpy()

    # Build ROI polygons in pixel space
    roi_polygons = {}
    for roi_name in ROIs:
        roi_coords_all = []
        for roi_key in adata.uns['ROI_names'][roi_name]:
            roi_info = adata.uns['ROI_info'][roi_key]['rois']
            for shape in roi_info['shapes']:
                roi_coords_all.append(np.array(shape['coords'], dtype=float))
        roi_coords_all = np.vstack(roi_coords_all)
        roi_polygons[roi_name] = Polygon(roi_coords_all)

    # Filter cells inside perivascular ROI
    perivascular_poly = roi_polygons[ROIs[-1]]
    inside_mask = np.array([perivascular_poly.contains(p) for p in cell_points])
    barcodes_filtered = cell_barcodes[inside_mask]
    coords_filtered = coords_px[inside_mask]

    adata.uns[uns_key] = {
        "barcodes": barcodes_filtered,
        "coords_filtered": coords_filtered,
        "roi_polygons": roi_polygons,
    }

    print(f"Vessel mask '{uns_key}' computed. {inside_mask.sum()} cells inside {ROIs[-1]}.")


def compute_distance_vessel_mask(
    adata,
    vessel_mask: str,
):
    """
    Compute distances in pixel coordinates aligned with image and ROIs.
    Normalized distances 0–1 match plotting behavior.
    """
    uns_key = f"{vessel_mask}_vessel_mask_information"
    if uns_key not in adata.uns:
        raise ValueError(f"Vessel mask '{uns_key}' not found.")

    coords_filtered = adata.uns[uns_key]['coords_filtered']
    roi_polygons = adata.uns[uns_key]['roi_polygons']
    roi_order = list(roi_polygons.keys())

    lumen_poly = roi_polygons[roi_order[0]]
    perivascular_poly = roi_polygons[roi_order[-1]]

    points = [Point(c) for c in coords_filtered]
    d_lumen_px = np.array([p.distance(lumen_poly.exterior) for p in points])
    d_outer_px = np.array([p.distance(perivascular_poly.exterior) for p in points])

    # Normalized distance
    d_norm = d_lumen_px / (d_lumen_px + d_outer_px)

    adata.uns[uns_key]['d_lumen_px'] = d_lumen_px
    adata.uns[uns_key]['d_outer_px'] = d_outer_px
    adata.uns[uns_key]['d_norm'] = d_norm

    print(f"Distance computation for '{vessel_mask}' complete.")
    print(f"Absolute distance to lumen: {d_lumen_px.min():.1f}–{d_lumen_px.max():.1f} px")
    print(f"Normalized distance range: {d_norm.min():.3f}–{d_norm.max():.3f}")

        

def plot_gene_across_vessel_mask(
    adata,
    vessel_mask,
    genes,
    celltypes=None,              # list of celltypes to subset; None = all
    category="cell_type",        # column in adata.obs
    distance_type="normalized",  # "normalized" or "absolute"
    n_bins=20,
    ci=False,                    # plot bootstrap confidence interval
    n_boot=500,                  # number of bootstrap iterations
    save=False,
):
    """
    Plot gene expression vs distance across a vessel mask.

    Parameters
    ----------
    adata : AnnData
    vessel_mask : str
        Name used in adata.uns (expects {vessel_mask}_vessel_mask_information).
    genes : list
        Genes to plot.
    celltypes : list or None
        If provided, subset to these cell types.
    category : str
        Column in adata.obs containing cell type labels.
    distance_type : str
        "normalized" or "absolute"
    n_bins : int
        Number of bins along distance
    ci : bool
        If True, compute and plot bootstrap confidence intervals (95% CI)
    n_boot : int
        Number of bootstrap iterations
    save : bool
    """

    uns_key = f"{vessel_mask}_vessel_mask_information"
    if uns_key not in adata.uns:
        raise ValueError(f"{uns_key} not found in adata.uns")

    data = adata.uns[uns_key]
    barcodes = np.array(data["barcodes"])
    d_norm_all = np.array(data["d_norm"])
    d_lumen_all = np.array(data["d_lumen_px"])  # convert externally if needed

    # 1️⃣ Subset adata to vessel cells
    adata_vessel = adata[barcodes].copy()
    if not np.array_equal(adata_vessel.obs_names.to_numpy(), barcodes):
        raise ValueError("Barcode order mismatch between adata and vessel mask.")

    # 2️⃣ Extract cell type labels
    if category not in adata_vessel.obs:
        raise ValueError(f"{category} not found in adata.obs")
    cell_type_labels = adata_vessel.obs[category].to_numpy()

    # 3️⃣ Optional subsetting
    mask = np.ones(adata_vessel.n_obs, dtype=bool)
    if celltypes is not None:
        if isinstance(celltypes, str):
            celltypes = [celltypes]
        mask = np.isin(cell_type_labels, celltypes)

    d_norm = d_norm_all[mask]
    d_lumen = d_lumen_all[mask]
    cell_type_labels = cell_type_labels[mask]
    adata_vessel = adata_vessel[mask]

    # 4️⃣ Extract expression
    expr_dict = {}
    for g in genes:
        if g not in adata_vessel.var_names:
            raise ValueError(f"{g} not found in adata.var_names")
        expr = adata_vessel[:, g].X
        if hasattr(expr, "toarray"):
            expr = expr.toarray().ravel()
        else:
            expr = np.asarray(expr).ravel()
        expr_dict[g] = expr

    # 5️⃣ Build dataframe
    df = pd.DataFrame({
        "d_norm": d_norm,
        "d_lumen": d_lumen,
        category: cell_type_labels
    })
    for g in genes:
        df[g] = expr_dict[g]

    print(f"Dataframe constructed with {len(df)} cells.")

    # 6️⃣ Choose distance type
    if distance_type.lower() == "normalized":
        bins = np.linspace(0, 1, n_bins + 1)
        df["bin"] = pd.cut(df["d_norm"], bins=bins, labels=False)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        xlabel = "Normalized distance (lumen → perivascular edge)"
        xlim = (0, 1)
    elif distance_type.lower() in ["absolute", "px"]:
        max_val = np.percentile(d_lumen, 95)
        bins = np.linspace(0, max_val, n_bins + 1)
        df["bin"] = pd.cut(df["d_lumen"], bins=bins, labels=False)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        xlabel = "Distance from lumen (px)"
        xlim = (0, max_val)
    else:
        raise ValueError("distance_type must be 'normalized' or 'absolute'")

    # 7️⃣ Plot
    plt.figure(figsize=(7, 5))

    for g in genes:
        mean_expr = df.groupby("bin")[g].mean()
        plt.plot(bin_centers, mean_expr, marker="o", label=g)

        if ci:
            # Bootstrap CI
            curves = []
            for _ in range(n_boot):
                sample = df.sample(frac=1, replace=True)
                means = sample.groupby("bin")[g].mean().reindex(range(n_bins))
                curves.append(means.values)
            curves = np.vstack(curves)
            lo = np.nanpercentile(curves, 2.5, axis=0)
            hi = np.nanpercentile(curves, 97.5, axis=0)
            plt.fill_between(bin_centers, lo, hi, alpha=0.25)

    plt.xlabel(xlabel)
    plt.ylabel("Mean expression per cell")
    if celltypes:
        plt.title(f"{vessel_mask}: Gene expression vs {distance_type} distance for selected celltypes")
    else:
        plt.title(f"{vessel_mask}: Gene expression vs {distance_type} distance")

    plt.legend()
    plt.grid(True)
    plt.xlim(xlim)

    if save:
        plt.savefig(f"{vessel_mask}_gene_expression_{distance_type}.pdf", dpi=300)

    plt.show()
    plt.close()

    


def plot_celltype_composition_across_vessel(
    adata,
    vessel_mask,
    celltypes_of_interest=None,
    category="cell_type",
    distance_type="normalized",
    n_bins=20,
    sigma=1,
    colors=None,
    save=False,
):
    """
    Plot cell-type composition vs distance across vessel.

    distance_type:
        "normalized" → uses d_norm (0–1)
        "absolute"   → uses d_lumen_px
    """

    uns_key = f"{vessel_mask}_vessel_mask_information"
    if uns_key not in adata.uns:
        raise ValueError(f"{uns_key} not found in adata.uns")

    data = adata.uns[uns_key]
    barcodes = np.array(data["barcodes"])
    d_norm_all = np.array(data["d_norm"])
    d_lumen_all = np.array(data["d_lumen_px"])

    # Subset to vessel cells
    adata_vessel = adata[barcodes].copy()

    if category not in adata_vessel.obs:
        raise ValueError(f"{category} not found in adata.obs")

    cell_labels_all = adata_vessel.obs[category].to_numpy()

    # -------------------------------------------------
    # Determine celltypes
    # -------------------------------------------------
    using_all = celltypes_of_interest is None

    if using_all:
        celltypes = np.unique(cell_labels_all)
        mask = np.ones(len(cell_labels_all), dtype=bool)
        print("Using ALL celltypes present in vessel mask.")
    else:
        if isinstance(celltypes_of_interest, str):
            celltypes_of_interest = [celltypes_of_interest]

        mask = np.isin(cell_labels_all, celltypes_of_interest)
        celltypes = np.array(celltypes_of_interest)

        print(f"Using {mask.sum()} cells across selected celltypes.")

    # Apply mask
    cell_labels = cell_labels_all[mask]
    d_norm = d_norm_all[mask]
    d_lumen = d_lumen_all[mask]

    if len(cell_labels) == 0:
        raise ValueError("No cells found after subsetting.")

    # -------------------------------------------------
    # Choose distance type
    # -------------------------------------------------
    if distance_type.lower() == "normalized":
        values = d_norm
        bins = np.linspace(0, 1, n_bins + 1)
        xlabel = "Normalized distance (0–1)"
        xlim = (0, 1)

    elif distance_type.lower() in ["absolute", "px"]:
        if len(d_lumen) == 0:
            raise ValueError("No distance values available.")
        max_val = np.percentile(d_lumen, 95)
        values = d_lumen
        bins = np.linspace(0, max_val, n_bins + 1)
        xlabel = "Distance from lumen (px)"
        xlim = (0, max_val)

    else:
        raise ValueError("distance_type must be 'normalized' or 'absolute'")

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # -------------------------------------------------
    # Histogram
    # -------------------------------------------------
    hist = []

    for ct in celltypes:
        ct_mask = cell_labels == ct
        counts, _ = np.histogram(values[ct_mask], bins=bins)
        hist.append(counts)

    hist = np.array(hist).T  # shape: bins × celltypes

    # Convert to percentages per bin
    hist_sum = hist.sum(axis=1, keepdims=True)
    hist_pct = np.divide(
        hist,
        hist_sum,
        out=np.zeros_like(hist, dtype=float),
        where=hist_sum != 0
    ) * 100

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    print(f"Plotting CELL-TYPE COMPOSITION vs {distance_type} distance...")

    plt.figure(figsize=(8, 5))

    for i, ct in enumerate(celltypes):
        color = None if colors is None else colors.get(ct, None)

        plt.plot(
            bin_centers,
            gaussian_filter1d(hist_pct[:, i], sigma=sigma),
            linewidth=2,
            color=color,
            label=ct
        )

    plt.xlabel(xlabel)
    plt.ylabel("Cell-type percentage (%)")

    if using_all:
        plt.title(f"Cell-type composition vs {distance_type} distance")
    else:
        plt.title(
            f"Cell-type composition vs {distance_type} distance\n"
            f"(selected celltypes)"
        )

    plt.xlim(xlim)
    plt.grid(True)

    plt.legend(
        title=category,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False
    )

    plt.tight_layout()

    if save:
        plt.savefig(
            f"{vessel_mask}_celltype_composition_{distance_type}_distance.pdf",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()



def plot_area_normalized_abundance(
    adata,
    vessel_mask,
    celltypes_of_interest=None,
    category="cell_type",
    n_bins=20,
    sigma=1,
    n_mc=200_000,
    colors=None,
    save=False,
):
    """
    Plot area-normalized cell abundance vs normalized distance.

    - First polygon = lumen
    - Last polygon = perivascular boundary
    - Monte Carlo sampling vectorized for speed
    """

    uns_key = f"{vessel_mask}_vessel_mask_information"

    if uns_key not in adata.uns:
        raise ValueError(f"{uns_key} not found in adata.uns")

    data = adata.uns[uns_key]

    # -----------------------------------------
    # Geometry
    # -----------------------------------------
    roi_polygons = list(data["roi_polygons"].values())

    if len(roi_polygons) < 2:
        raise ValueError("roi_polygons must contain at least two polygons")

    lumen_poly = roi_polygons[0]
    outer_poly = roi_polygons[-1]
    perivascular_poly = outer_poly
    lumen_boundary = lumen_poly.boundary
    outer_boundary = outer_poly.boundary

    # -----------------------------------------
    # Cells
    # -----------------------------------------
    barcodes = np.array(data["barcodes"])
    d_norm_all = np.array(data["d_norm"])
    adata_vessel = adata[barcodes].copy()
    cell_labels_all = adata_vessel.obs[category].to_numpy()

    # Subset celltypes
    using_all = celltypes_of_interest is None

    if using_all:
        celltypes = np.unique(cell_labels_all)
        mask = np.ones(len(cell_labels_all), dtype=bool)
        print("Using ALL celltypes present in vessel mask.")
    else:
        if isinstance(celltypes_of_interest, str):
            celltypes_of_interest = [celltypes_of_interest]
        mask = np.isin(cell_labels_all, celltypes_of_interest)
        celltypes = np.array(celltypes_of_interest)
        print(f"Using {mask.sum()} cells across selected celltypes.")

    d_norm = d_norm_all[mask]
    cell_labels = cell_labels_all[mask]

    # -----------------------------------------
    # Binning
    # -----------------------------------------
    bins_norm = np.linspace(0, 1, n_bins + 1)
    bin_centers_norm = 0.5 * (bins_norm[:-1] + bins_norm[1:])

    # -----------------------------------------
    # Vectorized Monte Carlo area sampling
    # -----------------------------------------
    print("Performing vectorized Monte Carlo area correction...")

    minx, miny, maxx, maxy = perivascular_poly.bounds

    # Generate random points in bounding box
    xs = np.random.uniform(minx, maxx, n_mc)
    ys = np.random.uniform(miny, maxy, n_mc)

    # Vectorized check which points are inside polygon
    mask_inside = contains(perivascular_poly, xs, ys)

    # Keep only points inside polygon
    xs_in = xs[mask_inside]
    ys_in = ys[mask_inside]

    # Compute normalized distance for each point
    # distance from lumen / (distance from lumen + distance to outer boundary)
    mc_points = np.column_stack([xs_in, ys_in])
    # distances from shapely
    from shapely.geometry import Point as Point

    # Vectorized distance computation using list comprehension
    mc_d_norm = np.array([
        lumen_boundary.distance(Point(x, y)) /
        (lumen_boundary.distance(Point(x, y)) + Point(x, y).distance(outer_boundary))
        for x, y in mc_points
    ])

    # Histogram & area fraction
    area_counts, _ = np.histogram(mc_d_norm, bins=bins_norm)
    area_fraction = area_counts / area_counts.sum()

    print("Area correction complete.")
    print("Plotting AREA-NORMALIZED cell density...")

    # -----------------------------------------
    # Plotting
    # -----------------------------------------
    plt.figure(figsize=(8, 5))

    for ct in celltypes:
        ct_mask = cell_labels == ct
        obs_counts, _ = np.histogram(d_norm[ct_mask], bins=bins_norm)

        # Area-normalized density
        density = np.divide(
            obs_counts,
            area_fraction,
            out=np.zeros_like(obs_counts, dtype=float),
            where=area_fraction != 0
        )

        # Relative scaling for shape comparison
        if np.nanmax(density) > 0:
            density = density / np.nanmax(density)

        color = None if colors is None else colors.get(ct, None)

        plt.plot(
            bin_centers_norm,
            gaussian_filter1d(density, sigma=sigma),
            linewidth=2,
            color=color,
            label=ct
        )

    plt.xlabel("Normalized distance (0–1)")
    plt.ylabel("Relative cell density (area-corrected)")
    plt.xlim(0, 1)
    plt.grid(True)

    plt.legend(
        title=category,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False
    )

    plt.tight_layout()

    if save:
        plt.savefig(
            f"{vessel_mask}_cell_density_area_normalized.pdf",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    plt.close()

    


def plot_celltype_comparison_area_corrected(
    adata,
    vessel_mask,
    celltypes_of_interest=None,
    category="cell_type",
    n_bins=20,
    sigma=1,
    n_mc=200_000,
    colors=None,
    save=False,
):
    """
    Compare radial distribution of celltypes across a vessel, area-corrected.

    This accounts for increasing annular area at larger radii using Monte Carlo
    sampling and shows **fraction of each celltype per bin** corrected by area.
    """

    uns_key = f"{vessel_mask}_vessel_mask_information"
    if uns_key not in adata.uns:
        raise ValueError(f"{uns_key} not found in adata.uns")

    data = adata.uns[uns_key]

    # ------------------------------
    # Geometry: first = lumen, last = perivascular
    # ------------------------------
    roi_polygons = list(data["roi_polygons"].values())
    if len(roi_polygons) < 2:
        raise ValueError("roi_polygons must contain at least two polygons")

    lumen_poly = roi_polygons[0]
    outer_poly = roi_polygons[-1]
    lumen_boundary = lumen_poly.boundary
    outer_boundary = outer_poly.boundary
    perivascular_poly = outer_poly

    # ------------------------------
    # Cells
    # ------------------------------
    barcodes = np.array(data["barcodes"])
    d_norm_all = np.array(data["d_norm"])
    adata_vessel = adata[barcodes].copy()
    cell_labels_all = adata_vessel.obs[category].to_numpy()

    # Subset celltypes
    if celltypes_of_interest is None:
        celltypes_of_interest = np.unique(cell_labels_all)
    elif isinstance(celltypes_of_interest, str):
        celltypes_of_interest = [celltypes_of_interest]

    mask = np.isin(cell_labels_all, celltypes_of_interest)
    d_norm = d_norm_all[mask]
    cell_labels = cell_labels_all[mask]

    print(f"Comparing {len(celltypes_of_interest)} celltypes across {mask.sum()} cells")

    # ------------------------------
    # Binning
    # ------------------------------
    bins_norm = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins_norm[:-1] + bins_norm[1:])

    # ------------------------------
    # Monte Carlo area sampling (vectorized)
    # ------------------------------
    print("Performing Monte Carlo area correction...")

    minx, miny, maxx, maxy = perivascular_poly.bounds

    xs = np.random.uniform(minx, maxx, n_mc)
    ys = np.random.uniform(miny, maxy, n_mc)
    mask_inside = contains(perivascular_poly, xs, ys)
    xs_in = xs[mask_inside]
    ys_in = ys[mask_inside]
    mc_points = np.column_stack([xs_in, ys_in])

    # Compute normalized distance for MC points
    mc_d_norm = np.array([
        lumen_boundary.distance(Point(x, y)) /
        (lumen_boundary.distance(Point(x, y)) + Point(x, y).distance(outer_boundary))
        for x, y in mc_points
    ])

    area_counts, _ = np.histogram(mc_d_norm, bins=bins_norm)
    area_fraction = area_counts / area_counts.sum()

    # ------------------------------
    # Histogram per celltype
    # ------------------------------
    hist = np.zeros((n_bins, len(celltypes_of_interest)))

    for i, ct in enumerate(celltypes_of_interest):
        ct_mask = cell_labels == ct
        obs_counts, _ = np.histogram(d_norm[ct_mask], bins=bins_norm)

        # Area-corrected counts
        density = np.divide(
            obs_counts,
            area_fraction,
            out=np.zeros_like(obs_counts, dtype=float),
            where=area_fraction != 0
        )

        hist[:, i] = density

    # ------------------------------
    # Normalize per bin to get fractions
    # ------------------------------
    hist_sum = hist.sum(axis=1, keepdims=True)
    fraction_per_bin = np.divide(hist, hist_sum, out=np.zeros_like(hist), where=hist_sum != 0)

    # ------------------------------
    # Smooth and plot
    # ------------------------------
    plt.figure(figsize=(8, 5))

    for i, ct in enumerate(celltypes_of_interest):
        y_smooth = gaussian_filter1d(fraction_per_bin[:, i], sigma=sigma)
        color = None if colors is None else colors.get(ct, None)

        plt.plot(
            bin_centers,
            y_smooth,
            linewidth=2,
            color=color,
            label=ct
        )

    plt.xlabel("Normalized distance (0–1)")
    plt.ylabel("Area-corrected fraction per bin")
    plt.title("Area-corrected celltype comparison across vessel")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.legend(
        title=category,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False
    )

    plt.tight_layout()

    if save:
        plt.savefig(f"{vessel_mask}_celltype_comparison_area_corrected.pdf", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

    

def plot_gene_across_multiple_vessels(
    adata,
    vessel_masks,
    genes,
    celltypes=None,
    category="cell_type",
    n_bins=20,
    ci=False,
    save=False,
):
    """
    Plot gene expression averaged across multiple vessel masks (normalized distance only).

    Parameters
    ----------
    adata : AnnData
    vessel_masks : list of str
        Vessel mask names to average across
    genes : list
        Genes to plot
    celltypes : list or None
        Subset to these celltypes if provided
    category : str
        Column in adata.obs with cell type labels
    n_bins : int
        Number of distance bins (0-1 normalized)
    ci : bool
        Whether to show confidence intervals across vessels
    save : bool
    """

    # Prepare bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Store mean expressions per vessel
    gene_means_per_vessel = {g: [] for g in genes}

    for vessel_mask in vessel_masks:
        uns_key = f"{vessel_mask}_vessel_mask_information"
        if uns_key not in adata.uns:
            raise ValueError(f"{uns_key} not found in adata.uns")
        data = adata.uns[uns_key]

        barcodes = np.array(data["barcodes"])
        d_norm_all = np.array(data["d_norm"])

        adata_vessel = adata[barcodes].copy()
        cell_type_labels = adata_vessel.obs[category].to_numpy()

        # Subset to celltypes if requested
        mask = np.ones(len(barcodes), dtype=bool)
        if celltypes is not None:
            if isinstance(celltypes, str):
                celltypes = [celltypes]
            mask = np.isin(cell_type_labels, celltypes)

        d_norm = d_norm_all[mask]
        cell_labels = cell_type_labels[mask]
        adata_vessel = adata_vessel[mask]

        # Extract expression
        expr_dict = {}
        for g in genes:
            expr = adata_vessel[:, g].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()
            expr_dict[g] = expr

        # Bin expression per vessel
        df_vessel = pd.DataFrame({"d_norm": d_norm})
        for g in genes:
            df_vessel[g] = expr_dict[g]

        for g in genes:
            mean_expr = df_vessel.groupby(pd.cut(df_vessel["d_norm"], bins=bins, labels=False))[g].mean()
            # Ensure all bins present
            mean_expr = mean_expr.reindex(range(n_bins), fill_value=np.nan)
            gene_means_per_vessel[g].append(mean_expr.values)

    # -------------------------------------------------
    # Average across vessels
    # -------------------------------------------------
    plt.figure(figsize=(8, 5))

    for g in genes:
        mat = np.vstack(gene_means_per_vessel[g])  # vessels x bins
        mean_all = np.nanmean(mat, axis=0)
        plt.plot(bin_centers, mean_all, marker="o", label=g)

        if ci:
            lo = np.nanpercentile(mat, 2.5, axis=0)
            hi = np.nanpercentile(mat, 97.5, axis=0)
            plt.fill_between(bin_centers, lo, hi, alpha=0.25)

    plt.xlabel("Normalized distance (0 = lumen, 1 = perivascular edge)")
    plt.ylabel("Mean expression per cell")
    plt.title(f"Average gene expression across {len(vessel_masks)} vessels")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)

    if save:
        plt.savefig("gene_expression_avg_multiple_vessels.pdf", dpi=300)

    plt.show()
    plt.close()

    
    


def plot_gene_across_vessels_with_ci(
    adata,
    vessel_masks,
    genes,
    celltypes=None,              # list of celltypes to subset; None = all
    category="cell_type",        # column in adata.obs
    n_bins=20,
    n_boot=500,                  # bootstrap iterations per vessel
    save=False,
):
    """
    Plot gene expression averaged across multiple vessels (normalized distance only),
    with vessel-to-vessel variability and per-vessel bootstrap confidence intervals.

    Parameters
    ----------
    adata : AnnData
    vessel_masks : list of str
        Vessel mask names to average across
    genes : list
        Genes to plot
    celltypes : list or None
        Subset to these celltypes if provided
    category : str
        Column in adata.obs with cell type labels
    n_bins : int
        Number of bins along normalized distance
    n_boot : int
        Number of bootstrap iterations per vessel
    save : bool
        Whether to save the figure
    """

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Store binned values per gene and per vessel
    gene_binned_vessels = {g: [] for g in genes}

    for vessel_mask in vessel_masks:
        uns_key = f"{vessel_mask}_vessel_mask_information"
        if uns_key not in adata.uns:
            raise ValueError(f"{uns_key} not found in adata.uns")
        data = adata.uns[uns_key]

        barcodes = np.array(data["barcodes"])
        d_norm_all = np.array(data["d_norm"])

        adata_vessel = adata[barcodes].copy()
        cell_type_labels = adata_vessel.obs[category].to_numpy()

        # Optional subsetting
        mask = np.ones(len(barcodes), dtype=bool)
        if celltypes is not None:
            if isinstance(celltypes, str):
                celltypes = [celltypes]
            mask = np.isin(cell_type_labels, celltypes)

        d_norm = d_norm_all[mask]
        adata_vessel = adata_vessel[mask]

        # Extract expression per gene
        expr_dict = {}
        for g in genes:
            expr = adata_vessel[:, g].X
            if hasattr(expr, "toarray"):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()
            expr_dict[g] = expr

        # Create dataframe for binning
        df_vessel = pd.DataFrame({"d_norm": d_norm})
        for g in genes:
            df_vessel[g] = expr_dict[g]

        # Bootstrap per vessel
        for g in genes:
            boot_curves = []
            for _ in range(n_boot):
                sample = df_vessel.sample(frac=1, replace=True)
                mean_expr = sample.groupby(pd.cut(sample["d_norm"], bins=bins, labels=False))[g].mean()
                mean_expr = mean_expr.reindex(range(n_bins), fill_value=np.nan)
                boot_curves.append(mean_expr.values)
            boot_curves = np.vstack(boot_curves)
            # Store mean per bin across bootstrap for this vessel
            gene_binned_vessels[g].append(boot_curves)

    # -------------------------------------------------
    # Combine vessels
    # -------------------------------------------------
    plt.figure(figsize=(8, 5))

    for g in genes:
        # gene_binned_vessels[g] is a list of (n_boot x n_bins) arrays
        # Stack across vessels → shape: (n_vessels * n_boot, n_bins)
        all_boot = np.vstack(gene_binned_vessels[g])
        mean_all = np.nanmean(all_boot, axis=0)
        ci_lo = np.nanpercentile(all_boot, 2.5, axis=0)
        ci_hi = np.nanpercentile(all_boot, 97.5, axis=0)

        plt.plot(bin_centers, mean_all, marker="o", label=g)
        plt.fill_between(bin_centers, ci_lo, ci_hi, alpha=0.25)

    plt.xlabel("Normalized distance (0 = lumen, 1 = perivascular edge)")
    plt.ylabel("Mean expression per cell")
    plt.title(f"Average gene expression across {len(vessel_masks)} vessels with CI")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)

    if save:
        plt.savefig("gene_expression_avg_vessels_bootstrap_CI.pdf", dpi=300)

    plt.show()
    plt.close()
