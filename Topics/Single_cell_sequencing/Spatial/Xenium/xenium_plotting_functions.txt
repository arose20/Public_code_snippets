import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KDTree


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm

from skimage.color import lab2rgb

from typing import Optional, Tuple, List, Dict, Union


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
                    ax.plot(roi_x, roi_y, color=color, linewidth=2, alpha=0.9, zorder=15)

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
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

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
