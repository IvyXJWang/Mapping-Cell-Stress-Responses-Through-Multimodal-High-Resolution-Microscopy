import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
from pathlib import Path
import feature_extraction_utils as utils
from tqdm import tqdm
import constants
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import math
from matplotlib.lines import Line2D
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# project modules
import constants


def distance_between_centroids(p, q, show_intermediate=False, outlines: list = None):
    """
    Inputs
        p,q -  dataframes : ['x','y']
    """
    if isinstance(p, pd.DataFrame):
        points1 = p.to_numpy()
        points2 = q.to_numpy()

    elif isinstance(p, (tuple, list)) and len(p) == 2:
        points1 = np.array([[p[0], p[1]]])  # shape (1,2)
        points2 = np.array([[q[0], q[1]]])  # shape (1,2)
    else:
        points1 = np.asarray(p)
        points2 = np.asarray(q)

    distance = np.linalg.norm(points1 - points2, axis=1)

    if show_intermediate:
        fig, ax = plt.subplots(figsize=(8, 8))

        if outlines is not None:
            alphabin = 1 / len(outlines)  # determine number of alpha levels
            a = 1
            z = 1
            for outline in outlines:
                plt.imshow(outline, cmap="gray", origin="upper", alpha=a, zorder=z)
                a -= alphabin
                z += 1

        ax.scatter(
            points1[:, 0],
            points1[:, 1],
            s=20,
            color="blue",
            label="Nucleus centroid",
            zorder=10,
        )
        ax.scatter(
            points2[:, 0],
            points2[:, 1],
            s=20,
            color="red",
            label="Cell centroid",
            zorder=11,
        )

        plt.title("Nucleus Displacement from Cell Center")
        plt.legend(loc="lower right")
        plt.axis("off")

        plt.show()

    return distance


def ellipse_orientation_from_mask(mask):
    """Get orientation (in degrees) of best-fitting ellipse to the mask."""

    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return np.nan
    return props[0].orientation, props[0]  # Radians, counter-clockwise from horizontal


def draw_region_ellipse(
    region, ax, edgecolour="white", draw_horizontal=True, zorder_min=6, draw_minor=False
):
    """Draw ellipse patch, major & minor axes, and short horizontal centroid line."""

    cy, cx = region.centroid  # note: (row, col)
    major = region.major_axis_length
    minor = region.minor_axis_length
    theta = region.orientation  # radians, skimage: CCW from horizontal in image coords

    # --- Correct half-axis vectors in PLOTTING coords (x right, y down) ---
    # major_vec points along the major axis (half-length)
    major_vec = np.array([-math.sin(theta), -math.cos(theta)]) * (major / 2.0)
    # minor_vec is perpendicular (half-length)
    minor_vec = np.array([math.cos(theta), -math.sin(theta)]) * (minor / 2.0)

    # --- draw parametric ellipse so it exactly matches axis vectors ---
    t = np.linspace(0.0, 2.0 * np.pi, 300)
    # center + cos(t)*major_vec + sin(t)*minor_vec  (consistent choice)
    xs = cx + np.cos(t) * major_vec[0] + np.sin(t) * minor_vec[0]
    ys = cy + np.cos(t) * major_vec[1] + np.sin(t) * minor_vec[1]
    ax.plot(
        xs, ys, linestyle="dotted", linewidth=2, color=edgecolour, zorder=zorder_min
    )
    # poly = Polygon(np.column_stack([xs, ys]), closed=True, facecolor='none',
    # edgecolor=edgecolour, linewidth=1.25, zorder=zorder_min-1)
    # ax.add_patch(poly)

    # --- major axis line (use major_vec) ---
    x1, y1 = cx - major_vec[0], cy - major_vec[1]
    x2, y2 = cx + major_vec[0], cy + major_vec[1]
    ax.plot([x1, x2], [y1, y2], color=edgecolour, linewidth=2, zorder=zorder_min + 1)

    # --- optional minor axis (use minor_vec) ---
    if draw_minor:
        xm1, ym1 = cx - minor_vec[0], cy - minor_vec[1]
        xm2, ym2 = cx + minor_vec[0], cy + minor_vec[1]
        ax.plot(
            [xm1, xm2], [ym1, ym2], color=edgecolour, linewidth=2, zorder=zorder_min + 1
        )

    # centroid marker
    ax.scatter([cx], [cy], c=edgecolour, s=20, zorder=zorder_min + 2)

    # short horizontal line centered on centroid (as requested earlier)
    if draw_horizontal:
        line_length = 600
        half = line_length / 2.0
        ax.plot(
            [cx - half, cx + half], [cy, cy], color="red", lw=2, zorder=zorder_min + 1
        )


def angle_between_ellipses(mask1, mask2, show_intermediate=False):
    """Compute smallest angle (in degrees) between two ellipse orientations."""
    angle1, region1 = ellipse_orientation_from_mask(mask1)
    angle2, region2 = ellipse_orientation_from_mask(mask2)

    if np.isnan(angle1) or np.isnan(angle2):
        return np.nan

    diff = np.abs(angle1 - angle2)
    diff = min(diff, np.pi - diff)

    if show_intermediate:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(np.zeros_like(mask1), cmap="gray", interpolation="nearest")
        ax.imshow(
            mask1.astype(float),
            cmap="gray",
            alpha=1,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            zorder=1,
        )
        ax.imshow(
            mask2.astype(float),
            cmap="gray",
            alpha=0.5,
            interpolation="nearest",
            vmin=0,
            vmax=1,
            zorder=2,
        )

        # Draw a line across the major axis to indicate orientation
        draw_region_ellipse(region1, ax, edgecolour="blue", draw_horizontal=True)
        draw_region_ellipse(region2, ax, edgecolour="lime", draw_horizontal=True)

        ax.set_xlim(-0.5, mask1.shape[1] - 0.5)
        ax.set_ylim(
            mask1.shape[0] - 0.5, -0.5
        )  # y inverted for image coord consistency
        ax.set_aspect("equal")
        ax.axis("off")

        legend_handles = [
            Line2D(
                [0], [0], color="blue", lw=2, linestyle="dotted", label="Cell Ellipse"
            ),
            Line2D([0], [0], color="blue", lw=2, label="Cell Major Axis"),
            Line2D(
                [0],
                [0],
                color="lime",
                lw=2,
                linestyle="dotted",
                label="Nucleus Ellipse",
            ),
            Line2D([0], [0], color="lime", lw=2, label="Nucleus Major Axis"),
            Line2D([0], [0], color="red", lw=2, label="Horizontal"),
        ]

        ax.legend(
            handles=legend_handles,
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="gray",
            fontsize=10,
        )

        plt.title("Nucleus Orientation")
        plt.show()

    return np.degrees(diff), np.degrees(angle1), np.degrees(angle2)


def single_cell_measurements_nucleus(
    cell,
    updated_dataframes: dict,
    all_segmentation: dict,
    DEBUG=False,
):
    cell_segmentation = all_segmentation[constants.CELL]
    nucleus_segmentation = all_segmentation[constants.NUCLEUS]

    cell_dataset = updated_dataframes[constants.CELL]
    nucleus_dataset = updated_dataframes[constants.NUCLEUS]

    subset_nucleus = nucleus_dataset[
        nucleus_dataset[constants.CELLID_COL] == cell
    ].reset_index()
    subset_cell = cell_dataset[cell_dataset[constants.CELLID_COL] == cell].reset_index()

    # calculations per cell
    nucleus_orientation, angle1, angle2 = angle_between_ellipses(
        cell_segmentation[cell], nucleus_segmentation[cell]
    )
    convex_props = utils.convexhull_props_singleroi(nucleus_segmentation[cell])
    # ext_props = utils.largest_distance_between_two_points(nucleus_segmentation[cell])
    centroid_nucleus = tuple(
        subset_nucleus.loc[
            subset_nucleus[constants.CELLID_COL] == cell,
            ["AreaShape_Center_X", "AreaShape_Center_Y"],
        ].iloc[0]
    )
    centroid_cell = tuple(
        subset_cell.loc[
            subset_cell[constants.CELLID_COL] == cell,
            ["AreaShape_Center_X", "AreaShape_Center_Y"],
        ].iloc[0]
    )
    # input for distance between centroids
    # centroid_cell = cell_dataset[["AreaShape_Center_X", "AreaShape_Center_Y"]]
    # centroid_nucleus = nucleus_dataset[["AreaShape_Center_X", "AreaShape_Center_Y"]]

    dataset_calcs = pd.DataFrame(
        {
            constants.CELLID_COL: cell,
            "Structure_NucleusOrientation": nucleus_orientation,
            # "AreaShape_MaximumExtension": ext_props["maximum_extension"], # too large
            "AreaShape_FormFactorRatio": subset_nucleus["AreaShape_FormFactor"]
            / convex_props["hull_circularity"],
            "AreaShape_AspectRatio": subset_nucleus["AreaShape_MajorAxisLength"]
            / subset_nucleus["AreaShape_MinorAxisLength"],
            "AreaShape_ConvexFormFactor": convex_props["hull_circularity"].values[0],
            "Structure_NucleusDisplacement": utils.distance_pt(
                centroid_cell, centroid_nucleus
            ),
            "Structure_ObjectProportion": subset_nucleus["AreaShape_Area"]
            / subset_cell["AreaShape_Area"],
            "Intensity_RangeEdge_xray": subset_nucleus[
                "Intensity_MaxIntensityEdge_xray"
            ]
            - subset_nucleus["Intensity_MinIntensityEdge_xray"],
            "Intensity_Range_xray": subset_nucleus["Intensity_MaxIntensity_xray"]
            - subset_nucleus["Intensity_MinIntensity_xray"],
        }
    )

    if DEBUG:
        return dataset_calcs

    dataset_avg = utils.texture_measure_average(subset_cell)
    per_cell_measures = pd.concat([dataset_calcs, dataset_avg], axis=1)

    return per_cell_measures


def feature_extract_nucleus(
    updated_dataframes: dict,
    all_segmentation: dict,
    savedir: Path = "",
    DEBUG=False,
):
    cell_dataset = updated_dataframes["cell"]
    nucleus_dataset = updated_dataframes["nucleus"]

    cell_list = list(cell_dataset[constants.CELLID_COL])
    with tqdm_joblib(desc="Processing", total=len(cell_list)):
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(single_cell_measurements_nucleus)(
                cell, updated_dataframes, all_segmentation
            )
            for cell in cell_list
        )

    if DEBUG:
        return results

    per_cell_measures_df = pd.concat(results)

    nucleus_dataset_updated = nucleus_dataset.copy()
    nucleus_dataset_updated = nucleus_dataset_updated.merge(
        per_cell_measures_df, on=[constants.CELLID_COL], how="left"
    )

    if savedir != "":
        nucleus_dataset_updated.to_excel(
            savedir / f"{constants.NUCLEUS}.xlsx", index=False
        )

    return nucleus_dataset_updated
