import numpy as np
from pointpats import centrography
import libpysal
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.morphology import skeletonize
from scipy.ndimage import (
    convolve,
    binary_dilation,
    distance_transform_edt,
    center_of_mass,
)
from skimage.graph import route_through_array
from scipy.spatial.distance import cdist, pdist, squareform
import feature_extraction_utils as utils
from skimage.measure import label, regionprops
from skan import Skeleton, summarize
from itertools import combinations
from pathlib import Path
import pandas as pd
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from utils import get_unique_filename
import constants

def object_extent_img(
    binary_image: np.ndarray, plot: bool = False, outline: np.ndarray = None
):
    contours, _ = cv2.findContours(
        binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    coordinates = np.vstack([c[:, 0, :] for c in contours])

    convex_hull_vertices = centrography.hull(coordinates)
    alpha_shape, alpha, circs = libpysal.cg.alpha_shape_auto(
        coordinates, return_circles=True
    )

    extent = alpha_shape.area

    if plot:
        plt.imshow(binary_image, cmap="gray")
        x, y = alpha_shape.exterior.xy
        plt.plot(x, y, color="red", linewidth=1.5, linestyle="dotted", label="Extent")

        if outline is not None:
            outline_img = (outline > 0).astype(np.uint8) * 255
            o_contours, _ = cv2.findContours(
                outline_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Plot each outline contour
            for oc in o_contours:
                if oc.shape[0] < 2:
                    continue
                oc_xy = oc[:, 0, :]
                # cv2 uses (x, y) order for points; matplotlib expects (x, y) pairs
                plt.plot(
                    oc_xy[:, 0],
                    oc_xy[:, 1],
                    color="blue",
                    linewidth=2,
                    label="Cell Outline"
                    if "Cell Outline" not in plt.gca().get_legend_handles_labels()[1]
                    else "",
                )

        plt.title("Extent")
        plt.legend(loc="lower right")
        plt.axis("off")
        plt.show()

    return extent


def cluster_number_shapes():
    return


# normalize angles to [0,360)
def _norm360(a):
    a = a % 360.0
    if a < 0:
        a += 360.0
    return a


# normalize and scale for plotting
def _scaled_vec(v, scale):
    n = np.linalg.norm(v)
    if n == 0:
        return np.array([0.0, 0.0])
    return (v / n) * scale


def object_orientation_pt(binary_roi, center, center_roi=None, show_intermediate=False):
    """
    Determine orientation of single ROI relative to reference point
    binary_roi: binary image with single roi (bool)
    center_roi: binary image withe nucleus (bool)

    """
    coords = np.column_stack(np.nonzero(binary_roi > 0))
    ref_x, ref_y = center[0], center[1]

    coords = coords[:, ::-1]  # Switch to (x, y) convention

    # pairwise distances
    dist_matrix = cdist(coords, coords, metric="euclidean")  # pair-wise distances
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

    # Get the two furthest-apart points
    x1, y1 = coords[i]
    x2, y2 = coords[j]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2

    # Vectors
    v_branch = np.array([x2 - x1, y2 - y1])
    v_ref_to_mid = np.array([mx - ref_x, my - ref_y])

    # Angle calculation
    dot = np.dot(v_branch, v_ref_to_mid)
    norms = np.linalg.norm(v_branch) * np.linalg.norm(v_ref_to_mid)

    if norms > 0:
        angle_rad = np.arccos(np.clip(dot / norms, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
    else:
        angle_deg = np.nan

    if show_intermediate:
        nrows, ncols = binary_roi.shape

        fig, ax = plt.subplots(figsize=(6, 6))

        # Show ROI image (display with origin='upper' so rows are y)
        ax.imshow(binary_roi, cmap="gray", origin="upper", interpolation="nearest")
        # overlay center mask if available
        if center_roi is not None:
            masked = np.ma.masked_where(~center_roi, center_roi)
            ax.imshow(
                masked,
                cmap="gray",
                origin="upper",
                alpha=0.8,
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

        # Endpoints
        ax.scatter(
            [x1, x2],
            [y1, y2],
            s=40,
            facecolors="red",
            edgecolors="red",
            label="endpoints",
        )

        # Midpoint
        ax.scatter([mx], [my], s=40, color="lime", label="midpoint")

        # Center point
        ax.scatter([ref_x], [ref_y], s=40, color="blue", label="center")

        # -----------------------------
        #   LINE: Endpoint → Endpoint
        # -----------------------------
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="red",
            linewidth=1,
            label="endpoint → endpoint",
            linestyle="dotted",
        )

        # -----------------------------
        #   LINE: Midpoint → Center
        # -----------------------------
        ax.plot(
            [mx, ref_x],
            [my, ref_y],
            color="blue",
            linewidth=1,
            label="midpoint → center",
            linestyle="dotted",
        )

        # full image preserved
        ax.set_title("Orientation to Nucleus")
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.axis("off")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    return angle_deg


# def object_orientation_axis(binary_roi, line):
# return angle_deg


def object_orientation_shape(binary_roi, binary_shape):
    utils.minimum_enclosing_circle(binary_shape, shape="ellipse")

    return


def branch_vector_angle(src, dst):
    vec = dst - src
    angle = np.degrees(np.arctan2(vec[0], vec[1]))  # y, x order
    return angle


def network_analysis(binary_img, show_intermediate=False):
    skeleton = skeletonize(binary_img)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 10, 1],  # center pixel marked strongly
            [1, 1, 1],
        ]
    )

    neighbor_count = (
        convolve(skeleton.astype(int), np.ones((3, 3)), mode="constant") - skeleton
    )

    # Branch points: >=3 neighbors
    branch_points = skeleton & (neighbor_count >= 3)

    # Endpoints: exactly 1 neighbor
    end_points = skeleton & (neighbor_count == 1)

    labels = label(skeleton)
    props = regionprops(labels)

    individual_count = 0
    network_count = 0
    num_branches_network_total = 0
    for i, region in enumerate(props, 1):
        mask = labels == i
        num_branches = np.count_nonzero(branch_points & mask)
        num_endpoints = np.count_nonzero(end_points & mask)

        if num_branches == 0:
            individual_count += 1
        else:
            network_count += 1
            num_branches_network_total += num_branches

    skeleton_graph = Skeleton(skeleton)
    branch_info = summarize(
        skeleton_graph, separator="_"
    )  # branch distance - curved length, branch_distance_euclidean - straight line distance

    # branch width
    distance_map = distance_transform_edt(binary_img)
    branch_widths = []
    for branch_id in range(len(branch_info)):
        coords = skeleton_graph.path_coordinates(branch_id)
        widths = [2 * distance_map[y, x] for y, x in coords]
        branch_widths.append(np.mean(widths))

    # Add widths to branch_data
    branch_info["mean_width"] = branch_widths

    # branch length - curved and euclidean
    network_id = branch_info["skeleton_id"].unique()
    branch_length_individual = []
    branch_length_individual_euclidean = []
    branch_length_network = []
    branch_length_network_euclidean = []
    branch_width_individual = []
    branch_width_network = []

    for network in network_id:
        branch_subset = branch_info.loc[branch_info["skeleton_id"] == network]

        if len(branch_subset) == 1:  # individual branch
            branch_length_individual.append(branch_subset["branch_distance"])
            branch_length_individual_euclidean.append(
                branch_subset["euclidean_distance"]
            )
            branch_width_individual.append(branch_subset["mean_width"])
        else:
            branch_length_network.append(branch_subset["branch_distance"])
            branch_length_network_euclidean.append(branch_subset["euclidean_distance"])
            branch_width_network.append(branch_subset["mean_width"])

    if len(branch_length_individual) == 0:  # no individual branches
        branch_length_individual_all = np.array(0)
    else:  # at least one individual branch
        branch_length_individual_all = pd.concat(
            branch_length_individual, ignore_index=True
        ).to_numpy()

    if len(branch_length_individual_euclidean) == 0:  # no individual branches
        branch_length_individual_euclidean_all = np.array(0)
    else:
        branch_length_individual_euclidean_all = pd.concat(
            branch_length_individual_euclidean, ignore_index=True
        ).to_numpy()

    if len(branch_width_individual) == 0:  # no individual branches
        branch_width_individual_all = np.array(0)
    else:
        branch_width_individual_all = pd.concat(
            branch_width_individual, ignore_index=True
        ).to_numpy()

    if len(branch_length_network) == 0:  # no individual branches
        branch_length_network_all = np.array(0)
    else:
        branch_length_network_all = pd.concat(
            branch_length_network, ignore_index=True
        ).to_numpy()

    if len(branch_length_network_euclidean) == 0:  # no individual branches
        branch_length_network_euclidean_all = np.array(0)
    else:
        branch_length_network_euclidean_all = pd.concat(
            branch_length_network_euclidean, ignore_index=True
        ).to_numpy()

    if len(branch_width_network) == 0:  # no individual branches
        branch_width_network_all = np.array(0)
    else:
        branch_width_network_all = pd.concat(
            branch_width_network, ignore_index=True
        ).to_numpy()

    # calculate branch angles
    coords = skeleton_graph.coordinates
    branch_info["angle"] = [
        branch_vector_angle(coords[src], coords[dst])
        for src, dst in zip(branch_info["node_id_src"], branch_info["node_id_dst"])
    ]

    branch_angles = {}
    for node_id in np.unique(branch_info[["node_id_src", "node_id_dst"]]):
        connected = branch_info[
            (branch_info["node_id_src"] == node_id)
            | (branch_info["node_id_dst"] == node_id)
        ]
        angles = connected["angle"].values
        if len(angles) >= 2:
            diffs = []
            for a1, a2 in combinations(angles, 2):
                diff = abs(a1 - a2) % 180  # angle between vectors
                diffs.append(min(diff, 180 - diff))  # keep acute angle
            branch_angles[node_id] = diffs

    all_angles = [val for sublist in branch_angles.values() for val in sublist]
    angles = np.array(all_angles) if len(all_angles) >= 1 else np.array(0)

    results = {
        "Structure_IndividualObjectNumber": individual_count,
        "Structure_BranchType0": (branch_info["branch_type"] == 0).sum(),
        "Structure_BranchType1": (branch_info["branch_type"] == 1).sum(),
        "Structure_BranchType2": (branch_info["branch_type"] == 2).sum(),
        "Structure_BranchType3": (branch_info["branch_type"] == 3).sum(),
        "Structure_NetworkNumber": network_count,
        "Structure_NetworkAverageBranchDensity": (
            num_branches_network_total / network_count if network_count != 0 else 0
        ),
        "Structure_NetworkBranchLengthMean": (
            np.mean(branch_length_network_all)
            if branch_length_network_all.size > 1
            else np.int64(branch_length_network_all)
        ),
        "Structure_NetworkBranchLengthMedian": (
            np.median(branch_length_network_all)
            if branch_length_network_all.size > 1
            else np.int64(branch_length_network_all)
        ),
        "Structure_NetworkBranchLengthStd": (
            np.std(branch_length_network_all)
            if branch_length_network_all.size > 1
            else 0
        ),
        "Structure_IndividualBranchLengthMean": (
            np.mean(branch_length_individual_all)
            if branch_length_individual_all.size > 1
            else np.int64(branch_length_individual_all)
        ),
        "Structure_IndividualBranchLengthMedian": (
            np.median(branch_length_individual_all)
            if branch_length_individual_all.size > 1
            else np.int64(branch_length_individual_all)
        ),
        "Structure_IndividualBranchLengthStd": (
            np.std(branch_length_individual_all)
            if branch_length_individual_all.size > 1
            else 0
        ),
        "Structure_NetworkBranchWidthMean": (
            np.mean(branch_width_network_all)
            if branch_width_network_all.size > 1
            else np.int64(branch_width_network_all)
        ),
        "Structure_NetworkBranchWidthMedian": (
            np.median(branch_width_network_all)
            if branch_width_network_all.size > 1
            else np.int64(branch_width_network_all)
        ),
        "Structure_NetworkBranchWidthStd": (
            np.std(branch_width_network_all) if branch_width_network_all.size > 1 else 0
        ),
        "Structure_IndividualBranchWidthMean": (
            np.mean(branch_width_individual_all)
            if branch_width_individual_all.size > 1
            else np.int64(branch_width_individual_all)
        ),
        "Structure_IndividualBranchWidthMedian": (
            np.median(branch_width_individual_all)
            if branch_width_individual_all.size > 1
            else np.int64(branch_width_individual_all)
        ),
        "Structure_IndividualBranchWidthStd": (
            np.std(branch_width_individual_all)
            if branch_width_individual_all.size > 1
            else 0
        ),
        "Structure_NetworkBranchAngleMean": (
            np.mean(angles) if angles.size > 1 else np.int64(angles)
        ),
        "Structure_NetworkBranchAngleStd": (np.std(angles) if angles.size > 1 else 0),
        "Structure_NetworkBranchAngleMed": (
            np.median(angles) if angles.size > 1 else np.int64(angles)
        ),
        "Structure_ProportionNetwork": (
            1 - ((branch_info["branch_type"] == 0).sum() / len(branch_info))
            if len(branch_info) != 0
            else 0
        ),
    }

    if show_intermediate:
        # Overlay points on skeleton
        ys, xs = np.where(skeleton)

        plt.figure(figsize=(6, 6))
        plt.imshow(binary_img, cmap="gray")

        # Plot skeleton pixels as thick points
        plt.scatter(xs, ys, s=0.5, c="red")  # s controls size (thickness)

        # Show branch points in red
        y_branch, x_branch = np.nonzero(branch_points)  #
        plt.scatter(x_branch, y_branch, c="blue", s=5, label="Branch points")

        # Show end points in green
        y_end, x_end = np.nonzero(end_points)
        plt.scatter(x_end, y_end, c="lime", s=5, label="End points")

        plt.legend(loc="lower right")
        plt.axis("off")
        plt.title("Skeleton Network")
        plt.show()

    return results


def find_endpoints(skel):
    # Kernel to count 8-connected neighbors
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    neighbors = convolve(skel.astype(int), kernel, mode="constant")
    # Endpoint pixels have exactly one neighbor (value 11)
    endpoints = np.argwhere(neighbors == 11)

    return endpoints


def tortuosity(binary_image, show_intermediate=False):
    """
    Input: binary image with single roi

    """
    # Skeletonize the mask
    skel = skeletonize(binary_image)

    # Find endpoints of skeleton
    endpoints = find_endpoints(skel)
    if len(endpoints) > 2:
        # print(f"found {len(endpoints)} endpoints --- using longest branch")
        pair_dist = squareform(pdist(endpoints, metric="euclidean"))
        i, j = np.unravel_index(np.argmax(pair_dist), pair_dist.shape)
        endpoints = [endpoints[i], endpoints[j]]

    path = None
    # Compute shortest path along skeleton (arc)
    if len(endpoints) == 2:
        cost = np.where(skel, 1.0, 1e10)  # high cost for non-skeleton
        path, _ = route_through_array(
            cost, tuple(endpoints[0]), tuple(endpoints[1]), fully_connected=True
        )
        path = np.array(path)
        arc_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

        # Chord length
        chord_length = np.linalg.norm(endpoints[0] - endpoints[1])
        tortuosity = arc_length / chord_length if chord_length > 0 else np.nan

    elif len(endpoints) < 2:
        tortuosity = 1

    else:
        tortuosity = np.nan

    if show_intermediate:
        # compute bounding box of ROI
        rows, cols = np.nonzero(binary_image)
        if rows.size == 0 or cols.size == 0:
            # nothing to show; display full image
            r0, r1 = 0, binary_image.shape[0] - 1
            c0, c1 = 0, binary_image.shape[1] - 1
        else:
            pad = 100
            r0 = max(0, rows.min() - pad)
            r1 = min(binary_image.shape[0] - 1, rows.max() + pad)
            c0 = max(0, cols.min() - pad)
            c1 = min(binary_image.shape[1] - 1, cols.max() + pad)

        # crop images
        crop = binary_image[r0 : r1 + 1, c0 : c1 + 1]
        # display
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(crop, cmap="gray", origin="upper", interpolation="nearest")

        # If we have a path, overlay it (convert (row, col) to (x=col, y=row))
        if path is not None and path.size:
            # shift coordinates because of crop
            path_rows = path[:, 0] - r0
            path_cols = path[:, 1] - c0
            ax.plot(
                path_cols,
                path_rows,
                linewidth=2.0,
                linestyle="-",
                label="arc",
                zorder=4,
            )

        # If we have endpoints, plot them and chord
        if len(endpoints) >= 1:
            ep_array = np.array(endpoints)
            ep_rows = ep_array[:, 0] - r0
            ep_cols = ep_array[:, 1] - c0
            ax.scatter(
                ep_cols,
                ep_rows,
                s=60,
                facecolors="none",
                edgecolors="red",
                label="endpoints",
                zorder=5,
            )

        if len(endpoints) == 2:
            # chord line between the two endpoints
            e0 = endpoints[0] - np.array([r0, c0])
            e1 = endpoints[1] - np.array([r0, c0])
            # x are columns, y are rows
            ax.plot(
                [e0[1], e1[1]],
                [e0[0], e1[0]],
                linewidth=2.0,
                linestyle="--",
                label="chord",
                zorder=3,
            )

        ax.axis("off")
        ax.set_xlim(-0.5, (c1 - c0) + 0.5)
        ax.set_ylim(
            (r1 - r0) + 0.5, -0.5
        )  # invert y-axis to match image origin='upper'
        # ax.set_title(f"Tortuosity = {tortuosity if not np.isnan(tortuosity) else 'nan'}")
        ax.set_title("Tortuosity: arc length / chord length ")

        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    return tortuosity


def single_cell_measurements_mito(
    cell, updated_dataframe, all_segmentation, DEBUG=False
):
    mitochondria_dataset = updated_dataframe["mitochondria"]
    nucleus_dataset = updated_dataframe["nucleus"]
    cell_dataset = updated_dataframe["cell"]

    mitochondria_segmentation = all_segmentation["mitochondria"]
    cell_segmentation = all_segmentation["cell"]
    nucleus_segmentation = all_segmentation["nucleus"]

    results_per_object = []
    subset_dataframe = mitochondria_dataset[
        mitochondria_dataset["Metadata_CellID"] == cell
    ].reset_index()

    mitochondria_points_centroid = list(
        zip(
            subset_dataframe["AreaShape_Center_X"],
            subset_dataframe["AreaShape_Center_Y"],
        )
    )
    subset_images_list = list(
        mitochondria_segmentation.loc[
            mitochondria_segmentation["Metadata_CellID"] == cell, "image"
        ].values
    )
    subset_imagenum_list = list(
        mitochondria_segmentation.loc[
            mitochondria_segmentation["Metadata_CellID"] == cell, "ImageNumber"
        ].values
    )

    centroid_nucleus = tuple(
        nucleus_dataset.loc[
            nucleus_dataset["Metadata_CellID"] == cell,
            ["AreaShape_Center_X", "AreaShape_Center_Y"],
        ].iloc[0]
    )
    area_cell = cell_dataset.loc[
        cell_dataset["Metadata_CellID"] == cell, "AreaShape_Area"
    ].iloc[0]

    # list of images of each roi individually
    all_mito_imgs = {}
    for image_num, layer in zip(subset_imagenum_list, subset_images_list):
        n_obj = layer.max()
        for obj in list(range(1, n_obj + 1)):
            all_mito_imgs[str(image_num) + "_" + str(obj)] = layer == obj

    # calculations per object
    for obj_id, single_roi in all_mito_imgs.items():
        other_roi = [img for obj, img in all_mito_imgs.items() if obj != obj_id]
        obj_id_split = obj_id.split(
            "_"
        )  # recover ImageNumber and ObjectNumber for mapping to dataframe
        neighbour_distance, neighbour_number = utils.neighbour_analysis(
            single_roi, other_roi, neighbor_radius=50
        )
        convex_props = utils.convexhull_props_singleroi(single_roi)
        # ext_props = utils.largest_distance_between_two_points(single_roi)
        labelled = label(single_roi)
        regions = regionprops(labelled)
        obj_centroid_x, obj_centroid_y = regions[0].centroid
        obj_centroid = (
            obj_centroid_y,
            obj_centroid_x,
        )  # (x,y) format reversed (match cellprofiler)

        # obj_centroid_cellprofiler = (
        # subset_dataframe.loc[
        # (subset_dataframe["ImageNumber"] == int(obj_id_split[0]))
        # & (subset_dataframe["ObjectNumber"] == int(obj_id_split[1])),
        # ["AreaShape_Center_X", "AreaShape_Center_Y"],
        # ].iloc[0]
        # )
        # print(obj_centroid, obj_centroid_cellprofiler)

        results_per_object.append(
            {
                "Metadata_CellID": cell,
                "ImageNumber": int(obj_id_split[0]),
                "ObjectNumber": int(obj_id_split[1]),
                "AreaShape_Tortuosity": tortuosity(single_roi),
                "Structure_OrientationNucleus": object_orientation_pt(
                    single_roi, centroid_nucleus
                ),
                "AreaShape_FormFactorRatio": subset_dataframe[
                    (subset_dataframe["ImageNumber"] == int(obj_id_split[0]))
                    & (subset_dataframe["ObjectNumber"] == int(obj_id_split[1]))
                ]["AreaShape_FormFactor"]
                / convex_props["hull_circularity"],
                "Structure_NearestNeighbourDistance": neighbour_distance,
                "Structure_NeighbourNumber": neighbour_number,
                "AreaShape_ConvexFormFactor": convex_props["hull_circularity"].values[
                    0
                ],
                "Structure_DistanceNucleus": utils.distance_pt(
                    obj_centroid, centroid_nucleus
                ),  # calculate per object distance from nucleus
                # "AreaShape_MaximumExtension": ext_props["maximum_extension"],
            }
        )

    per_object_measures_df = pd.DataFrame(results_per_object)

    dataset_calcs = pd.DataFrame(
        {
            "AreaShape_AspectRatio": (
                subset_dataframe["AreaShape_MajorAxisLength"]
                / subset_dataframe["AreaShape_MinorAxisLength"]
            ),
            "Intensity_Range_xray": (
                subset_dataframe["Intensity_MaxIntensity_xray"]
                - subset_dataframe["Intensity_MinIntensity_xray"]
            ),
            "Intensity_RangeEdge_xray": (
                subset_dataframe["Intensity_MaxIntensityEdge_xray"]
                - subset_dataframe["Intensity_MinIntensityEdge_xray"]
            ),
        }
    )
    dataset_avg = utils.texture_measure_average(subset_dataframe)
    per_object_all_measures_df = pd.concat(
        [per_object_measures_df, dataset_calcs, dataset_avg], axis=1
    )
    if DEBUG:
        return per_object_measures_df

    # calculations per cell
    std_distance_summary = utils.std_distance_pt_summary_measures(
        mitochondria_points_centroid, centroid_nucleus
    )
    _, _, _, ellipse_AR = utils.ellipse_pt(
        mitochondria_points_centroid, centroid_nucleus
    )
    collapsed_mitochondria_segmentation = utils.collapse_multilayer(subset_images_list)
    R_obs = utils.object_polarization(mitochondria_points_centroid, centroid_nucleus)
    cellshape = utils.binary_to_polygon(cell_segmentation[cell])
    R_sim = utils.boundary_random_point_dist(
        cellshape, centroid_nucleus, len(mitochondria_points_centroid)
    )

    perinuclear_obj_total = 0
    cell_edge_obj_number_total = 0
    for layer_img in subset_images_list:
        perinuclear_obj_number, perinuclear_area, _ = utils.perinuclear_region_obj(
            nucleus_segmentation[cell], layer_img
        )

        cell_edge_obj_number, cell_edge_area, _ = utils.cell_edge_obj(
            cell_segmentation[cell], layer_img
        )

        perinuclear_obj_total += perinuclear_obj_number
        cell_edge_obj_number_total += cell_edge_obj_number

    network_results = network_analysis(collapsed_mitochondria_segmentation)
    number_overlapping = (
        per_object_all_measures_df["Structure_NearestNeighbourDistance"] == 0
    ).sum()  # additive - problem with per_object dataframe
    cluster_num, cluster_size, _, _ = utils.cluster_number_points(
        mitochondria_points_centroid, eps=200
    )

    # add results to dataframe
    per_cell_measures = {
        "Metadata_CellID": cell,
        "Structure_StandardDistanceNucleus": utils.std_distance_pt(
            mitochondria_points_centroid, centroid_nucleus
        ),
        # "Structure_DistanceNucleus_std": std_distance_summary["std_distance"],
        # "Structure_DistanceNucleus_skew": std_distance_summary["skew_distance"],
        # "Structure_DistanceNucleus_MAD": utils.median_distance_pt(
        # mitochondria_points_centroid, centroid_nucleus
        # ),
        # "Structure_DistanceNucleus_GiniCoeff": utils.gini_distance(
        # mitochondria_points_centroid, centroid_nucleus
        # ),
        "Structure_StdDistEllipse_aspectRatio": ellipse_AR,
        "Structure_ObjectCoverage": object_extent_img(
            collapsed_mitochondria_segmentation
        )
        / area_cell,
        "Structure_PolarizationNucleus": utils.degree_nonrandom_polarization(
            R_obs, R_sim
        ),
        "Structure_PerinuclearObjectDensity": perinuclear_obj_total / perinuclear_area,
        "Structure_PerinuclearObjectProportion": perinuclear_obj_total
        / len(mitochondria_points_centroid),
        "Structure_CellEdgeObjectDensity": cell_edge_obj_number_total / cell_edge_area,
        "Structure_CellEdgeObjectProportion": cell_edge_obj_number_total
        / len(mitochondria_points_centroid),
        "Structure_ObjectProportion": collapsed_mitochondria_segmentation.sum()
        / area_cell,
        "Structure_ClusterNumber": cluster_num,
        "Structure_ClusterSizeAvg": cluster_size,
        "Structure_ObjectDensity": len(mitochondria_points_centroid) / area_cell,
        "Structure_OverlappingNumber": number_overlapping,
    }

    per_cell_measures.update(network_results)

    return per_cell_measures, per_object_all_measures_df


def feature_extract_mitochondria(
    updated_dataframes: dict, all_segmentation: dict, savedir: Path = "", DEBUG=False, batch = (1, -1)
):
    cell_dataset = updated_dataframes["cell"]
    mitochondria_dataset = updated_dataframes["mitochondria"]

    cell_list = list(cell_dataset["Metadata_CellID"])
    
    if DEBUG:
        start, end = batch
        cell_list = cell_list[start:end]

    with tqdm_joblib(desc="Processing", total=len(cell_list)):
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(single_cell_measurements_mito)(
                cell, updated_dataframes, all_segmentation
            )
            for cell in cell_list
        )

    per_cell_measures_list, per_object_measures_list = zip(*results)
    per_cell_measures_df = pd.DataFrame(per_cell_measures_list)
    per_object_measures_df = pd.concat(per_object_measures_list)

    mitochondria_dataset_updated = mitochondria_dataset.copy()
    mitochondria_dataset_updated = mitochondria_dataset_updated.merge(
        per_cell_measures_df, on=["Metadata_CellID"], how="left"
    )
    mitochondria_dataset_updated = mitochondria_dataset_updated.merge(
        per_object_measures_df,
        on=["ObjectNumber", "ImageNumber", "Metadata_CellID"],
        how="left",
    )

    if savedir != "":
        Path(savedir).mkdir(parents=True, exist_ok=True)

        filepath = savedir / f"{constants.MITOCHONDRIA}.xlsx"
        filename = get_unique_filename(filepath)
        
        mitochondria_dataset_updated.to_excel(
            savedir / filename, index=False
        )

    return mitochondria_dataset_updated
