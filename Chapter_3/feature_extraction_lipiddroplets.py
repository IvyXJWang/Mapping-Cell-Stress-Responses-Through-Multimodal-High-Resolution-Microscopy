# https://pysal.org/pointpats/_modules/pointpats/centrography.html#std_distance
# modified functions to calculate standard distances from user specified distance
import numpy as np
import math
from shapely.geometry import box, Point
from pointpats import centrography
import libpysal
import matplotlib as plt
import geopandas as gpd
from pathlib import Path
import pandas as pd
import feature_extraction_utils as utils
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from utils import get_unique_filename


import constants


def average_nearest_neighbor(points):
    """A function to calculate Average nearest neighbor ratio (https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/h-how-average-nearest-neighbor-distance-spatial-st.htm#ESRI_SECTION1_F44811E57F3F408A8C4880DD7370FB99)"""

    df_centroids = gpd.GeoDataFrame(
        pd.DataFrame(points, columns=["x", "y"]),
        geometry=[Point(x, y) for x, y in points],
        crs=None,
    )

    # Calculate DO: Observed mean distance DO: "Mean distance between each feature and its nearest neighbor"
    sj = df_centroids.sjoin_nearest(
        df_centroids, distance_col="distance", exclusive=True
    )
    DO = sj["distance"].mean().item()  # Calculate the mean distance between each pair

    # Calculate DE: "Expected mean distance for features given in a random pattern" 0.5 / square_root(n/A)
    DE = 0.5 / math.sqrt(len(points) / box(*df_centroids.total_bounds).area)

    # Calculate ANN: "Average nearest neighbor ratio" (NNI)
    ANN = DO / DE

    return ANN, sj


def object_extent_point(points, plot=False):
    coordinates = np.array(points)

    convex_hull_vertices = centrography.hull(coordinates)
    alpha_shape, alpha, circs = libpysal.cg.alpha_shape_auto(
        coordinates, return_circles=True
    )

    extent = alpha_shape.area

    if plot:
        f, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Plot a green alpha shape
        gpd.GeoSeries([alpha_shape]).plot(
            ax=ax,
            edgecolor="green",
            facecolor="green",
            alpha=0.2,
            # label="Tightest single alpha shape"
        )

        # Include the points for lipid droplets in black
        ax.scatter(*coordinates.T, color="k", marker=".", label="Lipid droplets")

        # plot the circles forming the boundary of the alpha shape
        for i, circle in enumerate(circs):
            # only label the first circle of its kind
            if i == 0:
                label = "Bounding Circles"
            else:
                label = None
                ax.add_patch(
                    plt.Circle(
                        circle,
                        radius=alpha,
                        facecolor="none",
                        edgecolor="r",
                        label=label,
                    )
                )

        # add a blue convex hull
        ax.add_patch(
            plt.Polygon(
                convex_hull_vertices,
                closed=True,
                edgecolor="blue",
                facecolor="none",
                linestyle=":",
                linewidth=2,
                label="Convex Hull",
            )
        )

        plt.legend()
        plt.show()

    return extent


def single_cell_measurements_lipid(
    cell, updated_dataframe, all_segmentation, DEBUG=False
):
    cell_segmentation = all_segmentation[constants.CELL]
    nucleus_segmentation = all_segmentation[constants.NUCLEUS]
    lipiddroplets_segmentation = all_segmentation[constants.LIPID_DROPLETS]

    cell_dataset = updated_dataframe[constants.CELL]
    lipiddroplets_dataset = updated_dataframe[constants.LIPID_DROPLETS]
    nucleus_dataset = updated_dataframe[constants.NUCLEUS]

    results_per_object = []
    subset_dataframe = lipiddroplets_dataset[
        lipiddroplets_dataset["Metadata_CellID"] == cell
    ].reset_index()

    lipid_droplet_points_centroid = list(
        zip(
            subset_dataframe["AreaShape_Center_X"],
            subset_dataframe["AreaShape_Center_Y"],
        )
    )
    subset_image = lipiddroplets_segmentation[cell]

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
    all_lipid_imgs = {}

    n_obj = subset_image.max()
    for obj in list(range(1, n_obj + 1)):
        all_lipid_imgs[str(obj)] = subset_image == obj

    for obj_id, single_roi in all_lipid_imgs.items():
        other_roi = [img for obj, img in all_lipid_imgs.items() if obj != obj_id]
        obj_centroid = lipid_droplet_points_centroid[int(obj_id) - 1]

        neighbour_distance, neighbour_number = utils.neighbour_analysis(
            single_roi, other_roi, neighbor_radius=50
        )

        # print(obj_centroid)
        results_per_object.append(
            {
                "Metadata_CellID": cell,
                "ObjectNumber": int(obj_id),
                "Structure_NearestNeighbourDistance": neighbour_distance,
                "Structure_NeighbourNumber": neighbour_number,
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

    # calculations per cell
    lipiddroplet_total_area = subset_dataframe["AreaShape_Area"].sum()
    std_distance_summary = utils.std_distance_pt_summary_measures(
        lipid_droplet_points_centroid, centroid_nucleus
    )
    _, _, _, ellipse_AR = utils.ellipse_pt(
        lipid_droplet_points_centroid, centroid_nucleus
    )
    R_obs = utils.object_polarization(lipid_droplet_points_centroid, centroid_nucleus)
    cellshape = utils.binary_to_polygon(cell_segmentation[cell])
    R_sim = utils.boundary_random_point_dist(
        cellshape, centroid_nucleus, len(lipid_droplet_points_centroid)
    )
    perinuclear_obj, perinuclear_area, _ = utils.perinuclear_region_obj(
        nucleus_segmentation[cell], lipiddroplets_segmentation[cell]
    )
    cell_edge_obj_number, cell_edge_area, _ = utils.cell_edge_obj(
        cell_segmentation[cell], lipiddroplets_segmentation[cell]
    )
    ANN, neighbor_distances = average_nearest_neighbor(lipid_droplet_points_centroid)
    cluster_num, cluster_size, _, _ = utils.cluster_number_points(
        lipid_droplet_points_centroid
    )

    # add results to dataframe
    per_cell_measures = {
        constants.CELLID_COL: cell,
        "Structure_StandardDistanceNucleus": utils.std_distance_pt(
            lipid_droplet_points_centroid, centroid_nucleus
        ),
        # "Structure_DistanceNucleus_std": std_distance_summary["std_distance"],
        # "Structure_DistanceNucleus_skew": std_distance_summary["skew_distance"],
        # "Structure_DistanceNucleus_MAD": utils.median_distance_pt(
        # lipid_droplet_points_centroid, centroid_nucleus
        # ),
        # "Structure_DistanceNucleus_GiniCoeff": utils.gini_distance(
        #  lipid_droplet_points_centroid, centroid_nucleus
        # ),
        "Structure_StdDistEllipse_aspectRatio": ellipse_AR,
        "Structure_ObjectCoverage": object_extent_point(lipid_droplet_points_centroid)
        / area_cell,
        "Structure_PolarizationNucleus": utils.degree_nonrandom_polarization(
            R_obs, R_sim
        ),
        "Structure_PerinuclearObjectDensity": perinuclear_obj / perinuclear_area,
        "Structure_PerinuclearObjectProportion": perinuclear_obj
        / len(lipid_droplet_points_centroid),
        "Structure_CellEdgeObjectDensity": cell_edge_obj_number / cell_edge_area,
        "Structure_CellEdgeObjectProportion": cell_edge_obj_number
        / len(lipid_droplet_points_centroid),
        "Structure_ObjectProportion": lipiddroplet_total_area / area_cell,
        "Structure_AverageNearestNeighbourIndex": ANN,
        "Structure_ClusterNumber": cluster_num,
        "Structure_ClusterSizeAvg": cluster_size,
        "Structure_ObjectDensity": len(lipid_droplet_points_centroid) / area_cell,
    }

    if DEBUG:
        return per_object_all_measures_df

    return per_cell_measures, per_object_all_measures_df


def feature_extract_lipids(
    updated_dataframes: dict, all_segmentation: dict, savedir: Path = "", DEBUG=False, batch = (1, -1)
):
    cell_dataset = updated_dataframes[constants.CELL]
    lipiddroplets_dataset = updated_dataframes[constants.LIPID_DROPLETS]
    cell_list = list(cell_dataset[constants.CELLID_COL])

    if DEBUG:
        start, end = batch
        cell_list = cell_list[start:end]

    with tqdm_joblib(desc="Processing", total=len(cell_list)):
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(single_cell_measurements_lipid)(
                cell, updated_dataframes, all_segmentation
            )
            for cell in cell_list
        )

    per_cell_measures_list, per_object_measures_list = zip(*results)
    per_cell_measures_df = pd.DataFrame(per_cell_measures_list)
    per_object_measures_df = pd.concat(per_object_measures_list)

    lipiddroplets_dataset_updated = lipiddroplets_dataset.copy()
    lipiddroplets_dataset_updated = lipiddroplets_dataset_updated.merge(
        per_cell_measures_df, on=[constants.CELLID_COL], how="left"
    )
    lipiddroplets_dataset_updated = lipiddroplets_dataset_updated.merge(
        per_object_measures_df,
        on=["ObjectNumber", "Metadata_CellID"],
        how="left",
    )

    if savedir != "":
        filepath = savedir / f"{constants.LIPID_DROPLETS}.xlsx"
        filename = get_unique_filename(filepath)

        lipiddroplets_dataset_updated.to_excel(
            filename, index=False
        )

    return lipiddroplets_dataset_updated

