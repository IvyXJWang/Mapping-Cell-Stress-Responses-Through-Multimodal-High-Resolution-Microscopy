import pandas as pd
import feature_extraction_utils as utils
from tqdm import tqdm
import pathlib as Path
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# project modules
import constants


def single_cell_measurements_cytoplasm(
    cell, updated_dataframes, all_segmentation, DEBUG=False
):
    cytoplasm_dataset = updated_dataframes[constants.CYTOPLASM]
    subset_cytoplasm = cytoplasm_dataset[
        cytoplasm_dataset[constants.CELLID_COL] == cell
    ].reset_index()

    # add measures to dataframe
    dataset_calcs = pd.DataFrame(
        {
            constants.CELLID_COL: cell,
            "Intensity_Range_xray": (
                subset_cytoplasm["Intensity_MaxIntensity_xray"]
                - subset_cytoplasm["Intensity_MinIntensity_xray"]
            ),
            "Intensity_RangeEdge_xray": (
                subset_cytoplasm["Intensity_MaxIntensityEdge_xray"]
                - subset_cytoplasm["Intensity_MinIntensityEdge_xray"]
            ),
        }
    )

    dataset_avg = utils.texture_measure_average(subset_cytoplasm)
    per_cell_measures = pd.concat([dataset_calcs, dataset_avg], axis=1)

    return per_cell_measures


def single_cell_measurements_cell(
    cell, updated_dataframes, all_segmentation, DEBUG=False
):
    mitochondria_segmentation = all_segmentation[constants.MITOCHONDRIA]
    lipiddroplet_segmentation = all_segmentation[constants.LIPID_DROPLETS]
    cell_segmentation = all_segmentation[constants.CELL]

    cell_dataset = updated_dataframes[constants.CELL]
    lipiddroplet_dataset = updated_dataframes[constants.LIPID_DROPLETS]
    mitochondria_dataset = updated_dataframes[constants.MITOCHONDRIA]
    cytoplasm_dataset = updated_dataframes[constants.CYTOPLASM]

    subset_cell = cell_dataset[cell_dataset[constants.CELLID_COL] == cell].reset_index()
    subset_lipiddroplet = lipiddroplet_dataset[
        lipiddroplet_dataset[constants.CELLID_COL] == cell
    ].reset_index()
    subset_mitochondria = mitochondria_dataset[
        mitochondria_dataset[constants.CELLID_COL] == cell
    ].reset_index()
    subset_cytoplasm = cytoplasm_dataset[
        cytoplasm_dataset[constants.CELLID_COL] == cell]

    subset_mito_images_list = list(
        mitochondria_segmentation.loc[
            mitochondria_segmentation[constants.CELLID_COL] == cell, "image"
        ].values
    )
    collapsed_mito_segmentation = utils.collapse_multilayer(subset_mito_images_list)
    lipiddroplet_cell_segmentation = lipiddroplet_segmentation[cell] > 0

    area_cytoplasm = subset_cytoplasm["AreaShape_Area"].values[0]
    area_cell = subset_cell["AreaShape_Area"].values[0]

    # calculations
    overlap_area = utils.image_overlap(
        collapsed_mito_segmentation, lipiddroplet_cell_segmentation
    )
    lipiddroplet_num = len(subset_lipiddroplet)
    mitochondria_num = len(subset_mitochondria)

    convex_props = utils.convexhull_props_singleroi(cell_segmentation[cell])

    # add measures to dataframe
    dataset_calcs = pd.DataFrame(
        {
            constants.CELLID_COL: cell,
            "Structure_AreaOccupiedObjects": (area_cell - area_cytoplasm) / area_cell,
            "Structure_LipidOverlapMitoPercent": overlap_area
            / lipiddroplet_cell_segmentation.sum(),  # overlap area over total lipid droplet area
            "Structure_LipidMitoRatio": lipiddroplet_num / mitochondria_num,
            "AreaShape_AspectRatio": (
                subset_cell["AreaShape_MajorAxisLength"]
                / subset_cell["AreaShape_MinorAxisLength"]
            ),
            "AreaShape_FormFactorRatio": (
                subset_cell["AreaShape_FormFactor"] / convex_props["hull_circularity"]
            ),
            "AreaShape_ConvexFormFactor": convex_props["hull_circularity"].values[0],
            "Intensity_Range_xray": (
                subset_cell["Intensity_MaxIntensity_xray"]
                - subset_cell["Intensity_MinIntensity_xray"]
            ),
            "Intensity_RangeEdge_xray": (
                subset_cell["Intensity_MaxIntensityEdge_xray"]
                - subset_cell["Intensity_MinIntensityEdge_xray"]
            ),
        }
    )

    dataset_avg = utils.texture_measure_average(subset_cell)
    per_cell_measures = pd.concat([dataset_calcs, dataset_avg], axis=1)

    return per_cell_measures


def feature_extract_cell(
    updated_dataframes: dict,
    all_segmentation: dict,
    savedir: Path = "",
    DEBUG=False,
):
    cell_dataset = updated_dataframes["cell"]

    cell_list = list(cell_dataset[constants.CELLID_COL])
    with tqdm_joblib(desc="Processing", total=len(cell_list)):
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(single_cell_measurements_cell)(
                cell, updated_dataframes, all_segmentation
            )
            for cell in cell_list
        )

    if DEBUG:
        return results

    per_cell_measures_df = pd.concat(results)

    cell_dataset_updated = cell_dataset.copy()
    cell_dataset_updated = cell_dataset_updated.merge(
        per_cell_measures_df, on=[constants.CELLID_COL], how="left"
    )

    if savedir != "":
        cell_dataset_updated.to_excel(savedir / f"{constants.CELL}.xlsx", index=False)

    return cell_dataset_updated


def feature_extract_cytoplasm(
    updated_dataframes: dict,
    all_segmentation: dict,
    savedir: Path = "",
    DEBUG=False,
):
    cytoplasm_dataset = updated_dataframes["cytoplasm"]

    cell_list = list(cytoplasm_dataset[constants.CELLID_COL])
    with tqdm_joblib(desc="Processing", total=len(cell_list)):
        results = Parallel(n_jobs=-1, backend="threading")(
            delayed(single_cell_measurements_cytoplasm)(
                cell, updated_dataframes, all_segmentation
            )
            for cell in cell_list
        )

    if DEBUG:
        return results

    per_cytoplasm_measures_df = pd.concat(results)

    cytoplasm_dataset_updated = cytoplasm_dataset.copy()
    cytoplasm_dataset_updated = cytoplasm_dataset_updated.merge(
        per_cytoplasm_measures_df, on=[constants.CELLID_COL], how="left"
    )

    if savedir != "":
        cytoplasm_dataset_updated.to_excel(
            savedir / f"{constants.CYTOPLASM}.xlsx", index=False
        )

    return cytoplasm_dataset_updated
