
from pathlib import Path
import pandas as pd

import constants
import data_analysis_utils as utils
import data_analysis_organelle_utils as org
from feature_extraction_utils import combine_files_to_sheets, load_segmentation, collapse_multilayer, mitochondria_region_obj

from utils import rename_df_columns_with_keyword
import numpy as np
#%% function definition

from scipy.ndimage import distance_transform_edt
# not actual calculation of dsitance - no subpixel accuracy
def fast_distance(point, binary_mask):
    # Invert mask: object = 0, background = 1
    inverted = 1 - binary_mask
    
    # Distance to nearest object pixel
    dist_map = distance_transform_edt(inverted)
    
    x, y = int(point[0]), int(point[1])  
    return dist_map[x, y]

def point_to_binary_object_distance(point, binary_mask):
    px, py = point
    
    # Get coordinates of object pixels (value = 1)
    object_points = np.argwhere(binary_mask == 1)
    
    if len(object_points) == 0:
        return None  # no object
    
    # Compute distances
    distances = np.sqrt((object_points[:, 0] - px)**2 + 
                        (object_points[:, 1] - py)**2)
    
    return np.min(distances)
#%%
run = "run4"
treatment = "inflammation"
datadir = constants.PARENTDIR / f"{treatment}/{run}"
segmentationdir = datadir / "results"

filedir = datadir / "cellprofiler" / "measurements_updated"
all_sheets_original = combine_files_to_sheets(filedir)

all_sheets = all_sheets_original.copy()
all_sheets[constants.MITOCHONDRIA] = utils.update_cellID_objectnum(
    all_sheets[constants.MITOCHONDRIA],
    "ImageNumber",
    "ObjectNumber",
    "Metadata_CellID",
)

all_sheets[constants.LIPID_DROPLETS] = utils.update_cellID_objectnum(
    all_sheets[constants.LIPID_DROPLETS],
    "ImageNumber",
    "ObjectNumber",
    "Metadata_CellID",
)

all_segmentation = load_segmentation(
    segmentationdir, organelle=constants.ORGANELLE_CP_LIST
)

#%% calculate perimitochondrial density and proportion
lipid_df = all_sheets["lipiddroplets"]

lipiddroplets_segmentation = all_segmentation[constants.LIPID_DROPLETS]
mitochondria_segmentation = all_segmentation[constants.MITOCHONDRIA]
mito_masks = {
    cell_id: (collapse_multilayer(group["image"].values) > 0).astype(np.uint8)
    for cell_id, group in mitochondria_segmentation.groupby("Metadata_CellID")
}

def compute_perimitochondrial_density(row):
    mito_obj_number, mito_area, _ = mitochondria_region_obj(
        mito_masks[row.Metadata_CellID], lipiddroplets_segmentation[row.Metadata_CellID], radius = 30
    )

    return mito_obj_number / mito_area

def compute_perimitochondrial_proportion(row):
    mito_obj_number, mito_area, _ = mitochondria_region_obj(
        mito_masks[row.Metadata_CellID], lipiddroplets_segmentation[row.Metadata_CellID], radius = 30
    )
    
    lipid_num = lipiddroplets_segmentation[row.Metadata_CellID].max()
    
    return mito_obj_number / lipid_num

lipid_df["Structure_PerimitochondrialObjectDensity"] = lipid_df.apply(compute_perimitochondrial_density, axis=1)
lipid_df["Structure_PerimitochondrialObjectProportion"] = lipid_df.apply(compute_perimitochondrial_proportion, axis=1)

#%%
import numpy as np

lipid_df = all_sheets["lipiddroplets"].copy()

lipiddroplets_segmentation = all_segmentation[constants.LIPID_DROPLETS]
mitochondria_segmentation = all_segmentation[constants.MITOCHONDRIA]

mito_masks = {
    cell_id: (collapse_multilayer(group["image"].values) > 0).astype(np.uint8)
    for cell_id, group in mitochondria_segmentation.groupby("Metadata_CellID")
}

density_by_cell = {}
proportion_by_cell = {}

for cell_id in lipid_df["Metadata_CellID"].unique():
    mito_mask = mito_masks[cell_id]
    lipid_seg = lipiddroplets_segmentation[cell_id]

    mito_obj_number, mito_area, _ = mitochondria_region_obj(
        mito_mask, lipid_seg, radius=30
    )

    lipid_num = lipid_seg.max()

    density_by_cell[cell_id] = mito_obj_number / mito_area if mito_area else np.nan
    proportion_by_cell[cell_id] = (
        mito_obj_number / lipid_num if lipid_num else np.nan
    )

lipid_df["Structure_PerimitochondrialObjectDensity"] = lipid_df["Metadata_CellID"].map(density_by_cell)
lipid_df["Structure_PerimitochondrialObjectProportion"] = lipid_df["Metadata_CellID"].map(proportion_by_cell)
#%%
lipid_df.to_excel(
            filedir / "lipiddroplets.xlsx", index=False
        )

#%% 

mito_df = all_sheets["mitochondria"]

cols = [
    "Structure_BranchType0",
    "Structure_BranchType1",
    "Structure_BranchType2",
    "Structure_BranchType3"
]

# Row-wise total
total = mito_df[cols].sum(axis=1)

# Divide each column by the row total
mito_df[cols] = mito_df[cols].div(total, axis=0)

mito_df.to_excel(
            filedir / "mitochondria.xlsx", index=False
        )
#%% recalculations bc i am an idiot :(
mitochondria_df = all_sheets["mitochondria"]
mitochondria_df["CellID"] = all_sheets_original["mitochondria"]["Metadata_CellID"]

row = mitochondria_df.iloc[10]
centroid = (row.AreaShape_Center_X, row.AreaShape_Center_Y)
nucleus_segmentation =  all_segmentation["nucleus"][row.CellID]

distance = point_to_binary_object_distance(centroid, nucleus_segmentation)

#%% recalculate all nucleus distances (mitochondria)

nucleus_dict = all_segmentation["nucleus"]

def compute_distance(row):
    centroid = (row.AreaShape_Center_X, row.AreaShape_Center_Y)
    nucleus_segmentation = nucleus_dict[row.CellID]
    
    return point_to_binary_object_distance(centroid, nucleus_segmentation)

mitochondria_df["Structure_DistanceNucleus"] = mitochondria_df.apply(compute_distance, axis=1)

mitochondria_df.to_excel(
            filedir / "mitochondria.xlsx", index=False
        )

#%% do the same for lipid droplets and add distance from nearest mitochondria (collapsed mask)
lipid_df = all_sheets["lipiddroplets"]
lipid_df["CellID"] = all_sheets_original["lipiddroplets"]["Metadata_CellID"]

row = lipid_df.iloc[1]
centroid = (row.AreaShape_Center_X, row.AreaShape_Center_Y)
nucleus_segmentation =  all_segmentation["nucleus"][row.CellID]
mitochondria_segmentation = all_segmentation["mitochondria"]
img_list = list(
            mitochondria_segmentation
            .loc[
                mitochondria_segmentation["Metadata_CellID"] == row.CellID,
                "image",
            ]
            .values
        )

mitochondria_mask = (collapse_multilayer(img_list) > 0).astype(np.uint8)

distance_nuc = point_to_binary_object_distance(centroid, nucleus_segmentation)
distance_mito = point_to_binary_object_distance(centroid, mitochondria_mask)

#%% recalculate all nucleus distances (lipid droplets)
lipid_df = all_sheets["lipiddroplets"]
lipid_df["CellID"] = all_sheets_original["lipiddroplets"]["Metadata_CellID"]

nucleus_dict = all_segmentation["nucleus"]
mitochondria_segmentation = all_segmentation["mitochondria"]
mito_masks = {
    cell_id: (collapse_multilayer(group["image"].values) > 0).astype(np.uint8)
    for cell_id, group in mitochondria_segmentation.groupby("Metadata_CellID")
}
    
def compute_distance_nucleus(row):
    centroid = (row.AreaShape_Center_X, row.AreaShape_Center_Y)
    nucleus_segmentation = nucleus_dict[row.CellID]
    
    return point_to_binary_object_distance(centroid, nucleus_segmentation)

def compute_distance_mitochondria(row):
    centroid = (row.AreaShape_Center_X, row.AreaShape_Center_Y)
    mitochondria_segmentation = mito_masks[row.CellID]
    
    return point_to_binary_object_distance(centroid, mitochondria_segmentation)

lipid_df["Structure_DistanceNucleus"] = lipid_df.apply(compute_distance_nucleus, axis=1)
lipid_df["Structure_DistanceMitochondria"] = lipid_df.apply(compute_distance_mitochondria, axis=1)

lipid_df.to_excel(
            filedir / "lipiddroplets.xlsx", index=False
        )
