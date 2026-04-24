# general modules
from pathlib import Path

# project modules
import constants

import feature_extraction_utils as utils
from feature_extraction_cell import feature_extract_cell, feature_extract_cytoplasm
from feature_extraction_lipiddroplets import feature_extract_lipids
from feature_extraction_mitochondria import feature_extract_mitochondria
from feature_extraction_nucleus import feature_extract_nucleus

# %% IO

treatment = "inflammation"
version = "run5_mito"
# Input: directory containing all the raw xray images, cell outline, .npz probability files
parentdir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
measuredir = parentdir / "cellprofiler"
segmentationdir = parentdir / "results"

# Output: resultsdir
measuredir_updated = measuredir / "measurements_updated"
Path(measuredir_updated).mkdir(parents=True, exist_ok=True)

updated_dataframes = utils.add_CellID_to_cellprofiler_measurements(
    measuredir, constants.ORGANELLE_CP_LIST
)  # add CellID to all measurement dataframes

# load in all binary maps
all_segmentation = utils.load_segmentation(
    segmentationdir, organelle=constants.ORGANELLE_CP_LIST
)
# %% main script

updated_dataframe_measures = {}
# Step 1: Additional cell measurements
updated_dataframe_measures["cell"] = feature_extract_cell(
    updated_dataframes, all_segmentation, savedir=measuredir_updated
)

# Step 2: Additional cytoplasm measurements 
updated_dataframe_measures["cytoplasm"] = feature_extract_cytoplasm(
    updated_dataframes, all_segmentation, savedir=measuredir_updated
)

# Step 3: Additional nuclei measurements
updated_dataframe_measures["nucleus"] = feature_extract_nucleus(
    updated_dataframes, all_segmentation, savedir=measuredir_updated
)

    
# Step 4: Additional lipid droplet measurements
updated_dataframe_measures["lipiddroplets"] = feature_extract_lipids(
    updated_dataframes, all_segmentation, savedir=measuredir_updated, DEBUG = True, batch = (65,67)
)


# Step 5: Additional mitochondria measurements
updated_dataframe_measures["mitochondria"] = feature_extract_mitochondria(
    updated_dataframes, all_segmentation, savedir=measuredir_updated, DEBUG = True, batch = (61,67)
)

#%% batch processing
updated_dataframe_measures = {}

step = 11
max_value = 66

intervals = [(i, i + step) for i in range(0, max_value, step)]

failed_intervals = []

for i in intervals:
    try:
        updated_dataframe_measures["mitochondria"] = feature_extract_mitochondria(
            updated_dataframes,
            all_segmentation,
            savedir=measuredir_updated / "batch_files",
            DEBUG=True,
            batch=i
        )

    except Exception as e:
        print(f"Error occurred in interval {i}")
        print(f"Error message: {e}")
        failed_intervals.append((i, str(e)))
        continue

 #%% combine batch processing files
from utils import load_path_into_dict
import pandas as pd
import constants

allfiles = load_path_into_dict(measuredir_updated / "batch_files", keywordregex = [r"\d{1,2}"], keyword="mitochondria", filetype = "xlsx")

def combine_files(files_dict, id_cols, output = "combined.xlsx", priority = "first"):
    
    frames = [
        pd.read_excel(path).set_index(id_cols)
        for _, path in sorted(files_dict.items())
    ]
    
    full_file = pd.concat(frames).groupby(level=[0,1]).first().reset_index()    
    
    return full_file

full_file = combine_files(allfiles, ["ImageNumber", "ObjectNumber"])

filename = f"{constants.MITOCHONDRIA}_full.xlsx"
full_file.to_excel(
    measuredir_updated / filename, index=False
)