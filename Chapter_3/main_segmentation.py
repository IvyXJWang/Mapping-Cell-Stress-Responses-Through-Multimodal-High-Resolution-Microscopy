# temp main script - segmentation
import utils as utils
import segmentation_utils as seg
from segmentation_xray import postprocess_xray
from segmentation_cell import postprocess_cell
from segmentation_lipiddroplets import postprocess_lipiddroplets
from segmentation_nucleus import postprocess_nucleus
from segmentation_mitochondria import postprocess_mitochondria
from segmentation_cytoplasm import postprocess_cytoplasm
import constants

from pathlib import Path
from tqdm import tqdm
import tifffile as tiff
import gc

# %% io
treatment = "control"
version = "run4"
# Input: directory containing all the raw xray images, cell outline, .npz probability files
parentdir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
inputdir = parentdir / "input"
# Output: resultsdir
resultsdir = parentdir / "results"
Path(resultsdir).mkdir(parents=True, exist_ok=True)
utils.rename_files(str(inputdir), "_0000", "_xray")

# %% Process probability maps
feature_list = ["au", "outline", "nucleus", "mitochondria", "lipiddroplets"]
seg.npz_to_tiff(inputdir / "probability_maps_npz", inputdir, feature_list)


# %% load in extracted probability files
keywords = ["xray", "cell", "lipiddroplets", "mitochondria", "nucleus", "au"]   
keywords = ["mitochondria", "cell"]
allpaths = {}
for key in keywords:
    allpaths[key] = seg.load_path_into_dict(parentdir / "input", keyword=key)
allpaths_cells = list(allpaths[list(allpaths.keys())[0]].keys())

allpaths_processed = {}
for key in keywords:
    allpaths_processed[key] = seg.load_path_into_dict(parentdir / "results", keyword=key)   
allpaths_processed_cells = list(allpaths_processed[list(allpaths_processed.keys())[0]].keys())

#%%
filtered_dict = {
    k: v
    for k, v in dict_a.items()
    if k in dict_b
}
# %% subset processing

allpaths_cells = list(allpaths[list(allpaths.keys())[0]].keys())
selected_cell = ""
selected_indices = None

if selected_cell in allpaths_cells:
    idx = allpaths_cells.index(selected_cell)
    allpaths_processing = utils.filter_dictionary_subset(
        allpaths, {k for k in allpaths_cells[idx : idx + 1]}
    )
elif selected_indices is not None:
    allpaths_processing = utils.filter_dictionary_subset(
    allpaths,
    {allpaths_cells[i] for i in selected_indices}
)
else:
    allpaths_processing = utils.filter_dictionary_subset(
        allpaths, {k for k in allpaths_cells[50:76]}
    )

# %% Lipid droplet postprocessing
postprocess_lipiddroplets(
    allpaths, savedir=resultsdir, saveintermediate="Au"
)  # lipid droplet processing should happen before erasing debris (artificially keeps Au beads that have been corrected)

# %% Nucleus postprocessing
postprocess_nucleus(
    allpaths_processing, savedir=resultsdir, saveintermediate=None, mode="build"
)

# %%

postprocess_mitochondria(allpaths_processing, savedir=resultsdir)

# %% Mitochondria postprocessing
allpaths_cells = list(allpaths[list(allpaths.keys())[0]].keys())

for cell in allpaths_cells[22:-1]:
    allpaths_processing = utils.filter_dictionary_subset(allpaths, {k for k in [cell]})
    postprocess_mitochondria(allpaths_processing, savedir=resultsdir)

# mito_output = postprocess_mitochondria(allpaths_processing, DEBUG=True)
# %% Cell outline postprocessing
postprocess_cell(allpaths, savedir=resultsdir)

# %% Xray postprocessing
postprocess_xray(allpaths_processing, savedir=resultsdir, intermediate_save=False)

# %% Cytoplasm postprocessing
organelles = [
    "cell",
    "au",
    "nucleus",
    "lipiddroplets",
    "combined",
]  # read in processed organelle paths for cytoplasm processing
allpaths_processed = {}
for organelle in organelles:
    allpaths_processed[organelle] = seg.load_path_into_dict(
        resultsdir, keyword=organelle
    )

postprocess_cytoplasm(
    allpaths_processed, ["lipiddroplets", "nucleus", "combined", "au"], savedir=resultsdir
)
# %% Display postprocessing results

# combine mitochondria layers into single layer
from data_analysis_figs import plot_overlays_masks_cell
from feature_extraction_utils import collapse_multilayer
import numpy as np


def combine_img_by_filename(directory, keywords: list, savedir=None):
    """
    takes all files in directory and combines all files containing all keywords
    """
    img_list = []
    for file in directory.iterdir():
        if all(keyword in file.name for keyword in keywords):
            img_list.append(tiff.imread(file))

    if not img_list:
        raise ValueError("No files matching keywords")

    combined = collapse_multilayer(img_list)

    if savedir is not None:
        savedir.mkdir(parents=True, exist_ok=True)

        combined_name = f"{keywords[0]}"
        filename = combined_name + "_layer0_combined.tif"
        tiff.imwrite(
            savedir / filename,
            combined.astype(np.uint8),
        )

    return combined


for cell in allpaths_processed_cells:

    mito_combined = combine_img_by_filename(
        resultsdir, [cell, "mitochondria"], savedir=resultsdir
    )
print("Combined images saved as tiff")


# %% plot segmentation overlay
import numpy as np
from data_analysis_figs import plot_overlays_masks_cell

organelle = constants.NUCLEUS
keywords = ["xray", "cell", "lipiddroplets", "mitochondria", "nucleus", "au"]
allpaths = {}
for key in keywords:
    allpaths[key] = seg.load_path_into_dict(parentdir / "results", keyword=key)
allpaths_cells = list(allpaths[list(allpaths.keys())[0]].keys())
#%%
cells_plot = allpaths_cells
cells_plot = ["CELL001"]

resultsdir_1 = parentdir / "results"
resultsdir_2 = (
    parentdir
    / "results_intermediate_files"
    / "mito3_adjustskelthreshold_multithresholdingsoft"
)

for cellid in cells_plot:
    i = 0
    obj_labelled = {}
    while True:
        obj_path = resultsdir_1 / f"{cellid}_layer{i}_{organelle}_labelled.tif"
        if obj_path.exists():
            obj_labelled[f"Layer {i}"] = np.asarray(tiff.imread(obj_path))

            i += 1

        else:
            break

    xray_img = np.asarray(
        tiff.imread(resultsdir / f"{cellid}_layer0_xray_processed.tiff")
    )

    fig, ax = plot_overlays_masks_cell(
        base_img=xray_img,
        outlines=obj_labelled,
        # title=f"{cellid} mito1 single thresholding",
        title=f"{cellid} Constant Threshold (0.95)",
        show_base_alone=True,
        legend=False,
    )

# %%
# Display postprocessing results (run2)
organelle = "nucleus"
for cellid in cells_plot:
    i = 0
    obj_labelled = {}
    while True:
        obj_path = resultsdir_2 / f"{cellid}_layer{i}_{organelle}_labelled.tif"
        if obj_path.exists():
            obj_labelled[f"Layer {i}"] = np.asarray(tiff.imread(obj_path))

            i += 1

        else:
            break

    xray_img = np.asarray(
        tiff.imread(resultsdir / f"{cellid}_layer0_xray_processed.tiff")
    )

    fig, ax = plot_overlays_masks_cell(
        base_img=xray_img,
        outlines=obj_labelled,
        # title=f"{cellid} mito3 multithresholding",
        title="Adaptive Multithresholding",
        show_base_alone=False,
        legend=False,
    )


# %% memory debugging

import psutil, os

proc = psutil.Process(os.getpid())
from operator import itemgetter

getter = itemgetter(3, 4, 5, 6)

# clean images
for cell in tqdm(allpaths["xray"].keys()):
    # for cell in tqdm(list(getter(allpaths_cells))):  # subset for debugging
    xray_clean, debris_mask = clean_img(
        allpaths["xray"][cell],
        allpaths["au"][cell],
        allpaths["lipiddroplets"][cell],
        zscore_thresh=3.0,
        convert_img=8,
        boundary=allpaths["cell"][cell],
    )

    # print(f"iter {i}: RSS = {proc.memory_info().rss / (1024**3):.3f} GiB")

    filename_clean = cell + "_layer0_xray_clean.tiff"
    filename_debris = cell + "_layer0_xray_debris.tiff"

    intermediate_outdir = resultsdir / "debris_intermediates"
    intermediate_outdir.mkdir(parents=True, exist_ok=True)

    tiff.imwrite(
        intermediate_outdir / filename_clean,
        xray_clean,
        compression=None,
    )

    tiff.imwrite(
        intermediate_outdir / filename_debris,
        debris_mask,
        compression=None,
    )

    del xray_clean, debris_mask
    gc.collect()


# %%
# Step 2: process cell outline
postprocess_cell(
    allpaths, savedir=resultsdir
)  # no output - save tiff to results folder

# Step 3: process organelles
organelles = [
    "xray",
    "cell",
]  # read in processed xray and cell paths for organelle processing
allpaths_processed = {}
for organelle in organelles:
    allpaths_processed[organelle] = seg.load_path_into_dict(
        resultsdir, keyword=organelle
    )


# Step 4: process cytoplasm
organelles = [
    "xray",
    "cell",
    "mitochondria",
    "nucleus",
    "lipiddroplets",
]  # read in processed organelle paths for cytoplasm processing
allpaths_processed = {}
for organelle in organelles:
    allpaths_processed[organelle] = seg.load_path_into_dict(
        resultsdir, keyword=organelle
    )

postprocess_cytoplasm(
    allpaths_processed, ["lipiddroplets", "nucleus", "mitochondria"], savedir=resultsdir
)
