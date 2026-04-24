import segmentation_utils as utils
from skimage.measure import label

from pathlib import Path
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import tifffile as tiff
import numpy as np
from PIL import Image


def is_tiff(file: Path) -> bool:
    return file.is_file() and (file.suffix == ".tif" or file.suffix == ".tiff")


def postprocess_nucleus_nonparallel(probability_files_dir: Path, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    probability_files: list[str] = [
        str(f) for f in probability_files_dir.iterdir() if is_tiff(f)
    ]
    cell_files: list[str] = [
        str(f) for f in results_dir.iterdir() if is_tiff(f) and "cell_binary" in str(f)
    ]

    cellid_pattern = r"CELL\d{3}"
    organelle = "nucleus"
    probmap_nucleus = utils.load_tiff_into_dict(
        probability_files, [cellid_pattern, organelle], padding=50
    )
    binary_cell = utils.load_tiff_into_dict(cell_files, [cellid_pattern], padding=50)

    binary_nucleus = {}
    binary_nucleus_size = {}
    binary_nucleus_filled = {}
    binary_nucleus_internal = {}
    binary_nucleus_final = {}
    binary_nucleus_labelled = {}

    for cellid, probmap in tqdm(
        probmap_nucleus.items(), desc="Processing probability file"
    ):
        binary_nucleus_dict = utils.segmentation_threshold_2D(
            probmap, confidence_thresholds=[0.95]
        )
        binary_nucleus[cellid] = binary_nucleus_dict["threshold" + str(0.95)]

        # Keep only internal structures
        binary_nucleus_internal[cellid], _, _ = utils.keep_internal_rois(
            binary_cell[cellid],
            binary_nucleus[cellid],
            min_overlap=0.5,
            separate_connecting_iter=20,
        )

        # Filter by object size
        binary_nucleus_size[cellid], _ = utils.filter_object_size(
            binary_nucleus_internal[cellid], min_size=500000
        )

        # Fill holes
        binary_nucleus_filled[cellid] = utils.fill_holes(
            binary_nucleus_size[cellid], gap_fill=150
        )

        # Smoothen structures
        binary_nucleus_final[cellid] = utils.smooth_shape_fft(
            binary_nucleus_filled[cellid], FD_retained=0.003
        )

        # Label objects
        binary_nucleus_labelled[cellid] = labelbin.label_binary(
            binary_nucleus_final[cellid]
        )

    for cell, image in binary_nucleus_labelled.items():
        num_labels = image.max()

        if num_labels > 1:
            print(f"Multiple nuclei found in {cell}")

    all_cellid = binary_cell.keys()
    for cell in all_cellid:
        nucleus_layer = f"{cell}_layer0"
        filename = nucleus_layer + "_nucleus_binary.tif"
        tiff.imwrite(
            results_dir / filename,
            binary_nucleus_labelled[cell].astype(np.uint16),
        )

        print(f"finished saving images for {cell}")


def nucleus_processing(
    cellid: str,
    probmap_path: Path,
    cell_path: Path,
    padding=50,
    saveintermediate=None,
    outputdir=None,
    mode="build",
    DEBUG=False,
    threshold_list_full = [0.95, 0.9, 0.8, 0.6, 0.5, 0.25, 0.1, 0.001, 0.0005]

):
    if outputdir is None and saveintermediate is not None:
        outputdir = probmap_path
        print("Saving intermediate images to probability map directory")

    probmap = np.pad(
        tiff.imread(probmap_path), pad_width=padding, mode="constant", constant_values=0
    )
    binary_cell = (
        np.pad(
            tiff.imread(cell_path),
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )
        > 0
    )

    if mode == "build":
        # threshold_list_full = [0.99,0.95,0.9,0.8,0.6,0.1]
        # threshold_list_full = [0.95, 0.9, 0.8, 0.6, 0.5, 0.25, 0.1, 0.001, 0.0005]
        # threshold_list_full = [0.99,0.95,0.1,0.0005]
        # threshold_list_full = [0.9,0.8,0.5,0.0001]
        # threshold_list_full = [0.999,0.95,0.9]
        # threshold_list_full = [0.999,0.95,0.6,0.25, 0.001, 0.000005]
        # threshold_list_full = [0.99,0.9,0.75,0.5,0.25,0.1,0.001]
        # threshold_list_full = [0.99,0.95,0.9,0.85,0.8,0.75,0.5,0.01]
        # threshold_list_full = [0.99,0.95,0.8,0.6,0.25,0.05,0.01,0.001,0.000005]
        # threshold_list_full = [0.95, 0.9, 0.8, 0.6, 0.5, 0.25, 0.1, 0.005, 0.001]

        # threshold_list_full = [0.999,0.95]#,0.95,0.9,0.8,0.6,0.5,0.25,0.1]#,0.001]
        # threshold_list_full = [0.99,0.9,0.5,0.1, 0.001, 0.0005] # CELL033

        threshold_list = threshold_list_full[:-1]
        binary_nucleus_dict = utils.segmentation_threshold_2D(
            probmap, confidence_thresholds=threshold_list_full
        )

        binary_nucleus, _, _ = utils.recover_from_thresholds(
            binary_nucleus_dict, binary_cell, threshold_list, piece="largest"
        )

    elif mode == "threshold":
        binary_nucleus_dict = utils.segmentation_threshold_2D(
            probmap, confidence_thresholds=[0.95]
        )
        binary_nucleus = binary_nucleus_dict["threshold" + str(0.95)]

    # Keep only internal structures
    binary_nucleus_internal, _, _ = utils.keep_internal_rois(
        binary_nucleus,
        binary_cell,
        min_overlap=0.9,  # 0.5,  # 0.8
        separate_connecting_iter=50,  # 10,  # 50
        largest_only=True,
    )

    # Filter by object size
    binary_nucleus_size, _ = utils.filter_object_size(
        binary_nucleus_internal, min_size=500000
    )

    # Fill concavities
    binary_nucleus_concavity_filled_thresh = utils.fill_concavities_thresholding(
        binary_nucleus_size.astype(np.uint8),
        binary_nucleus_dict["threshold" + str(threshold_list_full[-1])],
    )

    binary_nucleus_concavity_filled = utils.fill_concavities(
        binary_nucleus_concavity_filled_thresh,
        depth_threshold=5,
        shallowness=3,
    )
    # Keep only internal structures
    # binary_nucleus_internal_2, _, _ = utils.keep_internal_rois(
    # binary_cell, binary_nucleus, min_overlap=0.8, separate_connecting_iter=50
    # )

    # Fill holes
    binary_nucleus_filled = utils.fill_holes(binary_nucleus_concavity_filled)

    # Smoothen structures
    binary_nucleus_final = utils.smooth_shape_fft(
        binary_nucleus_filled, FD_retained=0.005 #0.005
    )

    if saveintermediate is not None and cellid == saveintermediate:
        utils.save_tiff(
            binary_nucleus, outputdir, f"{cellid}_nucleus_thresholded_py"
        )  # thresholded probability map nucleus
        utils.save_tiff(
            binary_nucleus_internal, outputdir, f"{cellid}_nucleus_noexternal"
        )  # external nuclei removed
        utils.save_tiff(
            binary_nucleus_size, outputdir, f"{cellid}_nucleus_sizefiltered"
        )  # small artifacts removed
        utils.save_tiff(
            binary_nucleus_filled, outputdir, f"{cellid}_nucleus_filled"
        )  # holes filled
        utils.save_tiff(
            binary_nucleus_final, outputdir, f"{cellid}_nucleus_smooth"
        )  # smoothened

    if DEBUG:
        return None

    # Label objects
    return {cellid: label(binary_nucleus_final).astype(np.uint8)}
    # return {cellid: binary_nucleus_size}


def postprocess_nucleus(
    allpaths: dict, padding=50, savedir=None, saveintermediate=None, mode="build"
):
    probmap_dict = allpaths["nucleus"]
    cell_processed = allpaths["cell"]

    all_CellID = list(cell_processed.keys())
    with tqdm_joblib(desc="Processing", total=len(all_CellID)):
        results = Parallel(n_jobs=2)(
            delayed(nucleus_processing)(
                cellid,
                probmap_dict[cellid],
                cell_processed[cellid],
                padding=padding,
                saveintermediate=saveintermediate,
                outputdir=savedir,
                mode=mode,
            )
            for cellid in all_CellID
        )

    # parallel results: [{key:value},{key:value}]
    binary_nucleus_labelled = {k: v for d in results for k, v in d.items()}

    if savedir is not None:
        for cell, image in binary_nucleus_labelled.items():
            labeled = label(image)
            props = regionprops(labeled)

            if len(props) > 1:
                print(f"Multiple nuclei found in {cell}")

                # find largest ROI by area
                largest_prop = max(props, key=lambda x: x.area)

                # create a new binary image with only the largest ROI
                largest_roi_image = np.zeros_like(image, dtype=image.dtype)
                largest_roi_image[labeled == largest_prop.label] = 1

                # replace the image in the dictionary
                binary_nucleus_labelled[cell] = largest_roi_image

        for cell in binary_nucleus_labelled.keys():
            filename = f"{cell}_layer0_nucleus_labelled.tif"
            tiff.imwrite(
                savedir / filename,
                binary_nucleus_labelled[cell].astype(np.uint8),
            )

        print("\nProcessed nuclei segmentation saved as tiff")
        return None  # no need to return dictionary if saving images
    else:
        return binary_nucleus_labelled  # return dictionary if not saving images


# %% test script

from constants import *

if __name__ == "__main__":
    treatment = "control"
    version = "run3"
    # Input: directory containing all the raw xray images, cell outline, .npz probability files
    parentdir = Path(
        rf"C:/Users/IvyWork/Desktop/projects/dataset/input_{treatment}/{version}"
    )
    # Output: resultsdir
    resultsdir = parentdir / "results"
    Path(resultsdir).mkdir(parents=True, exist_ok=True)

    keywords = ["xray", "cell", "lipiddroplets", "mitochondria", "nucleus", "au"]
    allpaths = {}
    for key in keywords:
        allpaths[key] = utils.load_path_into_dict(parentdir / "input", keyword=key)

    # Step 3: process organelles
    organelles = [
        "xray",
        "cell",
    ]  # read in processed xray and cell paths for organelle processing

    allpaths_processed = {}
    for organelle in organelles:
        allpaths_processed[organelle] = utils.load_path_into_dict(
            resultsdir, keyword=organelle
        )

    allpaths = {
        outer_k: {
            inner_k: [inner_v]
            for inner_k, inner_v in outer_v.items()
            if inner_k in ["CELL010"]
        }
        for outer_k, outer_v in allpaths.items()
    }

    allpaths_processed = {
        outer_k: {
            inner_k: [inner_v]
            for inner_k, inner_v in outer_v.items()
            if inner_k in ["CELL010"]
        }
        for outer_k, outer_v in allpaths_processed.items()
    }

    binary_nucleus_labelled = postprocess_nucleus(
        allpaths, allpaths_processed, padding=50, savedir=thesis_figs
    )
