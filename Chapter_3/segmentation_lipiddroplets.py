from matplotlib.pylab import f

import segmentation_utils as utils

from pathlib import Path
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops_table, label, regionprops
import numpy as np
import tifffile as tiff

import constants


def keep_internal_rois(binary_cell, binary_object, min_overlap=0.5):
    """
    Input
        binary_cell (2D array): binary image of cell boundary
        binary_object (2D array): binary image of objects
        min_overlap (int): fraction overlap threshold
    Output
        internal_objects (2D array): binary image with only regions with specified overlap within boundary
        kept_count (int): number of rois kept
        del_count (int): number of rois deleted

    """
    binary_cell = binary_cell.astype(bool)
    labeled_objects = label(binary_object)
    internal_objects = np.zeros_like(binary_object, dtype=bool)
    external_objects = np.zeros_like(binary_object, dtype=bool)

    for region in regionprops(labeled_objects):
        coords = tuple(region.coords.T)  # rows, cols
        area = region.area
        overlap = np.sum(binary_cell[coords])  # count overlapping pixels
        overlap_fraction = overlap / area

        if overlap_fraction >= min_overlap:
            internal_objects[coords] = True
        else:
            external_objects[coords] = True

    # count how many objects internal vs external in FOV
    labeled_kept = label(internal_objects)

    kept_count = labeled_kept.max()  # max label number corresponds to maximum rois
    total_count = labeled_objects.max()
    del_count = total_count - kept_count

    return internal_objects, external_objects, kept_count, del_count


def remove_Au(
    ref_img,
    binary_cell,
    binary_object,
    threshold=0.3,
    removebg=True,
    apply_watershed=True,
    DEBUG=False,
):
    """
    Input
        ref_img (2D array) : original grayscale image
        binary_cell (2D array): binary image of cell outline
        binary_object (2D array): binary image of roi
        threshold (int) : Au intensity threshold (if minintensityROI<threshold*meanpixel = Au)
        removebg (bool) : remove rois outside cell outline
    Output
        roi_mask (2D array): returns binary image with only roi with greater intensity than threshold
        roi_mask_deleted (2D array): returns binary image with only roi with lower intensity than
        threshold
        kept_count (int): number of rois kept
        del_count (int): number of rois deleted
    """

    masked_values = ref_img[binary_cell.astype(bool)]
    pixel_avg = masked_values.mean()

    if apply_watershed:
        distance = ndi.distance_transform_edt(binary_object)
        coordinates = peak_local_max(distance, labels=binary_object)
        markers = np.zeros_like(distance, dtype=int)
        for i, coord in enumerate(coordinates, 1):
            markers[tuple(coord)] = i

        binary_object = watershed(-distance, markers, mask=binary_object)

    # Label particles
    labeled_objects = label(binary_object)
    roi_properties = regionprops_table(
        labeled_objects,
        intensity_image=ref_img,
        properties=["label", "area", "min_intensity"],
    )

    roi_mask = np.zeros_like(labeled_objects, dtype=bool)  # initialize kept object mask
    roi_mask_deleted = np.zeros_like(
        labeled_objects, dtype=bool
    )  # initialize deleted object mask

    for label_id, area, min_intensity in zip(
        roi_properties["label"], roi_properties["area"], roi_properties["min_intensity"]
    ):
        if label_id == 0:
            continue  # Skip background

        mask = labeled_objects == label_id
        if min_intensity < threshold * pixel_avg:
            roi_mask_deleted[mask] = True
        else:
            roi_mask[mask] = True

    labeled_kept = label(roi_mask)
    labeled_del = label(roi_mask_deleted)

    kept_count = labeled_kept.max()  # max label number corresponds to maximum rois
    del_count = labeled_del.max()

    if removebg:
        roi_mask, external_objects, kept_count_internal, del_count_internal = keep_internal_rois(
            binary_cell, roi_mask, min_overlap=1
        )
        kept_count = kept_count_internal
        del_count = del_count + del_count_internal

    if DEBUG:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2)

        axs[0].plot(roi_mask, cmap="gray")
        axs[0].set_title(f"{kept_count} rois kept")
        axs[0].axis("off")

        axs[1].plot(roi_mask, cmap="gray")
        axs[1].set_title(f"{del_count} rois removed")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()

    return roi_mask, roi_mask_deleted, external_objects, kept_count, del_count


def lipiddroplet_processing(
    cellid: str,
    probmap_path: Path,
    cell_path: Path,
    xray_path: Path,
    Au_path: Path = None,
    padding=50,
    saveintermediate=False,
    outputdir=None,
    DEBUG=False,
):
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
    xray_cell = np.pad(tiff.imread(xray_path), pad_width=padding, mode="edge")
    if Au_path is not None:
        probmap_Au = np.pad(
            tiff.imread(Au_path), pad_width=padding, mode="constant", constant_values=0
        )
    else:
        probmap_Au = np.zeros_like(binary_cell)

    # Extract segmentation above confidence threshold
    binary_lipid_dict = utils.segmentation_threshold_2D(
        probmap, confidence_thresholds=[0.005]
    )
    binary_lipiddroplets = binary_lipid_dict["threshold" + str(0.005)]
    
    binary_Au_dict = utils.segmentation_threshold_2D(
        probmap_Au, confidence_thresholds=[0.1]
    )
    binary_Au = binary_Au_dict["threshold" + str(0.1)]

    # Fill holes
    binary_lipid_filled = utils.fill_holes(binary_lipiddroplets, gap_fill=False)

    # Filter by object size
    binary_lipiddroplets_size, lipiddroplet_sizes = utils.filter_object_size(
        binary_lipid_filled, min_size=200
    )

    if DEBUG:
        return binary_lipiddroplets_size

    # Keep only internal structures and remove Au
    binary_lipiddroplets_internal, binary_lipid_deleted, external_objects, _, _ = remove_Au(
        xray_cell,
        binary_cell,
        binary_lipiddroplets_size,
        threshold=0.3,
        apply_watershed=True,
    )

    binary_Au_full = np.logical_or(binary_Au, binary_lipid_deleted)

    # Filter by object size
    binary_lipiddroplets_final, lipiddroplet_sizes_internal = utils.filter_object_size(
        binary_lipiddroplets_internal, min_size=200
    )

    if saveintermediate:
        if outputdir is None:
            outputdir = probmap_path.parent / "lipiddroplet_processing_intermediates"
            Path(outputdir).mkdir(parents=True, exist_ok=True)

        utils.save_tiff(binary_lipid_deleted, outputdir, f"{cellid}_detected_Au")  # Au mask
        utils.save_tiff(external_objects, outputdir, f"{cellid}_external_ld")  # External lipid droplets

        utils.save_tiff(
            binary_lipiddroplets_internal,
            outputdir,
            f"{cellid}_lipiddroplets_noexternal_noAu",
        )  # external lipid droplets removed
        utils.save_tiff(
            binary_lipiddroplets_size, outputdir, f"{cellid}_lipiddroplets_sizefiltered"
        )  # small artifacts removed
        utils.save_tiff(
            binary_lipid_filled, outputdir, f"{cellid}_lipiddroplets_filled"
        )  # holes filled

    # Label objects
    return (
        {cellid: label(binary_lipiddroplets_final).astype(np.uint16)},
        {cellid: binary_Au_full.astype(np.uint8)},
        {cellid: external_objects.astype(np.uint8)},
    )


def postprocess_lipiddroplets(
    allpaths: dict,
    padding=50,
    savedir=None,
    saveintermediate=None,
):
    probmap_dict = allpaths[constants.LIPID_DROPLETS]
    cell_processed = allpaths[constants.CELL]
    xray_processed = allpaths[
        "xray"
    ]  # use original xray images to determine outliers!!
    Au_dict = allpaths[constants.AU]

    all_CellID = list(cell_processed.keys())
    with tqdm_joblib(desc="Processing", total=len(all_CellID)):
        results = Parallel(n_jobs=-1)(
            delayed(lipiddroplet_processing)(
                cellid,
                probmap_dict[cellid],
                cell_processed[cellid],
                xray_processed[cellid],
                Au_path=Au_dict[cellid],
                padding=padding,
                saveintermediate=None,
                outputdir=savedir,
            )
            for cellid in all_CellID
        )

    # parallel results: [{key:value},{key:value}]
    # results is list of (label_dict, deleted_dict)
    labelled_list, deleted_list = zip(*results)  # each is a tuple of dicts
    binary_lipiddroplets_labelled = {k: v for d in labelled_list for k, v in d.items()}
    au_mask = {k: v for d in deleted_list for k, v in d.items()}

    if saveintermediate == "Au":
        for cellid, au_mask_cell in au_mask.items():
            filename_Au = f"{cellid}_layer0" + f"_{constants.AU}_labelled"
            utils.save_tiff(
                au_mask_cell,
                savedir,
                filename_Au,
            )  # Au removed

    if savedir is not None:
        for cellid, lipiddroplet_mask in binary_lipiddroplets_labelled.items():
            filename = f"{cellid}_layer0_{constants.LIPID_DROPLETS}_labelled.tiff"
            if (
                lipiddroplet_mask.max() < 255
            ):  # save as smallest file type while preserving labels
                tiff.imwrite(
                    savedir / filename,
                    lipiddroplet_mask.astype(np.uint8),
                )
            else:
                tiff.imwrite(
                    savedir / filename,
                    lipiddroplet_mask.astype(np.uint16),
                )

        print("\nProcessed lipid droplet segmentation saved as tiff")

    else:
        return binary_lipiddroplets_labelled


# %% test script

import constants

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
            if inner_k in ["CELL049"]
        }
        for outer_k, outer_v in allpaths.items()
    }

    allpaths_processed = {
        outer_k: {
            inner_k: [inner_v]
            for inner_k, inner_v in outer_v.items()
            if inner_k in ["CELL049"]
        }
        for outer_k, outer_v in allpaths_processed.items()
    }

    binary_nucleus_labelled = postprocess_lipiddroplets(
        allpaths,
        allpaths_processed,
        padding=50,
        savedir=thesis_figs,
        saveintermediate="CELL049",
    )
