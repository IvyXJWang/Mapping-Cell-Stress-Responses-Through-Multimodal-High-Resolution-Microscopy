# xray image processing functions

# helper functions
import segmentation_utils as utils
import feature_extraction_utils as feu
import tifffile as tiff
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path

from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from segmentation_lipiddroplets import remove_Au
from skimage import exposure
from skimage import img_as_float
import matplotlib.pyplot as plt
import cv2
from patchmatch_cython import inpaint_pyramid
from skimage.measure import label, regionprops
from typing import Iterable, Optional


def image_equalize_histogram(image: np.ndarray, plot=False) -> np.ndarray:
    """
    Apply histogram equalization to the input image.

    Parameters:
    image (np.ndarray): Input image array.

    Returns:
    np.ndarray: Histogram equalized image.
    """
    # Check image type
    # Convert image to float
    image_float = img_as_float(image)

    # Apply histogram equalization
    equalized_image = exposure.equalize_hist(image)

    if plot:
        # Plot original and equalized images with their histograms
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        ax[0].imshow(image_float, cmap="gray")
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(equalized_image, cmap="gray")
        ax[1].set_title("Equalized Image")
        ax[1].axis("off")

        ax[2].hist(image_float.ravel(), bins=256)
        ax[2].set_title("Original Histogram")

        ax[3].hist(equalized_image.ravel(), bins=256)
        ax[3].set_title("Equalized Histogram")

        plt.show()

    return equalized_image


def build_reference_from_masks(
    images: Iterable[np.ndarray],
    masks: Iterable[Optional[np.ndarray]],
    value_min: int = 0,
    value_max: int = 100,
    plot=True,
) -> dict:
    """
    Build reference CDF from masked pixels of multiple uint8 images (masks can be per-image,
    images can be different sizes). Returns dict with 'edges','centers','cdf'.
    """
    # integer bins covering inclusive [value_min..value_max]
    bins = int(value_max - value_min + 1)  # e.g., 101
    edges = np.linspace(value_min - 0.5, value_max + 0.5, bins + 1, dtype=np.float64)
    centers = np.arange(
        value_min, value_max + 1, dtype=np.float64
    )  # exact integer values

    per_cdfs = []
    per_hists = []

    for im, m in zip(images, masks):
        if m is None:
            vals = im.ravel()
        else:
            vals = im[m]
        if vals.size == 0:
            raise ValueError("At least one masked region has no pixels")

        # compute histogram over the aligned edges
        hist, _ = np.histogram(vals, bins=edges, density=True)
        cdf = np.cumsum(hist).astype(np.float64)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
            hist = hist.astype(np.float64)  # / hist.sum()
        per_cdfs.append(cdf)
        per_hists.append(hist)

    # average across images (can also use median)
    ref_cdf = np.mean(np.vstack(per_cdfs), axis=0)
    ref_hist = np.mean(np.vstack(per_hists), axis=0)

    if plot:  # plot histogram instead of cdf
        fig, ax1 = plt.subplots(figsize=(8, 5))
        plt.bar(
            centers, ref_hist, color="gray", alpha=0.7, edgecolor="black", width=1.0
        )
        ax1.set_xlabel("Pixel Intensity")
        ax1.set_ylabel("Normalized Frequency", color="black")
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        ax2.plot(centers, ref_cdf, color="red", lw=2, linestyle="--", label="CDF")
        ax2.set_ylabel("Cumulative Distribution", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title("Reference Histogram and CDF")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xlim(value_min, value_max)
        plt.show()

    return {
        "edges": edges,
        "centers": centers,
        "cdf": ref_cdf,
        "value_min": int(value_min),
        "value_max": int(value_max),
    }


def match_to_reference(
    img,
    reference: dict,
    source_mask: Optional[np.ndarray] = None,
    apply_within_mask: bool = False,
    plot_img=False,
) -> np.ndarray:
    """
    Match `image` (2D uint8) to `reference` returned by build_reference_from_masks_uint8.
    - source_mask: optional boolean mask used to compute the source CDF (same shape as image).
    - apply_within_mask: if True and source_mask given, only masked pixels are modified.
    Returns matched uint8 image (same shape).
    """

    edges = reference["edges"]
    centers = reference["centers"]
    ref_cdf = reference["cdf"]

    # compute source values used for source CDF
    if source_mask is None:
        vals = img.ravel()
        source_mask_bool = None
    else:
        source_mask_bool = np.asarray(source_mask, dtype=bool)
        if source_mask_bool.shape != img.shape:
            raise ValueError("source_mask must match image shape")
        vals = img[source_mask_bool]
        if vals.size == 0:
            raise ValueError("source_mask contains no pixels")

    # source histogram/cdf over same edges as reference
    hist_src, _ = np.histogram(vals, bins=edges)
    src_cdf = np.cumsum(hist_src).astype(np.float64)
    if src_cdf[-1] > 0:
        src_cdf /= src_cdf[-1]

    # build mapping from source percentiles -> reference values (inverse ref CDF)
    # p_grid spans [0,1) for each bin
    eps = 1e-12
    ref_cdf_clip = np.clip(ref_cdf, 0.0, 1.0 - eps)
    p_grid = np.linspace(0.0, 1.0 - eps, ref_cdf_clip.size)

    # inverse CDF of reference: percentile p -> value
    ref_inv_vals = np.interp(
        p_grid, ref_cdf_clip, centers, left=centers[0], right=centers[-1]
    )

    # map each source bin's percentile (src_cdf) to a target value in reference
    mapped_centers = np.interp(
        np.clip(src_cdf, 0.0, 1.0 - eps),
        p_grid,
        ref_inv_vals,
        left=centers[0],
        right=centers[-1],
    )
    # mapped_centers is length = bins (value_min..value_max), i.e. target value for each source center

    # Build full 256-length LUT for cv2.LUT (so we can apply fast). Values outside [vmin..vmax] are clipped to ends.
    full_x = np.arange(256, dtype=np.float64)
    # create arrays to feed interp: x positions of mapped_centers are the original centers
    x_pos = centers  # e.g. 0..100
    y_pos = mapped_centers
    # We want LUT[x] = interp(x over x_pos to y_pos). For x outside x_pos range, use nearest mapped value.
    lut = np.interp(full_x, x_pos, y_pos, left=y_pos[0], right=y_pos[-1])
    lut = np.clip(np.rint(lut), 0, 255).astype(np.uint8)

    # Apply LUT
    if source_mask is not None and apply_within_mask:
        out = img.copy()
        mapped = cv2.LUT(img, lut)
        out[source_mask_bool] = mapped[source_mask_bool]
    else:
        out = cv2.LUT(img, lut)

    if plot_img:
        utils.grayscale_histogram(
            out,
            source_mask_bool,
            apply_within_mask=True,
            title="Reference Matched Pixel Intensities",
        )
        utils.grayscale_histogram(
            img,
            source_mask_bool,
            apply_within_mask=True,
            title="Origianl Pixel Intensities",
        )

    return out


def debris_eraser(
    img: np.ndarray,
    masklist=[],
    method="patch",
    mask_only=False,
    zscore_thresh=3.0,
    boundary=None,
) -> np.ndarray:
    """
    Detect and erase debris from the input image.
    Parameters:
    img (np.ndarray): Input image array (32-bit).
    method (str): Method to use for debris removal. Options are 'cv2', 'patch', or 'mean'.
    masklist (np.ndarray, optional): Additional debris list of masks. Defaults to [].
    mask_only (bool, optional): If True, only erase debris from the mask and do not detect additional debris. Defaults to False.
    save: filepath+name to save cleaned image to. Defaults to ""
    Returns:
    np.ndarray: Image with debris erased with selected method.
    *note: patch method pads image with edge values of the array (raises error for debris on the edge) then crops back to original size
    """

    # Create debris mask
    if mask_only:
        debris_mask = feu.collapse_multilayer(masklist)
    elif len(masklist) == 0:
        debris_mask = utils.detect_debris(img, zscore_thresh=zscore_thresh)
    else:
        masklist.append(utils.detect_debris(img, zscore_thresh=3.0))
        debris_mask = feu.collapse_multilayer(masklist)

    if boundary is not None:
        debris_mask = debris_mask.astype(bool)
        debris_mask_internal = np.logical_and(debris_mask, boundary).astype(np.uint8)
    else:
        debris_mask_internal = debris_mask

    # Erase debris
    if method == "cv2":
        img_clean = cv2.inpaint(img, debris_mask_internal, 3, cv2.INPAINT_TELEA)

    elif method == "patch":
        labelled_mask = label(debris_mask_internal)
        props = regionprops(labelled_mask)
        H, W = img.shape

        if len(props) == 0:
            return img, debris_mask_internal

        # get largest roi
        areas = {prop.label: prop.area for prop in props}
        max_area = max(areas.values())
        max_region = max(areas, key=areas.get)

        minr, minc, maxr, maxc = props[max_region - 1].bbox

        # check if largest roi touches edge
        touches_edge = minr < 50 or minc < 50 or maxr > H - 50 or maxc > W - 50

        if touches_edge and max_area > 5000:  # 10 000
            pad = 100  # 75
        else:
            pad = 16

        img_padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
        mask_padded = np.pad(
            debris_mask_internal,
            ((pad, pad), (pad, pad)),
            mode="constant",
            constant_values=0,
        )

        img_clean_padded = inpaint_pyramid(img_padded, mask_padded, 20)
        img_clean_padded = img_clean_padded[:, :, 0]  # inpaint_pyramid returns 3D array
        img_clean = img_clean_padded[pad:-pad, pad:-pad]

    elif method == "patch_roi":
        r = 10
        pad = r + 5
        halo = r + 6  # source region around each ROI

        img_clean = img.copy()

        labelled = label(debris_mask_internal)
        for region in regionprops(labelled):
            # bounding box of individual debris region
            minr, minc, maxr, maxc = region.bbox
            # expand bbox with halo but stay inside image
            y0 = max(0, minr - halo)
            y1 = min(img.shape[0], maxr + halo)
            x0 = max(0, minc - halo)
            x1 = min(img.shape[1], maxc + halo)

            # extract ROI
            img_roi = img[y0:y1, x0:x1]
            mask_roi = debris_mask_internal[y0:y1, x0:x1]

            img_padded = np.pad(img_roi, ((pad, pad), (pad, pad)), mode="edge")
            mask_padded = np.pad(
                mask_roi, ((pad, pad), (pad, pad)), mode="constant", constant_values=0
            )

            out_padded = inpaint_pyramid(img_padded, mask_padded, r)
            out_padded = out_padded[:, :, 0]  # inpaint_pyramid returns 3D array
            out_roi = out_padded[pad:-pad, pad:-pad]

            img_clean[y0:y1, x0:x1][mask_roi > 0] = out_roi[mask_roi > 0]

    elif method == "mean":
        img_clean = utils.erase_mask(img, debris_mask_internal)
    else:
        raise ValueError(
            "Invalid method. Choose 'cv2', 'patch', 'patch_roi', or 'mean'."
        )

    return img_clean, debris_mask_internal


def clean_img(
    imgpath_xray,
    masklist=[],
    convert_img=None,
    zscore_thresh=3.0,
    boundary=None,
    save=True,
    outputdir=None,
):
    if boundary is not None:
        boundary = tiff.imread(boundary).astype(bool)

    if convert_img == 32:
        xray = utils.convert_image_type_32bit(
            imgpath_xray, save=False
        )  # read in image and ensure image is 32-bit
    elif convert_img == 8:
        xray = utils.convert_image_type_8bit(
            imgpath_xray, save=False
        )  # read in image and ensure image is 8-bit
    else:
        xray = tiff.imread(imgpath_xray)

    corrected, debris_mask = debris_eraser(
        xray,
        masklist=masklist,
        method="patch",
        zscore_thresh=zscore_thresh,
        boundary=boundary,
    )

    del xray
    gc.collect()

    if save and outputdir is not None:
        filename_clean = str(Path(imgpath_xray).stem) + "_clean.tiff"
        filename_debris = str(Path(imgpath_xray).stem) + "_debris.tiff"

        tiff.imwrite(
            outputdir / filename_clean,
            corrected,
            compression=None,
        )

        tiff.imwrite(
            outputdir / filename_debris,
            debris_mask,
            compression=None,
        )

        # print(f"Cleaned image and debris mask saved to {outputdir}")
        return outputdir

    elif save and outputdir is None:
        outputdir = Path(imgpath_xray).parent / "debris_correction_intermediates"
        outputdir.mkdir(parents=True, exist_ok=True)

        filename_clean = str(Path(imgpath_xray).stem) + "_clean.tiff"
        filename_debris = str(Path(imgpath_xray).stem) + "_debris.tiff"

        tiff.imwrite(
            outputdir / filename_clean,
            corrected,
            compression=None,
        )

        tiff.imwrite(
            outputdir / filename_debris,
            debris_mask,
            compression=None,
        )
        print(f"Cleaned image and debris mask saved to {outputdir}")
        return outputdir

    else:
        print("Cleaned image and debris mask returned and not saved")
        return corrected, debris_mask


def clean_img_full(
    imgpath_xray,
    probmap_path_Au,
    probmap_path_ld,
    zscore_thresh=3.0,
    convert_img=None,
    boundary=None,
):
    probmap_Au = tiff.imread(probmap_path_Au)
    probmap_lipiddroplet = tiff.imread(probmap_path_ld)

    if boundary is not None:
        boundary = tiff.imread(boundary).astype(bool)

    if convert_img == 32:
        xray = utils.convert_image_type_32bit(
            imgpath_xray, save=False
        )  # read in image and ensure image is 32-bit
    elif convert_img == 8:
        xray = utils.convert_image_type_8bit(
            imgpath_xray, save=False
        )  # read in image and ensure image is 8-bit
    else:
        xray = tiff.imread(imgpath_xray)

    binary_au_dict = utils.segmentation_threshold_2D(
        probmap_Au, confidence_thresholds=[0.1]
    )
    binary_au = binary_au_dict["threshold" + str(0.1)]

    # Extract segmentation above confidence threshold
    binary_lipid_dict = utils.segmentation_threshold_2D(
        probmap_lipiddroplet, confidence_thresholds=[0.005]
    )
    binary_lipiddroplets = binary_lipid_dict["threshold" + str(0.005)]

    # Fill holes
    binary_lipid_filled = utils.fill_holes(binary_lipiddroplets, gap_fill=False)

    # Get deleted Au
    _, binary_lipid_deleted, _, _ = remove_Au(
        xray,
        binary_lipid_filled,
        binary_lipid_filled,
        threshold=0.3,
        apply_watershed=True,
        removebg=False,
    )

    binary_all_au = np.logical_or(binary_lipid_deleted, binary_au).astype(np.uint8)

    corrected, debris_mask = debris_eraser(
        xray,
        masklist=[binary_all_au],
        method="patch",
        zscore_thresh=zscore_thresh,
        boundary=boundary,
    )

    del (
        xray,
        probmap_Au,
        probmap_lipiddroplet,
        binary_au,
        binary_lipiddroplets,
        binary_lipid_filled,
        binary_lipid_deleted,
        binary_all_au,
    )
    gc.collect()

    return corrected, debris_mask


def unpad(img, padding):
    if padding <= 0:
        return img
    h, w = img.shape[:2]
    return img[padding : h - padding, padding : w - padding]


def postprocess_xray(
    allpaths_dict: dict,
    savedir: str = "",
    intermediate_save: bool = False,
    padding=50,
    example_cell="",
):
    debris_mask = {}
    xray_clean = {}
    xray_clean_final = {}

    # cleans all images and keeps in dictionary before histogram matching

    # remove debris
    for cell in tqdm(allpaths_dict["xray"].keys()):
        # for cell in tqdm([list(allpaths_dict["xray"].keys())[-1]]):  # subset for debugging
        # load in au mask
        au_mask_padded = tiff.imread(savedir / (f"{cell}_layer0_au_labelled.tiff"))
        au_mask = unpad(au_mask_padded, padding)

        if intermediate_save:
            cleandir = clean_img(
                allpaths_dict["xray"][cell],
                masklist=[au_mask],
                zscore_thresh=3.0,
                convert_img=8,
                save=intermediate_save,
                boundary=allpaths_dict["cell"][cell],
            )

            print(f"Cleaned image and debris mask saved to {cleandir.name}")

        else:
            xray_clean[cell], debris_mask[cell] = clean_img(
                allpaths_dict["xray"][cell],
                masklist=[au_mask],
                zscore_thresh=3.0,
                convert_img=8,
                save=intermediate_save,
                boundary=allpaths_dict["cell"][cell],
            )

    # load in all cleaned images
    if intermediate_save:
        xray_clean = utils.load_tiff_into_dict(cleandir, [r"CELL\d{3}", "clean"])

    # calculate reference intensity histogram from all debris corrected cells
    xray_imgs_all: list[np.ndarray] = [img for img in xray_clean.values()]
    cell_outline_all: list[np.ndarray] = [
        np.asarray(tiff.imread(file), dtype=bool)
        for file in allpaths_dict["cell"].values()
    ]
    ref_hist = build_reference_from_masks(xray_imgs_all, cell_outline_all)

    # histogram matching to reference histogram
    xray_matched = {}
    for cell, xray_img in xray_clean.items():
        if (
            example_cell != "" and cell == example_cell
        ):  # plot histogram matching of example cell
            outline = np.asarray(tiff.imread(allpaths_dict["cell"][cell]), dtype=bool)
            xray_matched = match_to_reference(
                xray_img,
                ref_hist,
                source_mask=outline,
                apply_within_mask=True,
                plot_img=True,
            )
            xray_clean_final[cell] = np.pad(
                xray_matched, pad_width=padding, mode="mean"
            )  # pad xray images
        else:
            outline = np.asarray(tiff.imread(allpaths_dict["cell"][cell]), dtype=bool)
            xray_matched = match_to_reference(
                xray_img,
                ref_hist,
                source_mask=outline,
                apply_within_mask=True,
                plot_img=False,
            )
            xray_clean_final[cell] = np.pad(
                xray_matched, pad_width=padding, mode="mean"
            )  # pad xray images

        # print(f"\nFinished processing {cell}")

    if savedir != "":
        for cell in xray_clean_final.keys():
            filename = f"{cell}_layer0_xray_processed.tiff"
            tiff.imwrite(
                savedir / filename,
                xray_clean_final[cell].astype(np.uint8),
            )

        return

    else:
        return xray_clean_final, xray_clean, debris_mask, ref_hist


def postprocess_xray_full(
    allpaths_dict: dict,
    savedir: str = "",
    intermediate_save: bool = False,
    padding=50,
    example_cell="",
):
    debris_mask = {}
    xray_clean = {}
    xray_clean_final = {}

    # cleans all images and keeps in dictionary before histogram matching

    # remove debris
    for cell in tqdm(allpaths_dict["xray"].keys()):
        # for cell in tqdm([list(allpaths_dict["xray"].keys())[-1]]):  # subset for debugging
        xray_clean[cell], debris_mask[cell] = clean_img(
            allpaths_dict["xray"][cell],
            allpaths_dict["au"][cell],
            allpaths_dict["lipiddroplets"][cell],
            zscore_thresh=3.0,
            convert_img=8,
        )

        if intermediate_save:
            filename_clean = cell + "_layer0_xray_clean.tiff"
            filename_debris = cell + "_layer0_xray_debris.tiff"

            intermediate_outdir = savedir / "debris_intermediates"
            intermediate_outdir.mkdir(parents=True, exist_ok=True)

            tiff.imwrite(
                intermediate_outdir / filename_clean,
                xray_clean[cell],
                compression=None,
            )

            tiff.imwrite(
                intermediate_outdir / filename_debris,
                debris_mask[cell],
                compression=None,
            )

            print(
                f"Intermediate {cell} clean image and debris mask saved to {intermediate_outdir}"
            )

    # calculate reference intensity histogram from all debris corrected cells
    xray_imgs_all: list[np.ndarray] = [img for img in xray_clean.values()]
    cell_outline_all: list[np.ndarray] = [
        np.asarray(tiff.imread(file), dtype=bool)
        for file in allpaths_dict["cell"].values()
    ]
    ref_hist = build_reference_from_masks(xray_imgs_all, cell_outline_all)

    # histogram matching to reference histogram
    xray_matched = {}
    for cell, xray_img in xray_clean.items():
        if (
            example_cell != "" and cell == example_cell
        ):  # plot histogram matching of example cell
            outline = np.asarray(tiff.imread(allpaths_dict["cell"][cell]), dtype=bool)
            xray_matched = match_to_reference(
                xray_img,
                ref_hist,
                source_mask=outline,
                apply_within_mask=True,
                plot_img=True,
            )
            xray_clean_final[cell] = np.pad(
                xray_matched, pad_width=padding, mode="mean"
            )  # pad xray images
        else:
            outline = np.asarray(tiff.imread(allpaths_dict["cell"][cell]), dtype=bool)
            xray_matched = match_to_reference(
                xray_img,
                ref_hist,
                source_mask=outline,
                apply_within_mask=True,
                plot_img=False,
            )
            xray_clean_final[cell] = np.pad(
                xray_matched, pad_width=padding, mode="mean"
            )  # pad xray images

    print(f"Finished processing {cell}")

    if savedir != "":
        for cell in xray_clean_final.keys():
            filename = f"{cell}_layer0_xray_processed.tiff"
            tiff.imwrite(
                savedir / filename,
                xray_clean_final[cell].astype(np.uint8),
            )

        return

    else:
        return xray_clean_final, xray_clean, debris_mask, ref_hist


def postprocess_xray_parallel(
    allpaths_dict: dict, outputdir_clean: str = "", padding=50
):  # not functional - crashes with jobs = -1/4
    with tqdm_joblib(desc="Processing", total=len(allpaths_dict["0000"].keys())):
        results = Parallel(n_jobs=2)(
            delayed(clean_img)(
                cell,
                xray_dict[cell],
                probmap_au_dict[cell],
                probmap_ld_dict[cell],
                zscore_thresh=3.0,
            )
            for cell in xray_dict.keys()
        )

    xray_clean = {cellid: corrected for cellid, corrected, mask in results}
    debris_mask = {cellid: mask for cellid, corrected, mask in results}

    return xray_clean, debris_mask


# %% test script
if __name__ == "__main__":
    import pathlib as Path

    #  io
    treatment = "control"
    cellid_pattern = [r"CELL\\d{3}"]

    input_dir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset/input_{treatment}")
    prob_map_dir = input_dir / "prob_maps"
    xray_dir = input_dir / "xray" / "32-bit"

    outputdir = input_dir / "results"
    outputdir.mkdir(parents=True, exist_ok=True)

    # read in images
    probability_files: list[str] = [
        str(f) for f in prob_map_dir.iterdir() if utils.is_tiff(f)
    ]
    xray_files: list[str] = [str(f) for f in xray_dir.iterdir() if utils.is_tiff(f)]

    # get Au detected in lipid droplet segmentation
    probmap_au_dict = {
        utils.extract_keyword(f, cellid_pattern): f
        for f in probability_files
        if "_au_" in str(f)
    }

    probmap_ld_dict = {
        utils.extract_keyword(f, cellid_pattern): f
        for f in probability_files
        if "lipiddroplets" in str(f)
    }

    xray_dict = {
        utils.extract_keyword(f, cellid_pattern): f
        for f in xray_files
        if "xray" in str(f)
    }

    cell_dict = {
        utils.extract_keyword(f, cellid_pattern): f
        for f in probability_files
        if "cell" in str(f)
    }

    xray_dict = {k: v for k, v in xray_dict.items() if k == "CELL048" or k == "CELL049"}
