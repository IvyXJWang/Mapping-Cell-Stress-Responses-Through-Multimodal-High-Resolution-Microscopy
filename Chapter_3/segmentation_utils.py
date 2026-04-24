# functions to process segmentation probability maps into binary labels for cellprofiler
# changes from v1 - increase processing speed of split_mito

import re
import zipfile
from pathlib import Path
from collections import defaultdict

import tifffile as tiff
import numpy as np
import cv2
from roifile import ImagejRoi
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops, find_contours
from scipy import ndimage as ndi
from skimage import measure, draw
from scipy.fft import fft, ifft
from skimage.morphology import reconstruction
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

from skimage.morphology import disk, binary_dilation

# from feature_extraction_utils import binary_mask_transform
from pyinpaint import Inpaint
from patchmatch_cython import inpaint_pyramid
from PIL import Image
from pathlib import Path
import os
from typing import Optional
from tqdm import tqdm
import math


def is_npz(file: Path) -> bool:
    return file.is_file() and file.suffix == ".npz"


def npz_to_tiff(input_dir: Path, output_dir: Path, feature_order: list):
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files: list[Path] = [f for f in input_dir.iterdir() if is_npz(f)]

    for npz_file in tqdm(npz_files):
        # TODO: clean up this implementation
        probability_data = np.load(npz_file)
        array_names = probability_data.files
        oldfilename = npz_file.stem

        # fix up filename rename segmentation files to CELL### format from CELL_###
        parts = oldfilename.split("_")
        filename = f"{parts[0]}{parts[1]}"

        for array_name in array_names:
            probabilities = probability_data[array_name]

            for feat_idx in range(len(feature_order)):
                feature_img = probabilities[feat_idx][0]
                tiff_img = Image.fromarray(feature_img.astype(np.float32))
                feature_name = feature_order[feat_idx]
                output_path = output_dir / f"{filename}_{feature_name}_segmentation.tif"
                tiff_img.save(output_path, "TIFF")


def save_tiff(img, outputdir, filename):
    im = Image.fromarray(img.astype(np.uint8))
    full_filename = filename + ".tiff"
    im.save(outputdir / full_filename, "TIFF")

    return


def is_tiff(file: Path) -> bool:
    return file.is_file() and (file.suffix == ".tif" or file.suffix == ".tiff")


def filelist_tiff(datadir: Path):
    """
    Return list of files in directory that are tiffs
    """
    return [str(f) for f in datadir.iterdir() if is_tiff(f)]


def extract_keyword(phrase: str, keywordregex: list, keyword_index=0):
    if type(keywordregex) != list:
        keywordregex = list(keywordregex)

    matches = []
    for keyword in keywordregex:
        match = re.findall(keyword, phrase)  # find pattern

        if match[0] not in matches:
            matches.append(match[0])  # uses first instance of keyword match

        if len(matches) == len(keywordregex):  # all keywords found in phrase
            return matches[
                keyword_index
            ]  # return specified keyword in list matching pattern


def load_tiff_into_dict(
    datadir, keywordregex=[r"CELL\d{3}"], padding=0, pad_mode="constant"
):
    """
    Loads all .tiff/.tif files in a directory into a dictionary based on keyword
    """

    filelist = filelist_tiff(datadir)

    # determine value to pad image with

    img_dict = {}

    for i, file in enumerate(filelist):
        matches = []
        for keyword in keywordregex:
            match = re.findall(keyword, file)  # find cellID pattern

            if not match:  # if no valid keywords found skip file
                continue

            matches.append(match[0])  # uses first instance of keyword match

        if len(matches) == len(keywordregex):  # all keywords found in file
            # key = "_".join(matches) # combine both keywords as key
            img = tiff.imread(file)
            if pad_mode == "constant":
                img_pad = np.pad(
                    img, pad_width=padding, mode=pad_mode, constant_values=0
                )
            elif pad_mode == "mean":
                img_pad = np.pad(img, pad_width=padding, mode=pad_mode)

            img_dict[matches[0]] = img_pad  # only use cellID as keys

    print(f"Successfully loaded {len(img_dict)} tiffs")

    return img_dict


def load_path_into_dict(datadir, keywordregex=[r"CELL\d{3}"], keyword=""):
    """ """
    filelist = filelist_tiff(datadir)  # all files in directory

    path_dict = {
        extract_keyword(f, keywordregex): f for f in filelist if keyword in str(f)
    }

    return path_dict


def save_bin_tiff(img, savedir, filename):
    tiff.imwrite(savedir + filename, img.astype(np.uint8) * 255)
    print(f"Image saved as {filename}")


def save_rois(roilist, savedir, filename):
    with zipfile.ZipFile(savedir + filename, mode="w") as zf:
        for i, roi in enumerate(roilist):
            # Assign a name if not already assigned
            if not hasattr(roi, "name") or roi.name is None:
                roi.name = f"region_{i + 1}.roi"
            else:
                # Ensure filename ends with .roi
                if not roi.name.endswith(".roi"):
                    roi.name += ".roi"

            roi_bytes = roi.tobytes()  # get bytes representation

            # Write bytes directly into the zip archive
            zf.writestr(roi.name, roi_bytes)


def segmentation_threshold_2D(
    image, confidence_thresholds, keyword="", DEBUG=False, save=False
):
    """
    Input
        image (2D array): grayscale input image
        confidence_thresholds (list) : list of confidence thresholds
        keyword (str) : only process files containing keyword
        save (str) : output image directory
    Output
        binary (dict) : returns dictionary of each threshold with binary image with only values greater than specified threshold(s)
    """

    binary = {}
    for thresh in confidence_thresholds:
        # Apply threshold
        binary["threshold" + str(thresh)] = (image >= thresh).astype(np.uint8)

    if DEBUG:
        import matplotlib.pyplot as plt

        for i, img in binary.items():
            plt.imshow(img, cmap="gray")
            plt.title(i + " Binary Image")
            plt.axis("off")  # Hides axis ticks
            plt.show()

    return binary


def keep_internal_rois(
    binary_object,  # bool or uint8 mask of object(s) to separate
    binary_cell,  # bool mask for overlap decision
    kernel_size=(10, 10),  # kernel used for erosion/dilation
    separate_connecting_iter=2,
    min_overlap=0.5,
    largest_only=True,
):
    # ensure dtype
    binary_object = (binary_object > 0).astype(np.uint8)
    binary_cell = (binary_cell > 0).astype(np.uint8)

    # erode to separate connected objects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    if separate_connecting_iter > 0:
        eroded = cv2.erode(binary_object, kernel, iterations=separate_connecting_iter)
    else:
        eroded = binary_object.copy()

    if largest_only:
        # Keep only largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded, connectivity=8
        )

        if num_labels <= 1:
            return np.zeros_like(binary_object, dtype=np.uint8), 0, 0

        areas = stats[:, cv2.CC_STAT_AREA]
        # ignore background (label 0)
        areas[0] = 0
        largest_label = int(areas.argmax())
        eroded = (labels == largest_label).astype(np.uint8)

    # Keep internal
    labeled_seeds = label(eroded)
    keep_seed_labels = []
    for region in regionprops(labeled_seeds):
        coords = tuple(region.coords.T)
        area = region.area
        overlap = np.sum(binary_cell[coords])
        if (overlap / area) >= min_overlap:
            keep_seed_labels.append(region.label)

    if len(keep_seed_labels) == 0:
        # nothing kept
        return np.zeros_like(binary_object, dtype=np.uint8), 0, labeled_seeds.max()

    # distance from background
    markers = labeled_seeds.copy()
    distance = distance_transform_edt(binary_object == 0)
    watershed_labels = watershed(
        -distance, markers=markers, mask=binary_object.astype(bool)
    )

    # watershed partitions corresponding to kept seeds
    kept_partition = np.isin(watershed_labels, keep_seed_labels).astype(np.uint8)

    # count number of objects kept
    kept_count = label(kept_partition).max()
    total_count = labeled_seeds.max()
    del_count = total_count - kept_count

    # recover original shape of kept regions
    recovered = np.zeros_like(binary_object, dtype=np.uint8)
    for seed_label in keep_seed_labels:
        marker = (labeled_seeds == seed_label).astype(np.uint8)
        mask = (watershed_labels == seed_label).astype(np.uint8)
        if marker.sum() == 0 or mask.sum() == 0:
            continue
        # reconstruction will dilate marker constrained by mask until stable
        rec = reconstruction(marker, mask, method="dilation")
        recovered = np.logical_or(recovered, rec > 0)

    recovered = recovered.astype(np.uint8)

    # same-kernel dilation mask recovery restricted to original mask
    temp = cv2.dilate(recovered, kernel, iterations=separate_connecting_iter)
    final_recovered = np.bitwise_and(temp, kept_partition)

    # convert filetype
    final_recovered = (final_recovered > 0).astype(np.uint8)

    return final_recovered, kept_count, del_count


def keep_internal_rois_opening(
    binary_cell,
    binary_object,
    min_overlap=0.5,
    separate_connecting_iter=0,
    largest_only=True,
):
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

    if separate_connecting_iter != 0:
        # Erode binary image to separate connected components before identifying internal objects
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        eroded_object = cv2.erode(
            binary_object.astype(np.uint8),
            kernel_erode,
            iterations=separate_connecting_iter,
        )
    else:
        eroded_object = binary_object

    if largest_only:
        # Keep only largest connected component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded_object, connectivity=8
        )

        if num_labels <= 1:
            return np.zeros_like(binary_object, dtype=bool), 0, 0

        areas = stats[:, cv2.CC_STAT_AREA]
        # ignore background (label 0)
        areas[0] = 0
        largest_label = int(areas.argmax())
        eroded_object = (labels == largest_label).astype(np.uint8)

    labeled_objects = label(eroded_object)
    internal_objects = np.zeros_like(binary_object, dtype=bool)

    for region in regionprops(labeled_objects):
        coords = tuple(region.coords.T)  # rows, cols
        area = region.area
        overlap = np.sum(binary_cell[coords])  # count overlapping pixels
        overlap_fraction = overlap / area

        if overlap_fraction >= min_overlap:
            internal_objects[coords] = True

    # count how many objects internal vs external in FOV
    labeled_kept = label(internal_objects)

    kept_count = labeled_kept.max()  # max label number corresponds to maximum rois
    total_count = labeled_objects.max()
    del_count = total_count - kept_count

    if separate_connecting_iter != 0:
        # Dilate to original size
        dilation_iter = int(separate_connecting_iter * 1.5)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated_object = cv2.dilate(
            internal_objects.astype(np.uint8), kernel_dilate, iterations=dilation_iter
        )
        # Mask from original binary shape to maintain original shape
        mask = binary_object.astype(np.uint8)
        recovered_object = cv2.bitwise_and(dilated_object, mask)
    else:
        recovered_object = internal_objects

    return recovered_object, kept_count, del_count


def recover_from_thresholds(
    binary_nucleus_dict, binary_cell, threshold_list, piece=None, DEBUG = False,
):
    # add feature to keep all above certain confidence - keep all internal above 0.9 confidence (instead of just largest region)
    binary_nucleus_full = np.zeros_like(list(binary_nucleus_dict.values())[0])
    binary_nucleus_deleted = np.zeros_like(list(binary_nucleus_dict.values())[0])
    binary_non_nucleus = np.zeros_like(list(binary_nucleus_dict.values())[0])

    recovered_dict = {}
    deleted_dict = {}
    binary_nucleus_processing_dict = {}
    for thresh in threshold_list:
        binary_nucleus = binary_nucleus_dict[f"threshold{str(thresh)}"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        binary_nucleus_deleted_dilated = cv2.dilate(
            binary_nucleus_deleted, kernel, iterations=1
        )
        binary_nucleus_processing = (binary_nucleus > 0) & (
            binary_nucleus_deleted_dilated == 0
        )
        binary_nucleus_processing = binary_nucleus_processing.astype(np.uint8)
        binary_nucleus_processing_dict[thresh] = binary_nucleus_processing

        if binary_nucleus_processing.max() == 0:
            continue

        filled = fill_holes(binary_nucleus_processing)
        internal = np.logical_and(filled > 0, binary_cell > 0)
        u8 = internal.astype(np.uint8)

        if piece == "largest":
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                u8, connectivity=8
            )

            if num_labels <= 1:
                continue
            areas = stats[:, cv2.CC_STAT_AREA]
            # ignore background (label 0)
            areas[0] = 0
            largest_label = int(areas.argmax())
            recovered_region = labels == largest_label
        elif piece == "two-largest":
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                u8, connectivity=8
            )

            if num_labels <= 1:
                continue
            areas = stats[:, cv2.CC_STAT_AREA]
            # ignore background (label 0)
            areas[0] = 0
            largest_labels = areas.argsort()[-2:]  # get indices of two largest
            recovered_region = np.isin(labels, largest_labels)
        elif piece == "three-largest":
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                u8, connectivity=8
            )

            if num_labels <= 1:
                continue
            areas = stats[:, cv2.CC_STAT_AREA]
            # ignore background (label 0)
            areas[0] = 0
            largest_labels = areas.argsort()[-3:]  # get indices of three largest
            recovered_region = np.isin(labels, largest_labels)
        elif piece == "dilated-largest":
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                u8, connectivity=8
            )

            if num_labels <= 1:
                continue
            areas = stats[:, cv2.CC_STAT_AREA]
            # ignore background (label 0)
            areas[0] = 0
            largest_label = int(areas.argmax())
            largest_region = (labels == largest_label).astype(np.uint8)
            recovered_region = cv2.dilate(largest_region, kernel, iterations=1)

        else:
            recovered_region = internal

        recovered_region_full = cv2.dilate(
            recovered_region.astype(np.uint8), kernel, iterations=1
        )
        recovered_dict[thresh] = recovered_region_full
        binary_nucleus_full = np.logical_or(
            binary_nucleus_full > 0, recovered_region_full > 0
        ).astype(np.uint8)
        
        # delete checked regions to not recheck regions
        binary_nucleus_deleted = np.logical_or(
            binary_nucleus_deleted > 0, binary_nucleus_processing > 0
        ).astype(np.uint8)

        binary_non_nucleus = ((binary_nucleus_processing & ~recovered_region) > 0).astype(np.uint8)

        deleted_dict[thresh] = binary_non_nucleus


    if DEBUG:
        return recovered_dict
    
    binary_nucleus_full_filled = fill_holes(binary_nucleus_full)

    return binary_nucleus_full_filled, binary_nucleus_processing_dict, recovered_dict, deleted_dict


def fill_concavities_thresholding(binary_object_boundary, binary_object_filled, DEBUG = False):
    contours = cv2.findContours(
        binary_object_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours[0], key=cv2.contourArea)

    hull_mask = np.zeros_like(binary_object_boundary)
    hull = cv2.convexHull(contour)
    cv2.fillPoly(hull_mask, [hull], 255)
    
    hull_mask_bool = hull_mask > 0

    binary_object_filled_bool = binary_object_filled > 0
    filled_object = np.logical_and(binary_object_filled_bool, hull_mask_bool)
    
    if DEBUG:
        binary_nucleus_filled = fill_holes(filled_object)

        return hull_mask, binary_object_filled, binary_nucleus_filled

    return filled_object.astype(np.uint8)


def fill_concavities(
    binary_object, depth_threshold=200, dilation_radius=2, shallowness=2, debug=False
):
    # shallowness: base:depth ratio threshold to ignore shallow defects

    img = (binary_object > 0).astype(np.uint8) * 255
    # h, w = img.shape

    # find external contours (process each object separately)
    contours, _ = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return img if not debug else (img, np.zeros_like(img), [])

    # candidate mask (to collect defect-fill regions)
    candidate_mask = np.zeros_like(img)

    defects_info = []  # for debug: list of (start,end,far,depth_px) for kept defects

    for cnt in contours:
        if cv2.contourArea(cnt) <= 0:
            continue

        # need hull indices for convexityDefects
        if cnt.shape[0] < 3:
            # too small to have defects
            continue
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            continue
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is None:
            continue

        # Heed: OpenCV's defect depth value is returned as an integer scaled by 256.
        # Convert to pixels:
        for i in range(defects.shape[0]):
            s, e, f, depth = defects[i, 0]
            depth_px = depth / 256.0  # convert fixed-point -> pixels

            if depth_px < depth_threshold:
                continue

            start_pt = tuple(cnt[s][0])
            end_pt = tuple(cnt[e][0])
            far_pt = tuple(cnt[f][0])

            # Build a triangle from start -> far -> end (covers the concavity)
            tri = np.array([[start_pt, far_pt, end_pt]], dtype=np.int32)
            base_len = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
            base_depth_ratio = base_len / depth_px

            if base_depth_ratio > shallowness:
                # too shallow, ignore
                continue

            tmp = np.zeros_like(img)
            cv2.fillPoly(tmp, tri, 255)

            # Dilate the triangle a little to better fill the concavity
            if dilation_radius > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * dilation_radius + 1, 2 * dilation_radius + 1),
                )
                tmp = cv2.dilate(tmp, kernel, iterations=1)

            # Add to candidate mask
            candidate_mask = cv2.bitwise_or(candidate_mask, tmp)
            defects_info.append((start_pt, end_pt, far_pt, float(depth_px)))

        out = cv2.bitwise_or(img, candidate_mask)

        if debug:
            return out, candidate_mask, defects_info
        else:
            return out


def fill_holes(binary_object, kernel=None, gap_fill=0, distancefill=False):
    """
    Input
        binary_object (2D array): binary image of roi
        distancefill (boolean): perform distance based filling as well
        distancefill (boolean): fill concavity with distance based filling (*****not functional currently)

    Output
        filled_object (2D array) : binary image of rois with holes filled
    """
    img_uint8 = (binary_object * 255).astype(np.uint8)

    # clean up gaps for closing holes
    if gap_fill != 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_fill, gap_fill))
        closed = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)
        closed = closed > 0
    else:
        closed = img_uint8

    # mask gap fill by convex hull (prevent overfilling distortion of shape)
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    all_points = np.vstack(contours)

    # convex hull
    hull = cv2.convexHull(all_points)

    # mask same shape as input
    hull_mask = np.zeros_like(img_uint8)
    cv2.fillConvexPoly(hull_mask, hull, 255)

    # keep only closed regions inside convex hull
    closed = np.logical_and(closed, hull_mask > 0)

    # fill holes
    if kernel is None:
        filled = ndi.binary_fill_holes(closed)
    else:
        filled = ndi.binary_fill_holes(closed, structure=kernel)

    # distance based filling
    if distancefill:
        binary = filled
        distance_map = ndi.distance_transform_edt(~binary)

        dist_vals = distance_map[(distance_map > 0) & (distance_map < 50)]
        threshold = np.percentile(dist_vals, 90)

        concavity_fill = distance_map < threshold
        filled_shape = binary | concavity_fill

        only_fill = filled_shape & ~binary

        labeled = label(only_fill)
        props = regionprops(labeled)

        mask_clean = np.zeros_like(binary)
        for region in props:
            if region.area < 500000:  # max size of hole to consider as concavity
                mask_clean[labeled == region.label] = 1

        # only fill concavity
        filled_object = binary | mask_clean

    else:
        filled_object = filled

    return filled_object


def smooth_shape_fft(
    binary_image, FD_retained=0.1, morphological_closing=0, morphological_opening=0
):
    """
    Input
        binary_image (2D array):
        FD_retained (float): relative proportion of Fourier descriptors kept (smaller number = smoother - less high frequency detailed shape information)
    Output
        smoothened_image (2D array): binary image smoothened

    """

    # Find the largest contour in the binary image
    contours = measure.find_contours(binary_image, 0.5)
    if not contours:
        return binary_image.copy()

    smooth_image = np.zeros_like(binary_image, dtype=bool)

    for contour in contours:
        if len(contour) < 8:
            continue  # Skip small/noisy contours

        # Convert contour to complex representation
        z = contour[:, 1] + 1j * contour[:, 0]  # x + i*y

        # Apply DFT (Fourier Descriptors)
        zf = fft(z)

        # Retain low frequencies, zero-out high frequencies
        N = len(zf)
        keep = int(FD_retained * N // 2)
        if keep < 1:
            continue

        # Zero out high-frequency components
        zf_filtered = np.zeros_like(zf)
        zf_filtered[:keep] = zf[:keep]
        zf_filtered[-keep:] = zf[-keep:]

        # Inverse FFT to get smoothed contour
        z_smooth = ifft(zf_filtered)

        # Convert back to row, col
        rr, cc = draw.polygon(np.imag(z_smooth), np.real(z_smooth), binary_image.shape)

        # Create binary mask from smoothed contour
        temp_mask = np.zeros_like(binary_image, dtype=bool)
        temp_mask[rr.astype(int), cc.astype(int)] = True
        temp_mask = ndi.binary_fill_holes(temp_mask)

        smooth_image |= temp_mask

        if morphological_closing != 0:
            img_uint8 = (smooth_image * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morphological_closing, morphological_closing)
            )
            closed = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)
            smoothened_image_final = closed > 0

        elif morphological_opening != 0:
            img_uint8 = (smooth_image * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (morphological_opening, morphological_opening)
            )
            opened = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
            smoothened_image_final = opened > 0

        else:
            smoothened_image_final = smooth_image

    return smoothened_image_final


def filter_object_size(binary_object, min_size=0, max_size=float("inf")):
    """
    Input
        binary_object (2D array): binary image of rois
        min_size (int): minimum size (px) DEFAULT: 0
        max_size (int): maximum size (px) DEFAULT: infinity
    Output
        size_filtered_object (2D array): binary image containing only rois within size threshold
    """
    binary_object = binary_object.astype(bool)

    labeled = label(binary_object)
    sizes = dict(
        zip(
            [idx for idx, r in enumerate(regionprops(labeled))],
            [r.area for r in regionprops(labeled)],
        )
    )
    size_filtered_object = np.zeros_like(binary_object, dtype=bool)

    for region in regionprops(labeled):
        if min_size <= region.area < max_size:
            coords = tuple(region.coords.T)
            size_filtered_object[coords] = True

    return size_filtered_object, sizes


def split_binary_into_individual(binary_mask):
    """
    Takes a binary mask with multiple shapes and returns a list of individual masks,
    one per shape.

    Parameters:
        binary_mask (2D np.array): The input binary image.

    Returns:
        list of 2D np.arrays: Each array is a binary mask for one object.
    """

    if binary_mask.max() <= 1:  # if image is binary then label image
        labeled = label(binary_mask)
    else:
        labeled = binary_mask.astype(int)

    regions = regionprops(labeled)

    individual_masks = []

    for region in regions:
        mask = np.zeros_like(binary_mask, dtype=bool)
        mask[tuple(region.coords.T)] = True
        individual_masks.append(mask)

    return individual_masks


def binary_to_rois(binary_img, region_idx_start=0):
    shapes = split_binary_into_individual(binary_img)

    roi_list = []
    for i, mask in enumerate(shapes):
        region_idx = i + region_idx_start
        contours = find_contours(mask.astype(float), level=0.5)

        contour = max(contours, key=len)
        contour = np.round(contour).astype(np.int16)  # (y, x)
        xy = contour[:, ::-1]  # Convert to (x, y)

        roi = ImagejRoi.frompoints(xy)
        roi.name = (
            f"region_{region_idx}"  # specify which roi number to start naming from
        )

        roi_list.append(roi)

    return roi_list


def binary_roi_dict(binary_dict):
    rois = {}
    for cell, binary in binary_dict.items():
        roi_dict = {}
        roi_list = binary_to_rois(binary)
        for roi in roi_list:
            roi_dict[roi.name] = roi

        rois[cell] = roi_dict

    return rois


def rois_to_binary(rois, img_shape):
    """Convert a list of ROI polygons to a binary mask."""

    mask = np.zeros(img_shape, dtype=np.uint8)
    for roi in rois:
        xy = np.array(roi.coordinates()).T
        rr, cc = draw.polygon(xy[1], xy[0], img_shape)
        mask[rr, cc] = 1
    return mask


def convert_float32_uint16(image_dir, filename_keyword=r"CELL\d{3}"):
    Path(image_dir / "16-bit").mkdir(parents=True, exist_ok=True)

    image_files = [
        str(f)
        for f in image_dir.glob("*.tif")
        if not f.name.startswith(".") and f.is_file()
    ] + [
        str(f)
        for f in image_dir.glob("*.tiff")
        if not f.name.startswith(".") and f.is_file()
    ]

    for file in image_files:
        img = tiff.imread(file)
        img_scaled = img * 100

        match = re.findall(filename_keyword, file)  # find cellID pattern

        if not match:  # if no valid keywords found skip file
            continue

        cellID = match[0]  # uses first instance of keyword match

        filename = cellID + "_xray_16-bit.tif"
        tiff.imwrite(image_dir / "16-bit" / filename, img_scaled.astype(np.uint16))

        print(f"Image saved as {filename}")


def image_diff(img1, img2):
    """
    Calculate the difference between two images.

    Args:
        img1 (np.array): First image.
        img2 (np.array): Second image.
    Returns:
        np.array: Difference image.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same shape.")
    if img1.dtype != img2.dtype:
        raise ValueError("Input images must have the same data type.")
    # Ensure images are in the same format
    if img1.dtype != np.uint8:
        img1 = img1.astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = img2.astype(np.uint8)
    # Calculate the absolute difference
    diff = cv2.absdiff(img1, img2)
    # Convert the difference to a binary mask
    _, binary_diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)

    return binary_diff


def combine_masks(mask_list, method="or"):
    """
    Combine a list of binary masks into a single mask.

    Parameters
    ----------
    mask_list : list of np.ndarray
        List of binary masks (0/1 or 0/255).
    method : str, optional
        How to combine the masks:
        - "or"  : union of all masks
        - "and" : intersection of all masks
        - "xor" : symmetric difference

    Returns
    -------
    np.ndarray
        Combined binary mask (0/1).
    """
    # normalize to 0/1
    masks = [(mask > 0).astype(np.uint8) for mask in mask_list]

    if method == "or":
        combined = np.any(masks, axis=0).astype(np.uint8)
    elif method == "and":
        combined = np.all(masks, axis=0).astype(np.uint8)
    elif method == "xor":
        combined = np.logical_xor.reduce(masks).astype(np.uint8)
    else:
        raise ValueError("method must be 'or', 'and', or 'xor'")

    return combined


def erase_debris(xray, method="mean", zscore_thresh=3.0, area_thresh=50):
    mean_val = np.mean(xray)
    std_val = np.std(xray)

    # z-score based outlier detection
    zscores = (xray - mean_val) / (std_val + 1e-8)
    mask = np.abs(zscores) > zscore_thresh

    # label connected regions
    labeled = label(mask)
    num = labeled.max()

    corrected = xray.copy()

    for region_idx in range(1, num + 1):
        region_mask = labeled == region_idx
        area = np.sum(region_mask)
        if area < area_thresh:
            continue  # ignore small regions

        # create border mask (dilate then subtract region)
        border_mask = (
            ndi.binary_dilation(region_mask, structure=disk(10)) & ~region_mask
        )
        border_vals = xray[border_mask]

        if method == "mean":
            fill_value = np.mean(border_vals) if len(border_vals) > 0 else mean_val
            corrected[region_mask] = fill_value

        elif method == "interpolate":  # does not work - takes too long
            # crude interpolation: use mean of neighbors pixel-by-pixel
            from scipy.ndimage import generic_filter

            interp = generic_filter(xray, np.nanmean, size=3, mode="mirror")
            corrected[region_mask] = interp[region_mask]

    return corrected, mask.astype(np.uint8)


"""
def erase_mask(xray, artifact_mask, method="mean", border_thickness=10):
    mean_val = np.mean(xray)
    labeled = label(artifact_mask)
    num = labeled.max()

    corrected = xray.copy()

    for region_idx in range(1, num + 1):
        region_mask = labeled == region_idx
        # create border mask
        # border_mask = (
        # ndi.binary_dilation(region_mask, structure=disk(10)) & ~region_mask
        # )

        expanded_mask = binary_mask_transform(
            region_mask, pixels=border_thickness, method="dilate"
        ).astype(bool)
        border_mask = expanded_mask & ~region_mask
        border_vals = xray[border_mask]

        if method == "mean":
            fill_value = np.mean(border_vals) if len(border_vals) > 0 else mean_val
            corrected[region_mask] = fill_value

        elif method == "interpolate":  # does not work - takes too long
            # crude interpolation: use mean of neighbors pixel-by-pixel
            from scipy.ndimage import generic_filter

            interp = generic_filter(xray, np.nanmean, size=3, mode="mirror")
            corrected[region_mask] = interp[region_mask]

    return corrected
"""


def erase_artifacts(img, mask_list=[], method="mean"):
    mean_val = np.mean(img)
    img_nodebris, deleted_debris = erase_debris(img, zscore_thresh=2.5)

    if mask_list != []:
        corrected = img_nodebris.copy()
        for mask in mask_list:
            labeled = label(mask)
            num = labeled.max()

            for region_idx in range(1, num + 1):
                region_mask = labeled == region_idx

                # create border mask (dilate then subtract region)
                border_mask = (
                    ndi.binary_dilation(region_mask, structure=disk(10)) & ~region_mask
                )
                border_vals = img_nodebris[border_mask]

                if method == "mean":
                    fill_value = (
                        np.mean(border_vals) if len(border_vals) > 0 else mean_val
                    )
                    corrected[region_mask] = fill_value

                elif method == "interpolate":  # does not work - takes too long
                    # crude interpolation: use mean of neighbors pixel-by-pixel
                    from scipy.ndimage import generic_filter

                    interp = generic_filter(
                        img_nodebris, np.nanmean, size=3, mode="mirror"
                    )
                    corrected[region_mask] = interp[region_mask]

            else:
                corrected = img_nodebris

            mask_list.append(deleted_debris)
            deleted_masks_debris = combine_masks(mask_list)

    return corrected, deleted_masks_debris


def detect_debris(
    img, zscore_thresh=3.0, area_min=50, area_max=200000, patch_radius=10
):
    mean_val = np.mean(img)
    std_val = np.std(img)

    # z-score based outlier detection
    zscores = (img - mean_val) / (std_val + 1e-8)
    mask = np.abs(zscores) > zscore_thresh

    # label connected regions
    labelled_mask = label(mask)
    debris_mask = np.zeros_like(mask, dtype=bool)
    H, W = img.shape

    for region in regionprops(labelled_mask):
        area = region.area
        minr, minc, maxr, maxc = region.bbox
        # remove large incomplete debris touching image edge
        touches_edge = minr == 0 or minc == 0 or maxr == H or maxc == W
        if area_max >= region.area >= area_min:
            if touches_edge and area > 4 * patch_radius:
                continue
            debris_mask[labelled_mask == region.label] = True

    debris_mask = binary_dilation(
        debris_mask, disk(patch_radius)
    )  # account for debris border

    return debris_mask.astype(np.uint8)


def erase_debris_inpainting(img, fill_mask):
    inpaint = Inpaint(img, fill_mask, ps=5)
    H, W = inpaint.org_img.shape[:2]
    ps = inpaint.ps
    num_patches = (H // ps) * (W // ps)
    k_patch = min(3, num_patches)
    print(k_patch)
    # corrected = inpaint(k_boundary=4, k_search=400, k_patch=3)

    return  # corrected


def convert_image_type_16bit(single_file, save=True):
    img = tiff.imread(single_file)
    max_val = img.max()

    if max_val < 256:
        img_uint8 = img.astype(np.uint8)
        img_bin = (img > 0).astype(bool)
    else:
        img_uint8 = img  # leave unchanged
        print(f"{single_file}: max={max_val}, left as is")

    out_path = Path(single_file).parent / "16-bit" / Path(single_file).name
    # tiff.imwrite(out_path, img_uint8)
    tiff.imwrite(
        out_path,
        img_bin,
        photometric="minisblack",  # <-- important
        compression=None,
    )

    return img_uint8


def convert_image_type_bool(single_file, save=True):
    img = tiff.imread(single_file)
    max_val = img.max()

    if max_val < 256:
        img_uint8 = img.astype(np.uint8)
        img_bin = (img > 0).astype(bool)
    else:
        img_uint8 = img  # leave unchanged
        print(f"{single_file}: max={max_val}, left as is")

    out_path = Path(single_file).parent / "16-bit" / Path(single_file).name
    # tiff.imwrite(out_path, img_uint8)
    tiff.imwrite(
        out_path,
        img_bin,
        photometric="minisblack",  # <-- important
        compression=None,
    )

    return img_uint8


def convert_image_type_32bit(single_file, save=True):
    img = tiff.imread(single_file)
    if img.dtype == np.float32:
        img_32 = img  # leave unchanged
        print(f"{Path(single_file).name}: image already 32-bit, left as is")
    else:
        img_32 = img.astype(np.float32)

    if save:
        out_dir = Path(single_file).parent / "32-bit"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / Path(single_file).name
        tiff.imwrite(
            out_path,
            img_32,
            compression=None,
        )
        print(f"32-bit image saved to {out_dir}")

    return img_32


def convert_image_type_8bit(single_file, save=True):
    img = tiff.imread(single_file)
    if img.dtype == np.uint8:
        img_8 = img  # leave unchanged
        print(f"{Path(single_file).name}: already 8-bit, left as is")

    else:
        img_8 = np.clip(np.floor(img + 0.5), 0, 255).astype(
            np.uint8
        )  # round to nearest integer and clip overflow values

    if save:
        out_dir = Path(single_file).parent / "8-bit"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / Path(single_file).name
        tiff.imwrite(
            out_path,
            img_8,
            compression=None,
        )
        print(f"8-bit image saved to {out_dir}")

    return img_8


def save_labelled_img(img_dict, label_limit=np.inf):
    for k, img in img_dict.items():
        num_labels = img.max()

        if num_labels > label_limit:  # notify if more labels than expected found
            print(f"Number of labels found in {k} greater than limit of {label_limit}")

    print("finished saving all images :)")


def grayscale_histogram(
    img,
    source_mask: Optional[np.ndarray] = None,
    apply_within_mask: bool = True,
    xmin: float = 0,
    xmax: float = 100,
    bins: int = 100,
    title="Histogram of Cell Pixel Intensities",
    plot_cdf=False,
):
    if apply_within_mask and source_mask is not None:
        source_mask = source_mask.astype(bool)
        region_values = img[source_mask]
    else:
        region_values = img

    if plot_cdf:
        counts, bin_edges = np.histogram(
            region_values.ravel(), bins=bins, range=(xmin, xmax), density=False
        )
        cdf = np.cumsum(counts).astype(float)
        cdf /= cdf[-1]  # normalize to 1.0

    # Plot histogram
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(
        region_values.ravel(),
        bins=bins,
        color="gray",
        alpha=0.7,
        edgecolor="black",
        density=True,
        range=(xmin, xmax),
    )

    ax1.set_title(title)
    ax1.set_xlabel("Pixel Intensity")
    ax1.set_ylabel("Normalized frequency")
    ax1.set_xlim(xmin, xmax)
    ax1.grid(True, linestyle="--", alpha=0.5)

    if plot_cdf:
        ax2 = ax1.twinx()
        # Use midpoints of each bin for CDF plot
        bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        ax2.plot(bin_mids, cdf, color="red", linestyle="--", linewidth=2, label="CDF")
        ax2.set_ylabel("Cumulative Distribution", color="red")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis="y", labelcolor="red")

    plt.show()

    return


def split_binary_size(binary_img, area_threshold=7500):
    labeled = label(binary_img)
    small_mask = np.zeros_like(binary_img, dtype=bool)
    large_mask = np.zeros_like(binary_img, dtype=bool)

    for region in regionprops(labeled):
        coords = tuple(region.coords.T)
        if region.area < area_threshold:
            small_mask[coords] = True
        else:
            large_mask[coords] = True

    return small_mask, large_mask


# %% test
"""
treatment = "control"
input_dir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset/input_{treatment}")
xray_files_dir = input_dir / "results" 

xray_file: list[str] = [str(f) for f in xray_files_dir.iterdir() if is_tiff(f) and "CELL012" in str(f) and "xray" in str(f)]
img = tiff.imread(xray_file)

filename = Path(xray_file[0]).stem + "_clean.tif"

img_clean, debris_mask = erase_debris(img, zscore_thresh = 2.5)
tiff.imwrite(xray_files_dir / filename, img_clean)
"""
# %% test
"""
treatment = "inflammation"
input_dir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset/input_{treatment}")

results_dirs = input_dir / "results" / "16-bit"
output_dirs = input_dir / "results"

cell_files: list[str] = [
    str(f) for f in results_dirs.iterdir() if is_tiff(f) and "cell_binary" in str(f)
]

img = {}
for file in cell_files:
    
    img[str(file)] = convert_image_type(file)

print("finished converting all images")
"""
