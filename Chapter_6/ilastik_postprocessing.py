from pathlib import Path
import tifffile as tiff
import numpy as np
import pandas as pd
import glob
from natsort import natsorted
import re
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import utils

#%% function definition 

def get_basename_before_keyword(filepath, keyword, case_sensitive=False):
    """
    Return the part of the filename (without extension) before the first
    occurrence of 'keyword'. If not found, returns None.
    """
    name = Path(filepath).stem
    if not case_sensitive:
        pattern = re.compile(rf"(.+?){re.escape(keyword)}", re.IGNORECASE)
    else:
        pattern = re.compile(rf"(.+?){re.escape(keyword)}")
    m = pattern.search(name)
    if not m:
        return None
    return m.group(1).rstrip("_- .")

def group_tiffs_by_basename(directory, keyword, pattern="*.tif", case_sensitive=False):
    """
    Return dict: {basename: [list_of_filepaths_sorted_numerically]}
    """
    directory = Path(directory)
    files = glob.glob(str(directory / pattern))
    if len(files) == 0:
        return {}
    groups = {}
    for f in files:
        base = get_basename_before_keyword(f, keyword, case_sensitive=case_sensitive)
        if base is None:
            # put unmatched files into a special group (optional)
            base = "__UNMATCHED__"
        groups.setdefault(base, []).append(f)
    # sort each group's files naturally
    for base, flist in groups.items():
        groups[base] = natsorted(flist)
    return groups

def combine_files_to_stack(files, outpath, imagej=True):
    """
    Combine a list of image file paths into a single ImageJ-compatible
    multi-page TIFF saved at outpath. Returns outpath (Path).
    This implementation will load into memory (fast) — if you have very large
    datasets you can modify to stream pages with TiffWriter.
    """
    if len(files) == 0:
        raise ValueError("No files provided to combine.")
    images = [tiff.imread(f) for f in files]
    # ensure consistent shapes
    shapes = {img.shape for img in images}
    if len(shapes) != 1:
        raise ValueError(f"Input images have differing shapes: {shapes}")
    stack = np.stack(images, axis=0)  # shape (N, Y, X) or (N, Y, X, C)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    # convert to string for tifffile
    tiff.imwrite(str(outpath), stack, imagej=imagej)
    return outpath

def combine_groups_to_stacks(directory, keyword, outputdir,
                             out_suffix="_probmap_stack.tif",
                             pattern="*.tiff",
                             case_sensitive=False):
    """
    For each basename (before keyword) found in directory, combine its tiffs
    into a single stack saved into outputdir. Returns dict {basename: output_path}.
    """
    directory = Path(directory)
    outputdir = Path(outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)

    groups = group_tiffs_by_basename(directory, keyword, pattern=pattern, case_sensitive=case_sensitive)
    if not groups:
        raise ValueError("No TIFFs found or no groups detected.")

    saved = {}
    filename = []
    for base, files in groups.items():
        # Skip the unmatched group if you don't want to save it:
        if base == "__UNMATCHED__":
            # optionally handle unmatched files; here we skip saving them
            print(f"Skipping {len(files)} unmatched files (no keyword '{keyword}')")
            continue

        # create a safe filename
        safe_base = re.sub(r'[\\/*?:"<>|]', "_", base)  # remove illegal Windows filename chars
        outname = f"{safe_base}{out_suffix}"
        outpath = outputdir / outname

        try:
            saved_path = combine_files_to_stack(files, outpath, imagej=True)
            saved[base] = str(saved_path)
            print(f"Saved group '{base}' -> {saved_path} ({len(files)} frames)")
        except Exception as e:
            print(f"Failed to save group '{base}': {e}")
        filename.append(outname)
        
    return saved, filename

def fill_holes_bin_2D(binary):
    
    mask = binary.copy().astype(np.uint8)
    
    h, w = mask.shape
    floodfill = mask.copy()
    
    # Create mask for floodFill (must be 2 pixels larger)
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    
    # Flood fill from corner
    cv2.floodFill(floodfill, ff_mask, (0, 0), 255)
    
    # Invert floodfilled image
    floodfill_inv = cv2.bitwise_not(floodfill)
    
    # Combine with original mask
    filled_mask = mask | floodfill_inv
    
    return filled_mask

def fill_holes_bin(binary):    
    
    binary_filled = binary_fill_holes(binary).astype(np.uint8)
    
    filled_mask_3D = np.zeros_like(binary_filled)
    i = 0
    
    for bin_slice in binary_filled:
        
        filled_mask = fill_holes_bin_2D(bin_slice)
        
        filled_mask_3D[i,:,:] = filled_mask
        
        i += 1
    
    return filled_mask_3D

def remove_debris(binary, min_size = 200):
    
    binary_mask_bool = binary > 0
    binary_mask = binary_mask_bool.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    binary_mask_closed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, k, iterations = 2)
    binary_labels = label(binary_mask_closed)

    size_filtered = np.zeros_like(binary_mask_closed, dtype=np.uint8)

    for region in regionprops(binary_labels):
        if  region.area > min_size:
            size_filtered[binary_labels == region.label] = 1

    return size_filtered
#%% IO

project = "3D_HighRes"
channel = "PXN"

parentdir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset4D/{project}")
inputdir = parentdir / "ilastik_probtiffs" / f"{channel}_probtiffs"

stack_dir = parentdir / "ilastik_probmaps" / f"{channel}_probmaps"
stack_dir.mkdir(parents=True, exist_ok=True)
bin_dir = parentdir / "binary_volumes" / f"{channel}_binvol"
bin_dir.mkdir(parents=True, exist_ok=True)

#%% combine output individual tiffs into stacks
files = combine_groups_to_stacks(inputdir, "Probabilities", stack_dir)
#%% probability maps to binary maps

prob_maps_df = utils.load_path_into_df(stack_dir, keywordregex=[r"Series\d{3}"], keyword="probmap")

binary_masks = []
for probmap_row in prob_maps_df.itertuples(index=False):
    
    # try otsu's thresholding method
    
    probmap = np.asarray(tiff.imread(probmap_row.Filepath))
    binary = probmap > 0.5
    binary = binary.astype(np.uint8)
    binary_MAX = np.any(binary, axis = 0).astype(np.uint8)
    binary_filled = ndi.binary_fill_holes(binary_MAX).astype(np.uint8)

    # histogram plot of probabilities - check threshold
    #utils.grayscale_histogram(PXN_segmentation, apply_within_mask=False, title = f"{probmap_row[0]}",xmin = 0.5, xmax = 1,)
    binary_masks.append(binary)
    
    # maximum intensity projection 
    # binary_mask_2D_bool = np.any(PXN_bin_filled.astype(bool), axis=0)
    
    # save binary stacks
    tiff.imwrite(bin_dir / f"{probmap_row.Line}_{probmap_row.Condition}_t{probmap_row.Timepoint}_{probmap_row.Series}_{channel}_binary.tiff", binary_filled)

prob_maps_df["Binary_mask"] = binary_masks

#%% probability maps to binary maps - max intensity projection + fill holes
from skimage.morphology import opening, disk

prob_maps_df = utils.load_path_into_df(stack_dir, keywordregex=[r"Series\d{3}"], keyword="probmap")
i = 0
binary_masks = []
for probmap_row in prob_maps_df.itertuples(index=False):
        
    probmap = np.asarray(tiff.imread(probmap_row.Filepath))
    probmap_avg = np.mean(probmap, axis = 0)
    binary = (probmap_avg > 0.5).astype(np.uint8)
    
    #binary_MAX = np.any(binary > 0, axis = 0).astype(np.uint8)
    #binary_filled = utils.fill_holes(binary_MAX, gap_fill = 0).astype(np.uint8)
    opened = opening(binary, disk(5))
    binary_filled = ndi.binary_fill_holes(opened).astype(np.uint8)
    # histogram plot of probabilities - check threshold
    #utils.grayscale_histogram(PXN_segmentation, apply_within_mask=False, title = f"{probmap_row[0]}",xmin = 0.5, xmax = 1,)
    binary_masks.append(binary_filled)
    
    if i == 25:
        plt.figure()
        plt.imshow(probmap_avg)
        plt.title(f"{probmap_row.Line}_{probmap_row.Condition}")
        plt.axis("off")
    # maximum intensity projection 
    # binary_mask_2D_bool = np.any(PXN_bin_filled.astype(bool), axis=0)
    i += 1
    # save binary stacks
    #tiff.imwrite(bin_dir / f"{probmap_row.Line}_{probmap_row.Condition}_t{probmap_row.Timepoint}_{probmap_row.Series}_{channel}_binary.tiff", binary_filled)

#prob_maps_df["Binary_mask"] = binary_masks
#%% IO - test dataset (individual stack)

inputdir = Path("/Users/IvyWork/Desktop/projects/4d-image-processing/ilastik/BF_PXN_segmentation_training_set")

files, filename = combine_groups_to_stacks(inputdir / "test_segmentation", "Probabilities", inputdir)
filename_bin = filename_bin = "_".join(filename[0].split("_")[:6])

segmentation_path = inputdir / filename[0]
segmentation = np.asarray(tiff.imread(segmentation_path))

segmentation_sum = segmentation.sum(axis = 0)

#%% test - binary stack
import cv2
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes

binary = segmentation > 0.2
binary = binary.astype(np.uint8)
filled_mask_3D = (fill_holes_bin(binary) > 0).astype(np.uint8)

mask_eroded = np.zeros_like(filled_mask_3D)
k = np.ones((3, 3), np.uint8)
i = 0

for mask_slice in filled_mask_3D:
    
    slice_eroded = cv2.erode(mask_slice, k, iterations = 3)
    mask_eroded[i,:,:] = slice_eroded
    
    i += 1

mask_sum = mask_eroded.sum(axis = 0)
mask_filtered = (mask_sum > 3).astype(np.uint8)

#mask_filled = fill_holes_bin_2D(mask_filtered)

mask_filled = binary_fill_holes(mask_filtered)
label_bin = label(mask_filled)
size_filtered = np.zeros_like(mask_filled, dtype=np.uint8)

for region in regionprops(label_bin):
    if  region.area > 500:
        size_filtered[label_bin == region.label] = 1

plt.imshow(mask_filled)
plt.axis("off")

#%%

a_slice = filled_mask_3D[0,:,:]

binary_mask_2D_bool = np.any(filled_mask_3D.astype(bool), axis=0)
binary_mask_2D = binary_mask_2D_bool.astype(np.uint8)

binary_mask_closed = cv2.morphologyEx(binary_mask_2D, cv2.MORPH_OPEN, k, iterations = 2)
binary_labels = label(binary_mask_closed)

size_filtered = np.zeros_like(binary_mask_closed, dtype=np.uint8)

for region in regionprops(binary_labels):
    if  region.area > 200:
        size_filtered[binary_labels == region.label] = 1


#tiff.imwrite(inputdir / "test_bin" / f"{filename_bin}_PXN_binary.tiff", filled_mask_3D)

#%%
# histogram plot of probabilities - check threshold
#utils.grayscale_histogram(PXN_segmentation, apply_within_mask=False, title = f"{probmap_row[0]}",xmin = 0.5, xmax = 1,)
# save binary stacks

#%% 2D probability maps to binary maps
prob_maps_df = utils.load_path_into_df(stack_dir, keywordregex=[r"Series\d{3}"], keyword="Probabilities")

binary_masks = []
for probmap_row in prob_maps_df.itertuples(index=False):
    
    name = Path(probmap_row.Filepath).stem
    plt.figure()
    
    probmap = np.asarray(tiff.imread(probmap_row.Filepath))
    binary = probmap > 0.5
    binary = binary.astype(np.uint8)
    
    plt.imshow(binary)
    plt.title(f"probmap {name}")
    plt.axis("off")
    
#%%
    binary_filled_3D = (fill_holes_bin(binary) > 0).astype(np.uint8)

    mask_eroded = np.zeros_like(binary_filled_3D)
    k = np.ones((3, 3), np.uint8)
    i = 0
    
    for filled_slice in binary_filled_3D:
        slice_eroded = cv2.erode(filled_slice, k, iterations = 3)
        mask_eroded[i,:,:] = slice_eroded
        
        i += 1
    
    binary_sum = mask_eroded.sum(axis = 0)
    binary_sum_filtered = (binary_sum > 3).astype(np.uint8)
    binary_sum_filled = binary_fill_holes(binary_sum_filtered)
    
    label_bin = label(binary_sum_filled)
    size_filtered = np.zeros_like(binary_sum_filled, dtype=np.uint8)

    for region in regionprops(label_bin):
        if  region.area > 250:
            size_filtered[label_bin == region.label] = 1

    # histogram plot of probabilities - check threshold
    #utils.grayscale_histogram(PXN_segmentation, apply_within_mask=False, title = f"{probmap_row[0]}",xmin = 0.5, xmax = 1,)
    binary_masks.append(size_filtered)
    
    # maximum intensity projection 
    # binary_mask_2D_bool = np.any(PXN_bin_filled.astype(bool), axis=0)
    
    # save binary stacks
    tiff.imwrite(bin_dir / f"{probmap_row.Line}_{probmap_row.Condition}_t{probmap_row.Timepoint}_{probmap_row.Series}_PXN_binary.tiff", size_filtered)

prob_maps_df["Binary_mask"] = binary_masks
#%% convert probability maps to binary maps - separate touching components

prob_maps_df = utils.load_path_into_df(stack_dir, keywordregex=[r"Series\d{3}"], keyword="stack")
#prob_maps = utils.load_path_into_dict(stack_dir, keywordregex=[r"Series\d{3}"], keyword="stack")

binary_masks = []
for probmap_row in prob_maps_df.itertuples(index=False):
    
    probmap = np.asarray(tiff.imread(probmap_row.Filepath))
    binary = probmap > 0.5
    binary = binary.astype(np.uint8)
    binary_filled_3D = (fill_holes_bin(binary) > 0).astype(np.uint8)

    mask_eroded = np.zeros_like(binary_filled_3D)
    k = np.ones((3, 3), np.uint8)
    i = 0
    
    for filled_slice in binary_filled_3D:
        slice_eroded = cv2.erode(filled_slice, k, iterations = 3)
        mask_eroded[i,:,:] = slice_eroded
        
        i += 1
    
    binary_sum = mask_eroded.sum(axis = 0)
    binary_sum_filtered = (binary_sum > 3).astype(np.uint8)
    binary_sum_filled = binary_fill_holes(binary_sum_filtered)
    
    label_bin = label(binary_sum_filled)
    size_filtered = np.zeros_like(binary_sum_filled, dtype=np.uint8)

    for region in regionprops(label_bin):
        if  region.area > 250:
            size_filtered[label_bin == region.label] = 1

    # histogram plot of probabilities - check threshold
    #utils.grayscale_histogram(PXN_segmentation, apply_within_mask=False, title = f"{probmap_row[0]}",xmin = 0.5, xmax = 1,)
    binary_masks.append(size_filtered)
    
    # maximum intensity projection 
    # binary_mask_2D_bool = np.any(PXN_bin_filled.astype(bool), axis=0)
    
    # save binary stacks
    tiff.imwrite(bin_dir / f"{probmap_row.Line}_{probmap_row.Condition}_t{probmap_row.Timepoint}_{probmap_row.Series}_PXN_binary.tiff", size_filtered)

prob_maps_df["Binary_mask"] = binary_masks

#%% overlay segmentation 


#%% resample to fill in z-stack gaps
mask_resampled = utils.isotropic_resampling_z(PXN_bin, spacing = (0.49,0.0962,0.0962))

#%% convert probability maps to binary maps - timelapse (each stack is a timepoint)

display_binary_slice = True
test = False
save = True

prob_maps_df = utils.load_path_into_df(stack_dir, keywordregex=[r"t\d{3}"], keyword="stack")

if test:
    rows = prob_maps_df.iloc[[0]] 
else:
    rows = prob_maps_df.copy()

binary_masks = []

for probmap_row in rows.itertuples(index=False):
    segmentation = np.asarray(tiff.imread(probmap_row.Filepath))
    binary = segmentation > 0.5
    binary = binary.astype(np.uint8)
    
    # histogram plot of probabilities - check threshold
    #utils.grayscale_histogram(PXN_segmentation, apply_within_mask=False, title = f"{probmap_row[0]}",xmin = 0.5, xmax = 1,)
    binary_masks.append(binary)
    
    # display slice
    if display_binary_slice:
        a_slice = binary[int(len(binary)/2),:,:]
    
    # save binary stacks
    if save:
        tiff.imwrite(bin_dir / f"{probmap_row.Line}_{probmap_row.Condition}_{probmap_row.Series}_{channel}_binary.tiff", binary)

if len(binary_masks) == len(prob_maps_df):
    prob_maps_df["Binary_mask"] = binary_masks

#%% Maximum Z-projection of macrophage stacks

