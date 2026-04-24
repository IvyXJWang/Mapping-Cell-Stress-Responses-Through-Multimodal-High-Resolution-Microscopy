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
from tqdm import tqdm
import constants

#%% function definition

def detect_holes(volume, connectivity=1, spacing=(1.0, 1.0, 1.0), min_voxels = 1):
    """
    Detect internal holes in a 3D binary volume.

    Parameters
    ----------
    volume : ndarray (bool or 0/1), shape (Z,Y,X)
        Input binary volume. True/1 = foreground (object), False/0 = background.
    connectivity : int (1 or 2 or 3)
        Connectivity for labeling background components:
          - connectivity=1 -> face connectivity (6-neighbors in 3D)
          - connectivity=2 -> faces+edges (18-neighbors)
          - connectivity=3 -> faces+edges+corners (26-neighbors)
     spacing : scalar or 3-tuple (z, y, x)
        Physical voxel spacing in same units you want the volume to be (e.g. microns).
        If scalar, isotropic spacing assumed.
    min_voxels : int
        Filter out holes smaller than this voxel count.
        
    Returns
    -------
    hole_mask : ndarray (bool), same shape as volume
        True for voxels belonging to internal holes (background components fully enclosed).
    hole_props : list of dict
        For each hole: {'label': int, 'voxel_count': int, 'bbox': (z0,z1, y0,y1, x0,x1)}
        bbox is in Python slice-exclusion form: [z0:z1, y0:y1, x0:x1] contains the region.
    filled_volume : ndarray (bool)
        Copy of `volume` with holes filled (i.e., those background voxels set to foreground).
    """
    vol_bool = volume.astype(bool)
    vz, vy, vx = spacing
    voxel_volume = vz * vy * vx  # units: spacing units^3
    
    # invert: background True, object False
    inv = ~vol_bool
    
    # generate connectivity structure and label background
    struct = ndi.generate_binary_structure(3, connectivity)
    labeled, n_labels = ndi.label(inv, structure=struct)
    if n_labels == 0:
        # no background components (edge-case)
        return np.zeros_like(vol_bool), [], vol_bool.copy()

    # find labels that touch the border -> these are exterior background
    borders = []
    # collect border slices
    border_slices = [
        labeled[0, :, :],    # z = 0
        labeled[-1, :, :],   # z = -1
        labeled[:, 0, :],    # y = 0
        labeled[:, -1, :],   # y = -1
        labeled[:, :, 0],    # x = 0
        labeled[:, :, -1],   # x = -1
    ]
    # unique labels on border (exclude 0)
    border_labels = np.unique(np.concatenate([np.unique(s) for s in border_slices]))
    border_labels = set(lbl for lbl in border_labels if lbl != 0)

    # all candidate hole labels (those not touching border)
    all_labels = np.array(sorted(list(set(range(1, n_labels + 1)) - border_labels)), dtype=int)
    if all_labels.size == 0:
        empty = np.zeros_like(vol_bool, dtype=bool)
        return empty, [], vol_bool.copy(), empty

    # create hole mask
    hole_mask = np.isin(labeled, all_labels)

    # compute voxel counts
    counts = ndi.sum(inv.astype(np.int64), labeled, index=all_labels)
    # find bounding boxes using find_objects
    objects = ndi.find_objects(labeled)  # list indexed by label-1
    
    all_props = []
    for i, lbl in enumerate(all_labels):
        obj = objects[lbl - 1]  # may be None
        if obj is None:
            bbox = None
            centroid = None
        else:
            zsl, ysl, xsl = obj
            bbox = (int(zsl.start), int(zsl.stop),
                    int(ysl.start), int(ysl.stop),
                    int(xsl.start), int(xsl.stop))

            local_mask = (labeled[obj] == lbl)
            data_mask = (inv[obj] & local_mask)
            if data_mask.sum() == 0:
                centroid = None
            else:
                zz, yy, xx = np.nonzero(data_mask)
                centroid = (float(zz.mean() + zsl.start),
                            float(yy.mean() + ysl.start),
                            float(xx.mean() + xsl.start))

        voxel_count = int(counts[i])
        phys_vol = round(voxel_count * voxel_volume, 2)

        all_props.append({
            'label': int(lbl),
            'voxel_count': voxel_count,
            'physical_volume': phys_vol,
            'voxel_volume': voxel_volume,
            'bbox': bbox,
            'centroid': centroid
        })
        
    keep_mask = np.ones(len(all_props), dtype=bool)
    if min_voxels is not None:
        keep_mask &= np.array([p['voxel_count'] >= int(min_voxels) for p in all_props])

    kept_indices = np.nonzero(keep_mask)[0]

    if kept_indices.size == 0:
        empty = np.zeros_like(vol_bool, dtype=bool)
        return hole_mask, [], vol_bool.copy(), empty

    kept_labels = all_labels[kept_indices]
    hole_mask_filtered = np.isin(labeled, kept_labels)
    hole_props_filtered = [all_props[i] for i in kept_indices]
    
    # optionally fill holes
    filled_volume = vol_bool.copy()
    filled_volume[hole_mask] = True

    return hole_mask, hole_props_filtered, filled_volume, hole_mask_filtered # needs to be fixed - holes not all detected

from skimage.measure import label, regionprops
from scipy.spatial import ConvexHull
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, binary_fill_holes

def fill_holes(vol):
    
    Z, Y, X = vol.shape
    filled = np.zeros_like(vol, dtype=bool)
    hole_mask = np.zeros_like(vol, dtype=bool)
    
    for z in range(Z):
        z_slice = vol[z]
        filled_slice = binary_fill_holes(z_slice)
        holes = filled_slice & ~z_slice
        
        filled[z] = filled_slice
        hole_mask[z] = holes
        
    return filled, hole_mask

def fill_borders_slices_3D(vol):
    
    Z, Y, X = vol.shape
    closed = np.zeros_like(vol, dtype=bool)
    
    for z in range(Z):
        
        z_slice = vol[z]
        
        if np.sum(z_slice) == 0: # empty zslice
            closed[z] = z_slice
            continue
        
        z_slice_cv = z_slice.astype(np.uint8)*255
        contours, _ = cv2.findContours(
            z_slice_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        all_points = np.vstack(contours)
    
        # convex hull
        hull = cv2.convexHull(all_points)
    
        # mask same shape as input
        hull_mask = np.zeros_like(z_slice)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        hull_mask_bool = (hull_mask != 0)

        hull_mask_eroded = binary_erosion(hull_mask_bool, structure=np.ones((3,3))).astype(bool)
        hull_mask_border = hull_mask_bool & (~hull_mask_eroded.astype(bool))
        closed_slice = z_slice | hull_mask_border

        closed[z] = closed_slice
    
    return closed.astype(np.uint8)


def detect_holes_regionprops(binary_stack, connectivity=2, pad_width_z = 2, spacing=(1.0, 1.0, 1.0), area_minmax = (10,100000), close = False):
    
    Z, Y, X = binary_stack.shape
    bin_bool = np.asarray(binary_stack).astype(bool, copy=False)
    padded = np.pad(bin_bool, ((pad_width_z, pad_width_z), (0,0), (0,0)), mode='constant', constant_values=False)
    
    if close:
        structure = np.ones((3,5,5), dtype=bool)   # adjust as needed
        closed = ndi.binary_closing(padded, structure=structure)
    else:
        closed = padded
        
    inv = ~closed
    
    labeled = label(inv, connectivity=connectivity)
    regions = regionprops(labeled)
    
    hole_mask_padded = np.zeros_like(padded, dtype=bool)
    hole_props = []

    for region in regions:
        if area_minmax[0] <= region.area <= area_minmax[1]:
            hole_mask_padded[labeled == region.label] = True

            hole_props.append({
                "label": region.label,
                "voxel_count": region.area,
                "volume_physical_um": region.area * np.prod(spacing),
                "centroid": region.centroid,
            })
            
    z0 = pad_width_z
    z1 = pad_width_z + Z
    
    hole_mask = hole_mask_padded[z0:z1, :, :]
    
    # optionally fill holes
    filled_volume = binary_stack.copy()
    filled_volume[hole_mask] = True
    
    return hole_mask, hole_props, filled_volume

import numpy as np
import cv2
from skimage.measure import label, regionprops
from typing import List, Tuple

def separate_overlapping(
    binary_img: np.ndarray,
    erosion_iters: int = 20,
    dilate_iters: int = 20,
    kernel_size: int = 3,
    min_area: int = 5,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Separate overlapping connected components by:
      1. heavy erosion to get seeds,
      2. label seeds and get per-region masks,
      3. dilate each seed while masking with the original binary mask.

    Args:
      binary_img: 2D binary image (dtype any). Non-zero means foreground.
      erosion_iters: iterations for cv2.erode to produce seeds (higher -> smaller seeds).
      dilate_iters: iterations for cv2.dilate applied to each seed (grow back).
      kernel_size: structuring element size (odd int, e.g. 3).
      min_area: ignore seed regions smaller than this area.

    Returns:
      (mask_list, labeled_image)
        mask_list: list of 2D uint8 masks (0/1) — one per region (same shape as input).
        labeled_image: uint16 (or larger) image where 0=background, 1..N = region labels
    """

    # --- normalize input to 0/1 uint8
    if binary_img.ndim != 2:
        raise ValueError("binary_img must be a 2D array")
    orig_mask = (binary_img != 0).astype(np.uint8)

    # structuring element
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # --- 1) Erode to get interior seeds
    eroded = cv2.erode(orig_mask, kernel, iterations=erosion_iters)

    # --- 2) Label seeds
    labeled_seeds = label(eroded)  # skimage.measure.label
    regions = regionprops(labeled_seeds)

    # Prepare outputs
    H, W = orig_mask.shape
    mask_list: List[np.ndarray] = []
    labeled_out = np.zeros((H, W), dtype=np.int32)

    label_id = 1
    for region in regions:
        if region.area < min_area:
            # skip very small seeds
            continue

        # bbox to crop region for speed (min_row, min_col, max_row, max_col)
        minr, minc, maxr, maxc = region.bbox
        # Expand bbox by kernel_size + dilate_iters to avoid clipping during dilation
        pad = kernel_size // 2 + dilate_iters + 1
        r0 = max(minr - pad, 0)
        c0 = max(minc - pad, 0)
        r1 = min(maxr + pad, H)
        c1 = min(maxc + pad, W)

        # small seed mask (crop)
        seed_crop = (labeled_seeds[r0:r1, c0:c1] == region.label).astype(np.uint8)

        if seed_crop.sum() == 0:
            continue

        # dilate the seed in crop-space
        dilated_crop = cv2.dilate(seed_crop, kernel, iterations=dilate_iters)

        # mask with original image crop so we don't grow outside original object
        orig_crop = orig_mask[r0:r1, c0:c1]
        final_crop = (dilated_crop.astype(np.uint8) & orig_crop.astype(np.uint8))

        if final_crop.sum() == 0:
            continue

        # place into full-size mask
        full_mask = np.zeros_like(orig_mask, dtype=np.uint8)
        full_mask[r0:r1, c0:c1] = final_crop

        # optionally: ensure we don't overwrite previously assigned pixels
        # If you prefer the first region to take precedence, uncomment next two lines:
        # conflict = (labeled_out != 0) & (full_mask != 0)
        # full_mask[conflict] = 0

        # assign label id in labeled_out only where not already assigned
        unlabeled_pixels = (full_mask != 0) & (labeled_out == 0)
        labeled_out[unlabeled_pixels] = label_id

        mask_list.append(full_mask)
        label_id += 1

    # convert labeled_out to smallest safe dtype
    max_label = labeled_out.max()
    if max_label <= np.iinfo(np.uint8).max:
        labeled_out = labeled_out.astype(np.uint8)
    elif max_label <= np.iinfo(np.uint16).max:
        labeled_out = labeled_out.astype(np.uint16)
    else:
        labeled_out = labeled_out.astype(np.int32)

    return mask_list, labeled_out

 #%% IO

parentdir = Path(r"C:/Users/IvyWork/Desktop/projects/dataset4D/3D_HighRes")

bin_dir = parentdir / "binary_volumes"
prob_dir = parentdir / "ilastik_probmaps" / "PXN_probmaps"
void_dir = bin_dir / "void_masks"
void_dir.mkdir(parents=True, exist_ok=True)

prob_maps_df = utils.load_path_into_df(prob_dir, keywordregex=[r"Series\d{3}"], keyword="probmap")
bin_maps_df = utils.load_path_into_df(bin_dir, keywordregex=[r"Series\d{3}"], keyword="PXN")
fl_maps_df = utils.load_path_into_df(parentdir / "ilastik_input", keywordregex=[r"Series\d{3}"], keyword="PXN")

lines_ordered = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]
#merge_cols = [col for col in prob_maps_df.columns if not col.endswith("filepath")]
#prob_maps_df = prob_maps_df.merge(fl_maps_df, on=merge_cols, how="outer")

#%% thresholding

bin_maps = []
for probmap_row in prob_maps_df.itertuples(index=False):
    
    img = tiff.imread(probmap_row.Filepath)
    binary = img > 0.8
    z,y,x = binary.shape
    
    bin_dict = {
        "Line": probmap_row.Line,
        "Series": probmap_row.Series,
        "Condition": probmap_row.Condition,
        "Timepoint": probmap_row.Timepoint,
        "BinaryMask": binary
        }
    
    plt.figure()
    plt.imshow(binary[round(z/2),:,:])
    plt.title(f"{probmap_row.Line} {probmap_row.Condition}")
    plt.axis("off")
    
    bin_maps.append(bin_dict)

bin_map_recalculated_df = pd.DataFrame(bin_maps)
combined_bin_fl = bin_map_recalculated_df.merge(
    fl_maps_df[["Series", "Line", "Condition", "Timepoint","Filepath"]],
    on=["Series", "Line", "Condition", "Timepoint"],
    how="left"
)
#%% label cells
from skimage.segmentation import clear_border
plot = False
mask_list = []

for probmap_row in tqdm(combined_bin_fl.itertuples(index=False), total=len(combined_bin_fl)):
    #binary_mask = tiff.imread(probmap_row.Filepath)
    binary_mask = probmap_row.BinaryMask
    img_fl = tiff.imread(probmap_row.Filepath)
    
    z,y,x = binary_mask.shape
    labeled = label(binary_mask)
    regions = regionprops(labeled)
    
    mask_large = np.zeros_like(labeled, dtype=bool)
    for region in regions: 
        if 50000 <= region.area:
            mask_large[labeled == region.label] = True
    
    labeled_large = label(mask_large)
    regions_large = regionprops(labeled_large)
    
    areas = [r.area for r in regions_large]
    max_area = np.percentile(areas, 0.75)
    
    mask_filtered = np.zeros_like(labeled, dtype=bool)
    area_um = []
    for region in regions_large:
        if region.area <= 150000: #max_area * 1.5:
            mask_filtered[labeled_large == region.label] = True
            area_um.append(region.area * 0.096 * 0.096)
    
    # stop processing if no objects remain after size filtering
    if not mask_filtered.any():
        print(f"{probmap_row.Line}_{probmap_row.Condition}_{probmap_row.Series} no cell detected")
        plt.figure()
        plt.imshow(binary_mask[round(z/2),:,:])
        plt.title(f"{probmap_row.Line} {probmap_row.Condition}: no cells found")
        plt.axis("off")
        continue
    
    labeled_filtered = label(mask_filtered)
    padded = np.pad(labeled_filtered, ((1, 1), (0, 0), (0, 0)), mode='constant')
    cleared = clear_border(padded) 
    labeled_filtered = cleared[1:-1]
    
    if plot:
        plt.figure()
        plt.imshow(labeled_filtered[round(z/2),:,:])
        plt.title(f"{probmap_row.Line} {probmap_row.Condition}: max area {round(max_area)}")
        plt.axis("off")
    
    labels_present = np.unique(labeled_filtered)

    masks_indiv = []
    for lbl in labels_present:
        if lbl == 0:
            continue
        mask = (labeled_filtered == lbl)
        mask = mask.astype(np.uint8) * lbl
        masks_indiv.append(mask)
    
    mask = {
        "Line": probmap_row.Line,
        "Series": probmap_row.Series,
        "Condition": probmap_row.Condition,
        "Timepoint": probmap_row.Timepoint,
        "Labeled_mask": labeled_filtered,
        "Original_fl": img_fl,
        "Individual_masks": masks_indiv
        }
    
    mask_list.append(mask)

mask_df = pd.DataFrame(mask_list)
mask_df_expanded = (
    mask_df
    .explode("Individual_masks")
    .assign(CellID=lambda df: df.groupby(level=0).cumcount() + 1)
    .reset_index(drop=True)
)

#%%
# pip install scikit-image scipy numpy pandas matplotlib
import numpy as np
from scipy import ndimage as ndi
from skimage import segmentation, morphology, measure, color
import pandas as pd

row = prob_maps_df.loc[0]
b = tiff.imread(row.Filepath)
hole_mask, hole_prop, filled = detect_holes_regionprops(b, connectivity=2, spacing = (0.49,0.0962,0.0962))
filled_2D = filled[14,:,:]
unfilled_2D = b[14,:,:]
hole_mask_2D = hole_mask[14,:,:]

#%% fill holes
b = tiff.imread(row.Filepath)
z,_,_ = b.shape

filled_img = fill_holes(b)
labeled = label(filled_img, connectivity = 1)
one_label = (labeled == 2).astype(np.uint8)

for z in range(z):
    
    plt.figure()
    plt.imshow(one_label[z])
    plt.title(f"slice {z}")
    plt.axis("off")

#%% fill holes in contour

row = mask_df_expanded.loc[0]
img = row.Individual_masks

#plt.imshow(img[14,:,:])

border_img = fill_borders_slices_3D(img)

z,_,_ = border_img.shape

for z in range(z):
    
    plt.figure()
    plt.imshow(border_img[z])
    plt.title(f"slice {z}")
    plt.axis("off")
    
#%%
#final_recovered, count = utils.separate_objects(filled_2D)

print(count)
plt.figure()
plt.imshow(unfilled_2D)
plt.axis("off")

plt.figure()
plt.imshow(hole_mask_2D)
plt.axis("off")

plt.figure()
plt.imshow(filled_2D)
plt.axis("off")
#%% detect voids
void_list = []

for row in tqdm(mask_df_expanded.itertuples(index=False), total=len(mask_df_expanded)):
    
    mask = row.Individual_masks
    closed_mask = fill_borders_slices_3D(mask)

    hole_mask, hole_prop, filled = detect_holes_regionprops(closed_mask, connectivity=2, spacing = (0.49,0.0962,0.0962))
    
    hole_vol_list = [d["volume_physical_um"] for d in hole_prop]
    if len(hole_vol_list) == 0: 
        hole_avg = 0
        hole_upper = 0
        hole_std = 0
        
    else:
        hole_avg = np.mean(hole_vol_list)
        hole_upper = np.percentile(hole_vol_list, 75)
        hole_std = np.std(hole_vol_list)
    
    void = {
        "CellID": row.CellID,
        "Line": row.Line,
        "Series": row.Series,
        "Condition": row.Condition,
        "Timepoint": row.Timepoint,
        "Hole_volumes": hole_vol_list,
        "Hole_vol_avg": hole_avg,
        "Hole_vol_75": hole_upper,
        "Hole_number": len(hole_vol_list),
        "Hole_vol_std": hole_std,
        "Hole_mask": hole_mask,
        "Filled_mask": filled,
        "Cell_volume": np.sum(filled.astype(np.uint8))*0.49*0.0962*0.0962
        }
    
    void_list.append(void)
    #tiff.imwrite(void_dir / f"{probmap_row.Line}_{probmap_row.Condition}_t{probmap_row.Timepoint}_{probmap_row.Series}_voids_binary.tiff", hole_mask.astype(np.uint8))
    
    #print(f"Found holes: {len(hole_prop)} in {probmap_row.Series}")

void_df = pd.DataFrame(void_list)
void_df_expanded = void_df.explode("Hole_volumes").reset_index(drop=True)
#%% per-cell measurements

cell_measures = []

for probmap_row in prob_maps_df.itertuples(index=False):
    
    labeled = label(probmap_row.Binary_mask)
    labeled_filtered = np.zeros_like(labeled)
    regions = regionprops(labeled)
    for region in regions:
        if 1000 <= region.area:
            labeled_filtered[labeled == region.label] = region.label
    
    labelled_2D = np.any(labeled_filtered > 0, axis=0).astype(np.uint8)
    
    masks, labeled = separate_overlapping(labelled_2D,
                                           erosion_iters=20,
                                           dilate_iters=20,
                                           kernel_size=3)
    plt.figure()
    plt.imshow(labeled)
    plt.axis("off")
#%%
    # measure void size per cell 
    regions_filtered = regionprops(labeled_filtered)
    for region_filtered in regions_filtered:
        #cell_mask = 
    
    cell_measure = {
        "Line": probmap_row.Line,
        "Condition": probmap_row.Condition,
        "Timepoint": probmap_row.Timepoint,
        "Series": probmap_row.Series,
        "Cell_area": ,
        "Void_volume_avg":,
        }
    
    cell_measures.append(cell_measure)
    
    plt.figure()
    plt.imshow(labeled_filtered[10,:,:])
    plt.axis("off")

cell_measure_df = pd.DataFrame(cell_measures)

#%% detect voids regionprops - function development
single_row = prob_maps_df.loc[0]
binary_mask = tiff.imread(single_row.Filepath)
hole_mask, hole_props = detect_holes_regionprops(binary_mask, connectivity = 2)

a_slice = hole_mask[12,:,:]
plt.figure()
plt.imshow(a_slice)
plt.axis("off")

import napari
# Create viewer
viewer = napari.Viewer()
viewer.dims.ndisplay = 3 

# Add volumes as separate layers
viewer.add_image(binary_mask, name="Mask", colormap="cyan", rendering="average", contrast_limits=(0, 1), gamma = 0.4)
viewer.add_image(hole_mask.astype(np.uint8), name="Voids", colormap="magenta", blending="additive", contrast_limits=(0, 1), gamma = 0.4)

napari.run()
#%%
# create csv
HD_properties = masked_df[["Series", "Condition", "Line", "Timepoint", "Void_volume"]].copy()
HD_properties_void = HD_properties.explode("Void_volume").reset_index(drop=True)
HD_properties_void.to_excel(parentdir / "Void_volume.xlsx")

#%% check 
import napari
import numpy as np

combined_df = mask_df_expanded.merge(
    void_df[["Series", "Line", "CellID", "Condition", "Timepoint","Hole_mask"]],
    on=["Series", "CellID", "Line", "Condition", "Timepoint"],
    how="left"
)

vol_idx = 2
volume1 = combined_df["Individual_masks"].iloc[vol_idx]
volume2 = combined_df["Hole_mask"].iloc[vol_idx]

# Create viewer
viewer = napari.Viewer()
viewer.dims.ndisplay = 3 

# Add volumes as separate layers
viewer.add_image(volume1, name="Mask", colormap="cyan", rendering="average", contrast_limits=(0, 1), gamma = 0.4)
viewer.add_image(volume2.astype(np.uint8), name="Voids", colormap="magenta", blending="additive", 
                 contrast_limits=(0, 1), gamma = 0.4)

napari.run()

#%% isolate single example cell
vol_idx = 88
vol_idx = 150
vol_idx = 17
img_df = combined_df[combined_df["Series"] == "Series010"]

volume1 = img_df["Individual_masks"].iloc[vol_idx]
volume2 = img_df["Hole_mask"].iloc[vol_idx]
volumefl = mask_df_expanded["Original_fl"].iloc[vol_idx]

volume1 = row.Individual_masks
vollume2 = row.Hole_mask

coords = np.where(volume1)

z0, y0, x0 = np.min(coords[0]), np.min(coords[1]), np.min(coords[2])
z1, y1, x1 = np.max(coords[0]) + 1, np.max(coords[1]) + 1, np.max(coords[2]) + 1
bbox = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

padding = (10,10,10)
vol_shape = isolated_cell.shape
pz, py, px = padding
# add padding
zsl, ysl, xsl = bbox
z0 = max(0, zsl.start - pz)
y0 = max(0, ysl.start - py)
x0 = max(0, xsl.start - px)
z1 = min(vol_shape[0], zsl.stop + pz)
y1 = min(vol_shape[1], ysl.stop + py)
x1 = min(vol_shape[2], xsl.stop + px)
bbox_padded = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

cell_mask_cropped = volume1[bbox_padded]
void_mask_cropped = volume2[bbox_padded]
volumefl_cropped = volumefl[bbox_padded]

# Create viewer
viewer = napari.Viewer()
viewer.dims.ndisplay = 3 

# Add volumes as separate layers
#viewer.add_image(volumefl_cropped, name="FL", colormap="blue", rendering="average", contrast_limits=(0, 1), gamma = 0.4)
viewer.add_image(cell_mask_cropped, name="Mask", colormap="cyan", rendering="average", contrast_limits=(0, 1), gamma = 0.4)
viewer.add_image(void_mask_cropped.astype(np.uint8), name="Voids", colormap="magenta", rendering = "iso", 
                 iso_threshold = 0, blending="additive", contrast_limits=(0, 1), gamma = 0.4)

napari.run()

#plt.imshow(cell_mask_cropped[10,:,:])
#%% show original fluorescence
volumefl = mask_df_expanded["Original_fl"].iloc[vol_idx]
volumefl_cropped = volumefl[bbox_padded]
#volumefl_cropped[volumefl_cropped < 3000] = 0

z,_,_ = volumefl_cropped.shape
for z_slice in range(z): 
    plt.figure()
    plt.imshow(volumefl_cropped[z_slice,:,:], cmap = "Greens_r")
    plt.axis("off")

tiff.imwrite(constants.FIG_DIR / "void" / 'original_fl_cropped.tiff', volumefl_cropped)

#%% napari images for figures - overlay
import napari
import numpy as np
import imageio
from napari.utils import io as napari_io
import constants

cell_mask_cropped = volume1[bbox_padded]
void_mask_cropped = volume2[bbox_padded]
tiff.imwrite(parentdir / "cell_mask_vol.tiff", cell_mask_cropped)
tiff.imwrite(parentdir / "void_mask_vol.tiff", void_mask_cropped)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3 

layer1 = viewer.add_image(cell_mask_cropped,
                         name='macrophage',
                         colormap='cyan',
                         rendering='average',
                         scale=(0.49, 0.48147, 0.48147), # example voxel sizes (z,y,x)
                         gamma = 0.4) 

# Configure contrast limits and opacity
layer1.contrast_limits = (0, 1)
layer1.opacity = 0.9

layer2 = viewer.add_image(void_mask_cropped,
                         name='Voids',
                         colormap='magenta',
                         rendering = "iso",
                         iso_threshold = 0,
                         blending='additive',
                         scale=(0.49, 0.48147, 0.48147),
                         gamma = 0.4)  # example voxel sizes (z,y,x)

# Configure contrast limits and opacity
layer2.contrast_limits = (0,1)
layer2.opacity = 0.9

# Example: set the 3D camera orientation and zoom
# (viewer.camera attributes differ between napari versions; the following are commonly available)
try:
    viewer.camera.angles = (-13, -14, -28)   # if supported in your napari version
except Exception:
    # fallback: set dims slider positions to center the volume
    for dim in range(viewer.dims.ndim):
        viewer.dims.set_point(dim, volume.shape[dim] // 2)
        
# Take a screenshot (returns numpy array)
img = viewer.screenshot(canvas_only=True)  # canvas_only to avoid UI chrome
# Save
imageio.imwrite(constants.FIG_DIR / "void" / 'segmentation_overlay.png', img)
napari.run()

# keep the window open for inspection, or the with-block exits and the app closes

#%% napari images for figures - cell only

import napari
import numpy as np
import imageio
from napari.utils import io as napari_io
import constants

cell_mask_cropped = volume1[bbox_padded]
void_mask_cropped = volume2[bbox_padded]

viewer = napari.Viewer()
viewer.dims.ndisplay = 3 

layer1 = viewer.add_image(cell_mask_cropped,
                         name='macrophage',
                         colormap='cyan',
                         rendering='average',
                         scale=(0.49, 0.48147, 0.48147), # example voxel sizes (z,y,x)
                         gamma = 0.4) 

# Configure contrast limits and opacity
layer1.contrast_limits = (0, 1)
layer1.opacity = 0.9

# Example: set the 3D camera orientation and zoom
# (viewer.camera attributes differ between napari versions; the following are commonly available)
try:
    viewer.camera.angles = (-13, -14, -28)   # if supported in your napari version
except Exception:
    # fallback: set dims slider positions to center the volume
    for dim in range(viewer.dims.ndim):
        viewer.dims.set_point(dim, volume.shape[dim] // 2)
        
# Take a screenshot (returns numpy array)
img = viewer.screenshot(canvas_only=True)  # canvas_only to avoid UI chrome
# Save
imageio.imwrite(constants.FIG_DIR / "void" / 'segmentation_macrophage.png', img)
napari.run()
#%% cut through part of volume

cell_mask_half = cell_mask_cropped[:,0:90,:]
void_mask_half = void_mask_cropped[:,0:100,:]

viewer = napari.Viewer()
viewer.dims.ndisplay = 3 

layer1 = viewer.add_image(cell_mask_half,
                         name='macrophage',
                         colormap='cyan',
                         rendering='iso',
                         scale=(0.49, 0.48147, 0.48147), # example voxel sizes (z,y,x)
                         gamma = 0.4) 

# Configure contrast limits and opacity
layer1.contrast_limits = (0, 1)
layer1.opacity = 0.9

layer2 = viewer.add_image(void_mask_half,
                         name='Voids',
                         colormap='magenta',
                         blending='additive',
                         scale=(0.49, 0.48147, 0.48147),
                         gamma = 0.4)  # example voxel sizes (z,y,x)

# Configure contrast limits and opacity
layer2.contrast_limits = (0,1)
layer2.opacity = 0.9

# Example: set the 3D camera orientation and zoom
# (viewer.camera attributes differ between napari versions; the following are commonly available)
try:
    viewer.camera.angles = (-151, 30, -160)   # if supported in your napari version
except Exception:
    # fallback: set dims slider positions to center the volume
    for dim in range(viewer.dims.ndim):
        viewer.dims.set_point(dim, volume.shape[dim] // 2)

# Take a screenshot (returns numpy array)
img = viewer.screenshot(canvas_only=True)  # canvas_only to avoid UI chrome

# Save
imageio.imwrite(constants.FIG_DIR / 'segmentation_overlay_cross_section.png', img)
napari.run()


#%% compare void volume - Anova (all volumes combined)
from scipy.stats import ttest_ind, f_oneway
import seaborn as sns
from statannotations.Annotator import Annotator
import constants

comparison_col = "Line"
measure = "Void_volume"

df_clean = utils.remove_outliers_iqr(void_df, comparison_col, measure)

grouped = [group[measure].astype(float).values 
           for name, group in df_clean.groupby(comparison_col)]

if len(grouped) == 2:
    stat, p = ttest_ind(*grouped)
    print("Used t-test")
else:
    stat, p = f_oneway(*grouped)
    print("Used ANOVA")

# violin plot
order = sorted(pd.unique(df_clean[comparison_col].astype(str)), reverse=True)
palette = dict(zip(
    order,
    sns.color_palette("colorblind", n_colors=len(order))
))

fig, ax = plt.subplots(figsize=(6,6))
sns.violinplot(data=df_clean, x=comparison_col, y=measure, order=order, palette = constants.CMAP, hue=comparison_col,
ax=ax, cut=0)

# jittered points overlay

sns.stripplot(data=df_clean, x=comparison_col, y=measure, order=order, dodge=False,
              size=2, jitter=True, color="k", alpha=0.6, ax=ax)

pairs = [(order[0], order[1])]

# Create annotator
annotator = Annotator(
    ax,
    pairs,
    data=df_clean,
    x=comparison_col,
    y=measure,
    order=order
)
annotator.configure(text_format="star", loc="inside")
annotator.set_pvalues_and_annotate([p])

plt.tight_layout()


#%% count n

# -----------------PARAMETERS---------------------------------------

condition = "infected"

#-------------------------------------------------------------------

line_order = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]

void_df_uninfected = df_filtered = void_df[void_df["Condition"] == condition]
pivot = void_df_uninfected.pivot_table(
    index='Timepoint',
    columns='Line',
    aggfunc='size',
    fill_value=0
).reset_index()

pivot = pivot[['Timepoint'] + line_order]
utils.create_table(pivot, fig_size=(5,1))

#%% compare features - scatter plot
import constants
color_by = "Line"
order = sorted(void_df["Timepoint"].unique())
order = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]

sns.scatterplot(
    data=void_df,
    x="Cell_volume",
    y="Hole_vol_avg",
    hue=color_by,        # Lunch vs Dinner
    #size="size",       # Party size
    #style="sex" ,       # Marker style
    palette = constants.CMAP,
    hue_order=order
)

plt.show()

#%% compare void volume - individual cell averaged over time

import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
from scipy.stats import mannwhitneyu

all_feats = list(void_df.columns)

# -------------Parameters ---------------------------------------
comparison_col = "Line" # x axis comparison col
condition_col = "Timepoint" # comparison within comparison col
measure = "Hole_vol_avg"
measure = "Hole_number"

indiv_condition = "Condition"
indiv = "infected" # filter for only plotting this group
remove_outliers = True
# ------------------------ ---------------------------------------

# prepare data 
df_filtered = void_df[void_df[indiv_condition] == indiv]

if remove_outliers:
    print("Removing outliers by IQR")
    df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()
    df_clean[measure] = pd.to_numeric(df_clean[measure], errors="coerce")
    df_clean = df_clean.dropna(subset=[measure]).copy()
else:
    df_clean = df_filtered.copy()

# detect labels for 0 and 4 (supports numeric or string)
unique_tps = pd.unique(df_clean[condition_col].astype(str))
tp_to_float = {}
for tp in unique_tps:
    try:
        tp_to_float[tp] = float(tp)
    except Exception:
        tp_to_float[tp] = None
time0_labels = [tp for tp, val in tp_to_float.items() if val is not None and val == 0.0] or [tp for tp in unique_tps if tp == "0"]
time4_labels = [tp for tp, val in tp_to_float.items() if val is not None and val == 4.0] or [tp for tp in unique_tps if tp == "4"]
if not time0_labels or not time4_labels:
    raise ValueError(f"Couldn't find Timepoint labels for 0 and 4. Detected: {list(unique_tps)}")
time0_label = time0_labels[0]
time4_label = time4_labels[0]

# restrict data to only those two timepoints (keeps plot clean)
df_plot = df_clean[df_clean[condition_col].astype(str).isin([time0_label, time4_label])].copy()
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
lines = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]
hue_order = [time0_label, time4_label]

# helper for p-values
def welch_p(a_vals, b_vals):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return np.nan
    _, p = ttest_ind(a_vals, b_vals, equal_var=False, nan_policy="omit")
    return p

# Build pair lists and p-value lists for each family
within_pairs = []          # ((Line, TP0), (Line, TP4))
within_pvals = []

between0_pairs = []        # ((LineA, TP0), (LineB, TP0))
between0_pvals = []

between4_pairs = []        # ((LineA, TP4), (LineB, TP4))
between4_pvals = []

# Within-line (0 vs 4)
for L in lines:
    a = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    b = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    #p = welch_p(a, b)
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    if not np.isnan(p):
        within_pairs.append(((L, time0_label), (L, time4_label)))
        within_pvals.append(p)
'''        
# Between-lines at time0 and time4 (all pairwise line combinations)
for aL, bL in itertools.combinations(lines, 2):
    a0 = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    b0 = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    p0 = welch_p(a0, b0)
    if not np.isnan(p0):
        between0_pairs.append(((aL, time0_label), (bL, time0_label)))
        between0_pvals.append(p0)

    a4 = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    b4 = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    p4 = welch_p(a4, b4)
    if not np.isnan(p4):
        between4_pairs.append(((aL, time4_label), (bL, time4_label)))
        between4_pvals.append(p4)
'''

# Apply FDR correction per family
within_corr = multipletests(within_pvals, method="fdr_bh")[1] if within_pvals else []
#between0_corr = multipletests(between0_pvals, method="fdr_bh")[1] if between0_pvals else []
#between4_corr = multipletests(between4_pvals, method="fdr_bh")[1] if between4_pvals else []

# Create single grouped violin plot: x=Line, hue=Timepoint (0 and 4)
#plt.figure(figsize=(max(8, len(lines) * 1.2), 6))
plt.figure(figsize=(5,4))

ax = sns.violinplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                    order=lines, hue_order=hue_order, split=False, cut=0, palette=constants.CMAP)
sns.stripplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col, order=lines,
              hue_order=hue_order, dodge=True, size=3, jitter=True, palette=["k","k"], alpha=0.6)

# The overlay above added a second legend; remove duplicate legend entries and keep single
handles, labels = ax.get_legend_handles_labels()
# keep only the first two (hue) and set the legend properly
if len(handles) >= 2:
    ax.legend(handles[:2], labels[:2], title=condition_col)

# Annotate using three separate Annotator instances (they accept pairs when both x and hue are used)
# 1) within-line
if within_pairs:
    annot_within = Annotator(ax, within_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                             order=lines, hue_order=hue_order)
    annot_within.configure(text_format="star", loc="inside")
    annot_within.set_pvalues_and_annotate(list(within_corr))
'''
# 2) between-lines at time0
if between0_pairs:
    annot_b0 = Annotator(ax, between0_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                         order=lines, hue_order=hue_order)
    annot_b0.configure(text_format="star", loc="outside", line_offset=6)  # put these outside so they don't overlap inside-line pairs
    annot_b0.set_pvalues_and_annotate(list(between0_corr))

# 3) between-lines at time4
if between4_pairs:
    annot_b4 = Annotator(ax, between4_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                         order=lines, hue_order=hue_order)
    annot_b4.configure(text_format="star", loc="outside", line_offset=12)  # offset further out so both timepoint families fit
    annot_b4.set_pvalues_and_annotate(list(between4_corr))
'''
ax.set_xlabel("")
ax.set_ylabel("")
#ax.set_title(f"{measure}: Time {time0_label} vs {time4_label} per Line; within-line and between-line comparisons")
plt.tight_layout()
plt.show()

#%% compare individual cell voids between lines
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import matplotlib.patches as mpatches

# ---------------- settings ----------------
comparison_col = "Line"
condition_col = "Timepoint"
measure = "Hole_number"
indiv_condition = "Condition"
indiv = "infected" # filter for only plotting this group

plot_timepoint_value = 4
# ------------------------------------------

# Prepare data
df_filtered = void_df[void_df[indiv_condition] == indiv]
df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()


# Detect timepoint label corresponding to 0
unique_tps = pd.unique(df_clean[condition_col].astype(str))
tp_label = None
for tp in unique_tps:
    try:
        if float(tp) == float(plot_timepoint_value):
            tp_label = tp
            break
    except:
        continue

if tp_label is None:
    raise ValueError(f"Could not find Timepoint {plot_timepoint_value}")

# Filter to Time 0
df_plot = df_clean[df_clean[condition_col].astype(str) == tp_label].copy()
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
lines = lines_ordered
ylimit = round(np.max(df_plot[measure])*1.75)

# Generate color palette per line
palette = dict(zip(lines, sns.color_palette("colorblind", n_colors=len(lines))))
palette = constants.CMAP

# ---- Pairwise Welch tests between lines ----
pairs = []
pvals = []

for aL, bL in itertools.combinations(lines, 2):
    a_vals = df_plot[df_plot[comparison_col].astype(str) == aL][measure].astype(float)
    b_vals = df_plot[df_plot[comparison_col].astype(str) == bL][measure].astype(float)
    if len(a_vals) < 2 or len(b_vals) < 2:
        continue
    #_, p = ttest_ind(a_vals, b_vals, equal_var=False)
    #p = welch_p(a, b)
    stat, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    pairs.append((aL, bL))
    pvals.append(p)

if not pairs:
    raise ValueError("Not enough samples for pairwise comparisons.")

# FDR correction
_, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

# ---- Plot ----
plt.figure(figsize=(max(8, len(lines)*1.2), 6))
ax = sns.violinplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    palette=palette,
    cut=0
)

sns.stripplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    color="k",
    size=3,
    jitter=True,
    alpha=0.6
)

# Annotate comparisons
annot = Annotator(ax, pairs, data=df_plot, x=comparison_col, y=measure, order=lines)
annot.configure(text_format="star", loc="outside")
annot.set_pvalues_and_annotate(list(pvals_corr))

# ---- Custom legend showing Lines ----
legend_handles = [mpatches.Patch(color=palette[line], label=line) for line in lines]
ax.legend(handles=legend_handles, title="Line", bbox_to_anchor=(1.02, 1), loc="upper left")

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(top = ylimit)
#ax.set_title(f"{measure} at Time {tp_label} — Between-Line Comparisons")

plt.tight_layout()
plt.show()

#%% compare per line and per condition 
# Compare Void_volume per Line: Infected vs Uninfected (plot one subplot per Line)
from scipy.stats import ttest_ind, f_oneway
from statannotations.Annotator import Annotator
import numpy as np
import math

comparison_col = "Line"
condition_col = "Condition"   
measure = "Void_volume"

# clean and numeric conversion (reuse your outlier removal)
df_clean = utils.remove_outliers_iqr(HD_properties_void, comparison_col, measure).copy()
df_clean[measure] = pd.to_numeric(df_clean[measure], errors="coerce")
df_clean = df_clean.dropna(subset=[measure])

lines = sorted(pd.unique(df_clean[comparison_col].astype(str)), reverse=True)
n_lines = len(lines)

# layout: try up to 3 columns
ncols = min(3, n_lines) if n_lines > 1 else 1
nrows = math.ceil(n_lines / ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
axes = axes.flatten()

# palette: prefer constants.CMAP if present, else create mapping
try:
    global_palette = constants.CMAP
except Exception:
    global_palette = None

for i, line in enumerate(lines):
    ax = axes[i]
    sub = df_clean[df_clean[comparison_col].astype(str) == line].copy()
    # get conditions present for this line (preserve original labels)
    conds = sorted(pd.unique(sub[condition_col].astype(str)))
    if len(sub) == 0:
        ax.set_title(f"{line} (no data)")
        ax.axis("off")
        continue

    # choose palette for conditions
    if isinstance(global_palette, dict):
        # global_palette might be mapping by Line in your code; if it's keyed by line, use that for background
        # but for hue we want per-condition colors
        palette = dict(zip(conds, sns.color_palette("colorblind", n_colors=len(conds))))
    else:
        palette = dict(zip(conds, sns.color_palette("colorblind", n_colors=len(conds))))

    # If you want Condition on x-axis (so categories are side-by-side), do x=condition_col and facet by Line
    sns.violinplot(
        data=sub,
        x=condition_col,
        y=measure,
        order=conds,
        palette=palette,
        ax=ax,
        cut=0
    )

    sns.stripplot(
        data=sub,
        x=condition_col,
        y=measure,
        order=conds,
        dodge=False,
        size=2,
        jitter=True,
        color="k",
        alpha=0.6,
        ax=ax
    )

    # If exactly two conditions -> pairwise t-test and annotate that pair.
    title_extra = ""
    if len(conds) == 1:
        title_extra = " — only one condition"
    elif len(conds) == 2:
        arrs = [sub.loc[sub[condition_col].astype(str) == c, measure].astype(float).values for c in conds]
        # if either group is empty or too small, skip test
        if len(arrs[0]) < 2 or len(arrs[1]) < 2:
            p = np.nan
            title_extra = " — too few samples for t-test"
        else:
            stat, p = ttest_ind(arrs[0], arrs[1], equal_var=False, nan_policy="omit")  # Welch t-test
            title_extra = f" — t-test p={p:.3g}"
            pairs = [(conds[0], conds[1])]
            annot = Annotator(ax, pairs, data=sub, x=condition_col, y=measure, order=conds)
            annot.configure(text_format="star", loc="inside")
            annot.set_pvalues_and_annotate([p])
    else:
        # >2 conditions: ANOVA across groups
        groups = [sub.loc[sub[condition_col].astype(str) == c, measure].astype(float).values for c in conds]
        # only include groups with at least one item
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            p = np.nan
            title_extra = " — not enough groups for ANOVA"
        else:
            try:
                stat, p = f_oneway(*groups)
                title_extra = f" — ANOVA p={p:.3g}"
            except Exception as e:
                p = np.nan
                title_extra = f" — ANOVA error: {e}"

            # Annotate the global ANOVA p-value at top-center
            ax.text(0.5, 0.95, f"ANOVA p={p:.3g}" if not np.isnan(p) else "ANOVA n/a",
                    transform=ax.transAxes, ha="center", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    ax.set_title(f"{line}{title_extra}")
    ax.set_xlabel("")  # keep x label clean
    ax.set_ylabel(measure)

# turn off any empty axes
for j in range(len(lines), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

#%% Hole volume distribution histogram

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from itertools import combinations
from scipy.stats import ttest_ind
from statannotations.Annotator import Annotator
import numpy as np

comparison_col = "Line"
condition_col = "Condition"
measure = "Void_volume"

df_clean = utils.remove_outliers_iqr(HD_properties_void, comparison_col, measure).copy()
df_clean[measure] = pd.to_numeric(df_clean[measure], errors="coerce")
df_clean = df_clean.dropna(subset=[measure])

# ---------------------------
# TWO-WAY ANOVA
# ---------------------------
model = smf.ols(f"{measure} ~ C({comparison_col}) * C({condition_col})", data=df_clean).fit()
anova_results = anova_lm(model, typ=2)

print("\nTwo-way ANOVA:")
print(anova_results)

# ---------------------------
# Plot
# ---------------------------
order = sorted(df_clean[comparison_col].unique(), reverse=True)
hue_order = sorted(df_clean[condition_col].unique())

fig, ax = plt.subplots(figsize=(7,6))

sns.violinplot(
    data=df_clean,
    x=comparison_col,
    y=measure,
    hue=condition_col,
    order=order,
    hue_order=hue_order,
    palette=constants.CMAP,
    cut=0,
    ax=ax
)

sns.stripplot(
    data=df_clean,
    x=comparison_col,
    y=measure,
    hue=condition_col,
    order=order,
    hue_order=hue_order,
    dodge=True,
    color="k",
    size=2,
    alpha=0.5,
    ax=ax
)

# remove duplicate legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(hue_order)], labels[:len(hue_order)], title=condition_col)

plt.tight_layout()
plt.show()
#%% plot quartile
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

def compare_line_condition(
    df,
    comparison_col="Line",
    condition_col="Condition",
    measure="Void_volume",
    remove_outliers=True,
    correction="holm",            # None | "holm" | "bonferroni"
    alpha=0.05,
    only_annotate_significant=True,
    violin_palette=None,
    figsize=(12,6)
):
    """
    Compare within-line (Conditions in same Line) and between-lines (same Condition across Lines).
    Plots x=Line, hue=Condition (violin + strip) and annotates significant comparisons.
    Returns: fig, ax, df_filt, results
    results keys: pairs_all, pvals_raw, pvals_corrected, annotated_pairs, annotated_pvals, labels
    """
    # copy and numeric coercion
    df = df.copy()
    df[measure] = pd.to_numeric(df[measure], errors="coerce")
    df = df.dropna(subset=[measure])

    # remove outliers per (Line,Condition) if requested
    if remove_outliers:
        df = remove_outliers_iqr(df, [comparison_col, condition_col], measure)

    # filtered dataset
    df_filt = df.copy()

    # plotting order
    order = sorted(pd.unique(df_filt[comparison_col].astype(str)), reverse=True)
    hue_order = sorted(pd.unique(df_filt[condition_col].astype(str)))

    # palette fallback
    if violin_palette is None:
        try:
            violin_palette = constants.CMAP
        except Exception:
            violin_palette = dict(zip(hue_order, sns.color_palette("colorblind", n_colors=len(hue_order))))

    # create plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df_filt,
        x=comparison_col,
        y=measure,
        hue=condition_col,
        order=order,
        hue_order=hue_order,
        palette=violin_palette,
        cut=0,
        dodge=True,
        ax=ax
    )
    sns.stripplot(
        data=df_filt,
        x=comparison_col,
        y=measure,
        hue=condition_col,
        order=order,
        hue_order=hue_order,
        dodge=True,
        size=2,
        jitter=True,
        color="k",
        alpha=0.5,
        ax=ax
    )

    # clean legend (keep only once)
    handles, labels = ax.get_legend_handles_labels()
    if len(hue_order) > 0:
        ax.legend(handles[:len(hue_order)], labels[:len(hue_order)], title=condition_col)
    else:
        ax.legend_.remove()

    # helpers
    def safe_ttest(a, b):
        a = np.asarray(a).astype(float)
        b = np.asarray(b).astype(float)
        if len(a) < 2 or len(b) < 2:
            return np.nan
        try:
            _, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            return p
        except Exception:
            return np.nan

    def safe_anova(list_of_arrays):
        groups = [np.asarray(g).astype(float) for g in list_of_arrays if len(g) > 0]
        if len(groups) < 2:
            return np.nan
        try:
            _, p = f_oneway(*groups)
            return p
        except Exception:
            return np.nan

    # Build ALL candidate comparisons and compute raw p-values
    pairs = []        # Annotator-format pairs
    labels = []       # human-readable labels for results
    pvals = []        # raw p-values in same order

    # 1) Within-line: for each Line, compare Conditions (pairwise or ANOVA)
    for line in order:
        sub = df_filt[df_filt[comparison_col].astype(str) == line]
        conds_present = sorted(pd.unique(sub[condition_col].astype(str)))
        if len(conds_present) < 2:
            continue

        if len(conds_present) == 2:
            a, b = conds_present
            arr_a = sub[sub[condition_col].astype(str) == a][measure].values
            arr_b = sub[sub[condition_col].astype(str) == b][measure].values
            p = safe_ttest(arr_a, arr_b)
            pairs.append(((line, a), (line, b)))
            labels.append(f"Within-line {line}: {a} vs {b}")
            pvals.append(p)
        else:
            # ANOVA across conditions for this line — annotate using the first two conditions as anchor
            groups = [sub[sub[condition_col].astype(str) == c][measure].values for c in conds_present]
            p = safe_anova(groups)
            pairs.append(((line, conds_present[0]), (line, conds_present[1])))
            labels.append(f"Within-line {line}: ANOVA across {len(conds_present)} conditions")
            pvals.append(p)

    # 2) Between-lines for each Condition: compare Lines pairwise within each Condition
    for cond in hue_order:
        sub = df_filt[df_filt[condition_col].astype(str) == cond]
        lines_present = sorted(pd.unique(sub[comparison_col].astype(str)), reverse=True)
        if len(lines_present) < 2:
            continue
        for a, b in itertools.combinations(lines_present, 2):
            arr_a = sub[sub[comparison_col].astype(str) == a][measure].values
            arr_b = sub[sub[comparison_col].astype(str) == b][measure].values
            p = safe_ttest(arr_a, arr_b)
            pairs.append(((a, cond), (b, cond)))
            labels.append(f"Between-lines {cond}: {a} vs {b}")
            pvals.append(p)

    # Multiple-testing correction
    pvals_array = np.array(pvals, dtype=float)
    pvals_corrected = np.full_like(pvals_array, np.nan, dtype=float)
    if correction in ("holm", "bonferroni"):
        finite_mask = np.isfinite(pvals_array)
        if finite_mask.any():
            rej, p_corr, _, _ = multipletests(pvals_array[finite_mask], alpha=alpha, method=correction)
            pvals_corrected[finite_mask] = p_corr
    else:
        pvals_corrected = pvals_array.copy()

    # Select only significant pairs for annotation if requested
    if only_annotate_significant:
        sig_mask = np.isfinite(pvals_corrected) & (pvals_corrected < alpha)
    else:
        sig_mask = np.isfinite(pvals_corrected)  # annotate only computed pvals (no NaNs)

    annotated_pairs = [pairs[i] for i in range(len(pairs)) if sig_mask[i]]
    annotated_pvals = [float(pvals_corrected[i]) for i in range(len(pvals)) if sig_mask[i]]
    annotated_labels = [labels[i] for i in range(len(labels)) if sig_mask[i]]

    # Annotate using statannotations Annotator (only if at least one pair)
    if annotated_pairs:
        annot = Annotator(
            ax,
            annotated_pairs,
            data=df_filt,
            x=comparison_col,
            y=measure,
            hue=condition_col,
            order=order,
            hue_order=hue_order
        )
        annot.configure(text_format="star", loc="outside")
        annot.set_pvalues_and_annotate(annotated_pvals)
    else:
        print("No significant comparisons to annotate (after correction).")

    ax.set_title(f"{measure} — compare within-line and between-lines (alpha={alpha}, correction={correction})")
    ax.set_xlabel(comparison_col)
    ax.set_ylabel(measure)
    plt.tight_layout()

    results = {
        "pairs_all": pairs,
        "labels_all": labels,
        "pvals_raw": list(pvals),
        "pvals_corrected": list(pvals_corrected),
        "annotated_pairs": annotated_pairs,
        "annotated_pvals": annotated_pvals,
        "annotated_labels": annotated_labels
    }

    return fig, ax, df_filt, results

df_filtered = HD_properties_void[HD_properties_void["Condition"] == "infected"]
df_clean = utils.remove_outliers_iqr(HD_properties_void, ["Line","Timepoint"], "Void_volume")
fig, ax, df_filt, results = compare_line_condition(
    df_clean,
    comparison_col="Line",
    condition_col="Condition",
    measure="Void_volume",
    remove_outliers=False,      # already removed
    correction=None,
    alpha=0.05,
    only_annotate_significant=True,
    figsize=(12,6)
)
#%% distribution of void volume (histogram) - population
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

void_df_expanded = void_df.explode("Hole_volumes").reset_index(drop=True)

#----------- PARAMETERS --------------------
condition = "infected"
comparison_col = "Line"
condition_col = "Timepoint"
measure = "Hole_volumes"
lines_ordered = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]
#-------------------------------------------

df_filtered = void_df_expanded[void_df_expanded["Condition"] == condition]
df_clean = utils.remove_outliers_iqr(df_filtered, ["Timepoint", "Line"], measure).copy()
df_plot = df_clean.copy()

xlimit = np.max(df_plot[measure]) * 1.5

g = sns.displot(
    data=df_plot,
    x=measure,
    col=comparison_col,      # separate panel per Line
    col_order = lines_ordered,
    hue=condition_col,       # color by Condition
    kind="hist",
    bins=50,
    element="step",          # cleaner overlay
    stat="probability", # count for raw counts
    common_norm=False,
    height=4,
    aspect=1,
)

g.set_axis_labels("Void volume (\u00B5m\u00B3)", "Count")
g.set_axis_labels("","")
g.set(xlim=(0, xlimit), ylim=(0, 0.75))
g.set_titles("")
#g.fig.suptitle(f"{measure} distribution by {comparison_col} and {condition_col}", y=1.05)

# Tick label size
for ax in g.axes.flat:
    ax.tick_params(axis='both', labelsize=12)
    
plt.tight_layout()
plt.show()

#%% lines void volume violin - population

# Difference between infected and uninfected
#---------PARAMETERS--------------
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import constants 

# columns / measure
comparison_col = "Line"       # x-axis grouping
condition_col = "Condition"   # hue (we'll compare two levels here)
measure = "Hole_volumes"

condition_col = "Condition"
comparison_col = "Line"
constant_col = "Timepoint"
constant = 0
#---------------------------------

# params: choose the two condition labels you want to compare.
# If either is None, the code will pick the first two unique conditions it finds.
condA = "infected"   # e.g. "Control"
condB = "uninfected"   # e.g. "Treated"

# prepare data (do not filter by equality to a literal value)
df = void_df_expanded.copy()

# remove outliers by IQR using your utils function (keeps both grouping columns)
df_filtered = df[df[constant_col] == str(constant)]
df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()

# detect two condition labels
unique_conds = sorted(pd.unique(df_clean[condition_col].astype(str)))
if condA is None or condB is None:
    if len(unique_conds) < 2:
        raise ValueError(f"Need at least two condition levels in column '{condition_col}'. Found: {unique_conds}")
    # default: pick first two unique values
    condA = condA or unique_conds[0]
    condB = condB or unique_conds[1]

if condA not in unique_conds or condB not in unique_conds:
    raise ValueError(f"Couldn't find requested conditions. Available: {unique_conds}. Requested: {condA}, {condB}")

# restrict to only the two conditions (keeps plot clean)
df_plot = df_clean[df_clean[condition_col].astype(str).isin([condA, condB])].copy()

# lines (order). keep your custom ordering if you want
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
# if you have a preferred explicit order, set it here (like previously)
lines = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]

hue_order = [condA, condB]

# helper for p-values (Welch)
def welch_p(a_vals, b_vals):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return np.nan
    _, p = ttest_ind(a_vals, b_vals, equal_var=False, nan_policy="omit")
    return p

# Build pair lists and p-value lists for each family (within-line comparisons between conditions)
within_pairs = []    # ((Line, condA), (Line, condB))
within_pvals = []

# compute within-line p-values (condA vs condB for each Line)
for L in lines:
    a = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == condA)][measure].astype(float).values
    b = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == condB)][measure].astype(float).values
    p = welch_p(a, b)
    if not np.isnan(p):
        within_pairs.append(((L, condA), (L, condB)))
        within_pvals.append(p)

# (optional) between-lines comparisons at each condition (commented out; enable if you want)
# betweenA_pairs = []
# betweenA_pvals = []
# betweenB_pairs = []
# betweenB_pvals = []
# for aL, bL in itertools.combinations(lines, 2):
#     aA = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == condA)][measure].astype(float).values
#     bA = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == condA)][measure].astype(float).values
#     pA = welch_p(aA, bA)
#     if not np.isnan(pA):
#         betweenA_pairs.append(((aL, condA), (bL, condA)))
#         betweenA_pvals.append(pA)
#
#     aB = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == condB)][measure].astype(float).values
#     bB = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == condB)][measure].astype(float).values
#     pB = welch_p(aB, bB)
#     if not np.isnan(pB):
#         betweenB_pairs.append(((aL, condB), (bL, condB)))
#         betweenB_pvals.append(pB)

# Apply FDR correction per family
within_corr = multipletests(within_pvals, method="fdr_bh")[1] if within_pvals else []
# betweenA_corr = multipletests(betweenA_pvals, method="fdr_bh")[1] if betweenA_pvals else []
# betweenB_corr = multipletests(betweenB_pvals, method="fdr_bh")[1] if betweenB_pvals else []

# Create grouped violin plot: x=Line, hue=Condition (condA vs condB)
plt.figure(figsize=(max(8, len(lines) * 1.2), 6))
ax = sns.violinplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                    order=lines, hue_order=hue_order, split=False, cut=0, palette=constants.CMAP)
# overlay points; use dodge to separate the two hue groups
sns.stripplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col, order=lines,
              hue_order=hue_order, dodge=True, size=3, jitter=True, palette=["k","k"], alpha=0.6)

# fix duplicate legend from the two layer calls
handles, labels = ax.get_legend_handles_labels()
if len(handles) >= 2:
    ax.legend(handles[:2], labels[:2], title=condition_col)

# Annotate: within-line comparisons (condA vs condB per Line)
if within_pairs:
    annot_within = Annotator(ax, within_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                             order=lines, hue_order=hue_order)
    annot_within.configure(text_format="star", loc="inside")
    annot_within.set_pvalues_and_annotate(list(within_corr))

# (optional) annotate between-lines if you enabled that code above
# if betweenA_pairs:
#     annot_bA = Annotator(ax, betweenA_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
#                          order=lines, hue_order=hue_order)
#     annot_bA.configure(text_format="star", loc="outside", line_offset=6)
#     annot_bA.set_pvalues_and_annotate(list(betweenA_corr))
#
# if betweenB_pairs:
#     annot_bB = Annotator(ax, betweenB_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
#                          order=lines, hue_order=hue_order)
#     annot_bB.configure(text_format="star", loc="outside", line_offset=12)
#     annot_bB.set_pvalues_and_annotate(list(betweenB_corr))

ax.set_xlabel("")
ax.set_ylabel("")
#ax.set_title(f"{measure}: {condA} vs {condB} per {comparison_col} (within-line comparisons)")
plt.tight_layout()
plt.show()

#%% compare t0 to t0 between lines
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu

# ---------------- settings ----------------
comparison_col = "Line" # grouping col along x
condition_col = "Timepoint" # constant col
measure = "Hole_volumes"
condition = "infected" # constant col
plot_timepoint_value = str(0)
# ------------------------------------------

# Prepare data
df_filtered = void_df_expanded[void_df_expanded["Condition"] == condition]
df_clean = utils.remove_outliers_iqr(df_filtered, ["Timepoint", "Line"], measure).copy()
ylimit = np.max(df_clean[measure])*1.75

# Filter to Time 0
df_plot = df_clean[df_clean[condition_col].astype(str) == plot_timepoint_value].copy()
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
lines = lines_ordered

# Generate color palette per line
palette = dict(zip(lines, sns.color_palette("colorblind", n_colors=len(lines))))
palette = constants.CMAP

# ---- Pairwise tests between lines ----
pairs = []
pvals = []

for aL, bL in itertools.combinations(lines, 2):
    a_vals = df_plot[df_plot[comparison_col].astype(str) == aL][measure].astype(float)
    b_vals = df_plot[df_plot[comparison_col].astype(str) == bL][measure].astype(float)
    if len(a_vals) < 2 or len(b_vals) < 2:
        continue
    #_, p = ttest_ind(a_vals, b_vals, equal_var=False)
    stat, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    pairs.append((aL, bL))
    pvals.append(p)

if not pairs:
    raise ValueError("Not enough samples for pairwise comparisons.")

# FDR correction
_, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

# ---- Plot ----
plt.figure(figsize=(max(8, len(lines)*1.2), 6))
ax = sns.violinplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    palette=palette,
    cut=0
)

sns.stripplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    color="k",
    size=3,
    jitter=True,
    alpha=0.6
)

# Annotate comparisons
annot = Annotator(ax, pairs, data=df_plot, x=comparison_col, y=measure, order=lines)
annot.configure(text_format="star", loc="outside")
annot.set_pvalues_and_annotate(list(pvals_corr))

# ---- Custom legend showing Lines ----
legend_handles = [mpatches.Patch(color=palette[line], label=line) for line in lines]
ax.legend(handles=legend_handles, title="Line", bbox_to_anchor=(1.02, 1), loc="upper left")

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(top = ylimit)
#ax.set_title(f"{measure} at Time {tp_label} — Between-Line Comparisons")

plt.tight_layout()
plt.show()

#%% compare upper quantile

def bootstrap_quantile(x, q=0.75, n_boot=1000):
    x = np.array(x)
    boots = []
    for _ in range(n_boot):
        sample = np.random.choice(x, size=len(x), replace=True)
        boots.append(np.quantile(sample, q))
    return np.array(boots)

# ---------------- settings ----------------
comparison_col = "Line" # grouping col along x
condition_col = "Timepoint" # constant col
measure = "Hole_volumes"
condition = "uninfected" # constant col
plot_timepoint_value = str(4)
# ------------------------------------------

# Prepare data
df_filtered = void_df_expanded[void_df_expanded["Condition"] == condition]
df_clean = utils.remove_outliers_iqr(df_filtered, ["Timepoint", "Line"], measure).copy()
ylimit = np.max(df_clean[measure])*1.75

# Filter to Time 0
df_plot = df_clean[df_clean[condition_col].astype(str) == plot_timepoint_value].copy()
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
lines = lines_ordered

# Generate color palette per line
palette = dict(zip(lines, sns.color_palette("colorblind", n_colors=len(lines))))
palette = constants.CMAP

pairs = []
pvals = []

for aL, bL in itertools.combinations(lines, 2):
    a_vals = df_plot[df_plot[comparison_col].astype(str) == aL][measure].astype(float)
    b_vals = df_plot[df_plot[comparison_col].astype(str) == bL][measure].astype(float)

    if len(a_vals) < 5 or len(b_vals) < 5:
        continue

    # bootstrap distributions of 75th percentile
    a_boot = bootstrap_quantile(a_vals, q=0.75)
    b_boot = bootstrap_quantile(b_vals, q=0.75)

    # difference distribution
    diff = a_boot - b_boot

    # two-sided p-value
    p = 2 * min(
        np.mean(diff <= 0),
        np.mean(diff >= 0)
    )

    pairs.append((aL, bL))
    pvals.append(p)
    
# FDR correction
_, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

# ---- Plot ----
plt.figure(figsize=(max(8, len(lines)*1.2), 6))
ax = sns.violinplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    palette=palette,
    cut=0
)

sns.stripplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    color="k",
    size=3,
    jitter=True,
    alpha=0.6
)

# Annotate comparisons
annot = Annotator(ax, pairs, data=df_plot, x=comparison_col, y=measure, order=lines)
annot.configure(text_format="star", loc="outside")
annot.set_pvalues_and_annotate(list(pvals_corr))

# ---- Custom legend showing Lines ----
legend_handles = [mpatches.Patch(color=palette[line], label=line) for line in lines]
ax.legend(handles=legend_handles, title="Line", bbox_to_anchor=(1.02, 1), loc="upper left")

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(top = ylimit)
#ax.set_title(f"{measure} at Time {tp_label} — Between-Line Comparisons")

plt.tight_layout()
plt.show() 
#%% plot table of all void averages

condition = "infected"
df_filtered = void_df_expanded[void_df_expanded["Condition"] == condition]
df_filtered = df_filtered[df_filtered["Timepoint"] == str(4)]

summary_df = df_filtered.groupby('Line')['Hole_volumes'].agg(
    void_mean='mean',
    void_std='std'
).round(2).reset_index()

summary_df["void_mean"] = (
    summary_df["void_mean"].round(2)
)
summary_df = summary_df.rename(columns = {"void_mean": "x\u0304 Void Volume (\u00B5m\u00B3)", "void_std": "Void Volume \u03C3"})
summary_df = summary_df.set_index('Line').reindex(lines_ordered).reset_index()

utils.create_table(summary_df, body_fontsize=8, header_fontsize=8, fig_size=(5,1.25))

#%% violin plot population void volume over time 

import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
all_feats = list(void_df.columns)

# -------------Parameters ---------------------------------------
comparison_col = "Line" # x axis comparison col
condition_col = "Timepoint" # comparison within comparison col
measure = "Hole_volumes"
indiv_condition = "Condition"
indiv = "infected" # filter for only plotting this group
remove_outliers = True
# ------------------------ ---------------------------------------

# prepare data 
df_filtered = void_df_expanded[void_df_expanded[indiv_condition] == indiv]

if remove_outliers:
    print("Removing outliers by IQR")
    df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()
    df_clean[measure] = pd.to_numeric(df_clean[measure], errors="coerce")
    df_clean = df_clean.dropna(subset=[measure]).copy()
else:
    df_clean = df_filtered.copy()

# detect labels for 0 and 4 (supports numeric or string)
unique_tps = pd.unique(df_clean[condition_col].astype(str))
tp_to_float = {}
for tp in unique_tps:
    try:
        tp_to_float[tp] = float(tp)
    except Exception:
        tp_to_float[tp] = None
time0_labels = [tp for tp, val in tp_to_float.items() if val is not None and val == 0.0] or [tp for tp in unique_tps if tp == "0"]
time4_labels = [tp for tp, val in tp_to_float.items() if val is not None and val == 4.0] or [tp for tp in unique_tps if tp == "4"]
if not time0_labels or not time4_labels:
    raise ValueError(f"Couldn't find Timepoint labels for 0 and 4. Detected: {list(unique_tps)}")
time0_label = time0_labels[0]
time4_label = time4_labels[0]

# restrict data to only those two timepoints (keeps plot clean)
df_plot = df_clean[df_clean[condition_col].astype(str).isin([time0_label, time4_label])].copy()
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
lines = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]
hue_order = [time0_label, time4_label]

# helper for p-values
def welch_p(a_vals, b_vals):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return np.nan
    _, p = ttest_ind(a_vals, b_vals, equal_var=False, nan_policy="omit")
    return p

# Build pair lists and p-value lists for each family
within_pairs = []          # ((Line, TP0), (Line, TP4))
within_pvals = []

between0_pairs = []        # ((LineA, TP0), (LineB, TP0))
between0_pvals = []

between4_pairs = []        # ((LineA, TP4), (LineB, TP4))
between4_pvals = []

# Within-line (0 vs 4)
for L in lines:
    a = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    b = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    p = welch_p(a, b)
    if not np.isnan(p):
        within_pairs.append(((L, time0_label), (L, time4_label)))
        within_pvals.append(p)
'''        
# Between-lines at time0 and time4 (all pairwise line combinations)
for aL, bL in itertools.combinations(lines, 2):
    a0 = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    b0 = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    p0 = welch_p(a0, b0)
    if not np.isnan(p0):
        between0_pairs.append(((aL, time0_label), (bL, time0_label)))
        between0_pvals.append(p0)

    a4 = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    b4 = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    p4 = welch_p(a4, b4)
    if not np.isnan(p4):
        between4_pairs.append(((aL, time4_label), (bL, time4_label)))
        between4_pvals.append(p4)
'''
# Apply FDR correction per family
within_corr = multipletests(within_pvals, method="fdr_bh")[1] if within_pvals else []
#between0_corr = multipletests(between0_pvals, method="fdr_bh")[1] if between0_pvals else []
#between4_corr = multipletests(between4_pvals, method="fdr_bh")[1] if between4_pvals else []

# Create single grouped violin plot: x=Line, hue=Timepoint (0 and 4)
#plt.figure(figsize=(max(8, len(lines) * 1.2), 6))
plt.figure(figsize=(5,4))

ax = sns.violinplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                    order=lines, hue_order=hue_order, split=False, cut=0, palette=constants.CMAP)
sns.stripplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col, order=lines,
              hue_order=hue_order, dodge=True, size=3, jitter=True, palette=["k","k"], alpha=0.6)

# The overlay above added a second legend; remove duplicate legend entries and keep single
handles, labels = ax.get_legend_handles_labels()
# keep only the first two (hue) and set the legend properly
if len(handles) >= 2:
    ax.legend(handles[:2], labels[:2], title=condition_col)

# Annotate using three separate Annotator instances (they accept pairs when both x and hue are used)
# 1) within-line
if within_pairs:
    annot_within = Annotator(ax, within_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                             order=lines, hue_order=hue_order)
    annot_within.configure(text_format="star", loc="inside")
    annot_within.set_pvalues_and_annotate(list(within_corr))
'''
# 2) between-lines at time0
if between0_pairs:
    annot_b0 = Annotator(ax, between0_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                         order=lines, hue_order=hue_order)
    annot_b0.configure(text_format="star", loc="outside", line_offset=6)  # put these outside so they don't overlap inside-line pairs
    annot_b0.set_pvalues_and_annotate(list(between0_corr))

# 3) between-lines at time4
if between4_pairs:
    annot_b4 = Annotator(ax, between4_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                         order=lines, hue_order=hue_order)
    annot_b4.configure(text_format="star", loc="outside", line_offset=12)  # offset further out so both timepoint families fit
    annot_b4.set_pvalues_and_annotate(list(between4_corr))
'''
ax.set_xlabel("")
ax.set_ylabel("")
#ax.set_title(f"{measure}: Time {time0_label} vs {time4_label} per Line; within-line and between-line comparisons")
plt.tight_layout()
plt.show()