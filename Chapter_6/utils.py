from scipy import ndimage
import tifffile as tiff
import os
import numpy as np
from pathlib import Path
import re
import cv2
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import reconstruction
import napari
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from collections import defaultdict, deque
from itertools import combinations
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt
from skimage import draw
import constants
from skimage import measure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import matplotlib.patches as mpatches
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def is_tiff(file: Path) -> bool:
    return file.is_file() and (file.suffix == ".tif" or file.suffix == ".tiff")

def is_csv(file: Path) -> bool:
    return file.is_file() and (file.suffix == ".csv")

def filelist_tiff(datadir: Path):
    """
    Return list of files in directory that are tiffs
    """
    return [str(f) for f in datadir.iterdir() if is_tiff(f)]

def filelist_csv(datadir: Path):
    """
    Return list of files in directory that are csv
    """
    return [str(f) for f in datadir.iterdir() if is_csv(f)]

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

def load_path_into_dict(datadir, keywordregex=[r"CELL\d{3}"], keyword="", filetype = None):
    """ """
    if filetype == "tiff":
        filelist = filelist_tiff(datadir)  # all files in directory
    elif filetype == "csv":
        filelist = filelist_csv(datadir)  # all files in directory
    else: 
        filelist = [str(f) for f in datadir.iterdir()]
        
    path_dict = {
        extract_keyword(f, keywordregex): f for f in filelist if keyword in str(f)
    }

    return path_dict

def load_path_into_df(datadir, keywordregex=[r"CELL\d{3}"], keyword=None, filetype = None):
    """ """
    if filetype == "tiff":
        filelist = filelist_tiff(datadir)  # all files in directory
    elif filetype == "csv":
        filelist = filelist_csv(datadir)  # all files in directory
    else: 
        filelist = [str(f) for f in datadir.iterdir()]
    
    df_rows = []
    for file in filelist:
        filepath = Path(file)

        if keyword and keyword not in str(filepath):
            continue
        
        if not filepath.is_file():
            continue
        
        filename_split =  filepath.stem.split("_")
        if "halfbac" in file:
            condition = "infected"
        elif "nobac" in file:
            condition = "uninfected"
        else:
            condition = filename_split[1]
            
        df_row = {
            "Series" : extract_keyword(file, keywordregex),
            "Condition": condition,
            "Line": filename_split[0],
            "Timepoint": filename_split[2][1:],
            "Filepath": str(file),
            }        
            
        df_rows.append(df_row)
        
    path_df = pd.DataFrame(df_rows)
    
    return path_df

def projections_from_zstack(stack, method='all', smooth_sigma=1.0, eps=1e-8):
    """
    Compute projection images from a z-stack along the last axis (z).
    
    Parameters
    ----------
    stack : ndarray
        H x W x Z image stack
    method : str
        'std' | 'iqr' | 'cv' | 'mad' | 'all'
    smooth_sigma : float
        Gaussian smoothing sigma (pixels). 0 -> no smoothing.
    eps : float
        small value to avoid division by zero for CV.
    
    Returns
    -------
    projections : dict
        keys in {'std','iqr','cv','mad'} mapping to 2D arrays (H x W)
    """
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("stack must be a 3D array (H x W x Z)")
    H, W, Z = stack.shape
    methods = ['std','iqr','cv','mad'] if method == 'all' else [method.lower()]
    projections = {}
    
    if 'std' in methods:
        # population std: sqrt(mean((x - mean)^2))
        mean = np.mean(stack, axis=2)
        var = np.mean((stack - mean[..., None])**2, axis=2)
        stdimg = np.sqrt(var)
        projections['std'] = stdimg
    
    if 'iqr' in methods:
        # IQR = 75th - 25th percentile per pixel along z
        p75 = np.percentile(stack, 75, axis=2)
        p25 = np.percentile(stack, 25, axis=2)
        iqrimg = p75 - p25
        projections['iqr'] = iqrimg
    
    if 'cv' in methods:
        mean = np.mean(stack, axis=2)
        var = np.mean((stack - mean[..., None])**2, axis=2)
        stdimg = np.sqrt(var)
        cvimg = stdimg / (mean + eps)
        projections['cv'] = cvimg
    
    if 'mad' in methods:
        med = np.median(stack, axis=2)
        mad = np.median(np.abs(stack - med[..., None]), axis=2)
        projections['mad'] = mad
    
    # optional smoothing
    if smooth_sigma and smooth_sigma > 0:
        for k in list(projections.keys()):
            projections[k] = ndimage.gaussian_filter(projections[k], sigma=smooth_sigma, mode='reflect')
    
    return projections

def read_tiff_stack(path):
    """Read a multipage TIFF into a H x W x Z numpy array."""
    arr = tiff.imread(path)
    # tifffile often returns Z x H x W if saved as pages, so try to normalize
    if arr.ndim == 3:
        # If shape is (Z, H, W) we want (H, W, Z)
        if arr.shape[0] == arr.shape[1] or arr.shape[0] == arr.shape[2]:
            # ambiguous; we'll assume arr is (Z, H, W) -> transpose
            # but if the user provides a stack shaped (H, W, Z) already, we keep it
            pass
        # Heuristic: if first dimension equals Z (and likely smaller than others), transpose
        # Safer approach: if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2], it's likely Z
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            arr = np.transpose(arr, (1,2,0))
    return arr

def save_projections(projections, out_dir='/mnt/data', basename='projections'):
    """Save each projection as a single-page TIFF in out_dir. Returns paths dict."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for k,v in projections.items():
        # normalize to float32 for saving
        fn = f"{basename}_{k}.tif"
        path = os.path.join(out_dir, fn)
        # scale to full 0..1 then to float32 for decent viewing
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if vmax > vmin:
            norm = (v - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(v)
        tiff.imwrite(path, (norm * 65535).astype(np.uint16))
        paths[k] = path
    return paths

from scipy.ndimage import binary_erosion, binary_fill_holes

def separate_objects(binary_img,  # kernel used for erosion/dilation
    structure = 3, area_threshold = (10,10000)):
    
    binary_object = (binary_img > 0).astype(np.uint8)
    area_min, area_max = area_threshold
    
    if separate_connecting_iter > 0:
        eroded = binary_erosion(binary_object, structure=np.ones((structure,structure)))
    else:
        eroded = binary_object.copy()
    
    labeled_seeds = label(eroded)
    keep_seed_labels = []
    for region in regionprops(labeled_seeds):
        if area_min < region.area < area_max:
            keep_seed_labels.append(region.label)
    
    
    # distance from background
    markers = labeled_seeds.copy()
    distance = distance_transform_edt(binary_object == 0)
    watershed_labels = watershed(
        -distance, markers=markers, mask=binary_object.astype(bool)
    )
    kept_partition = np.isin(watershed_labels, keep_seed_labels).astype(np.uint8)

    total_count = label(kept_partition).max()

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
    
    return label(final_recovered), int(total_count)

def napari_view_3D(img):
    viewer = napari.Viewer()
    viewer.dims.ndisplay = 3 
    viewer.add_image(img,
                     rendering="average",
                     contrast_limits=(0, 1), 
                     gamma = 0.4,
                     )
    napari.run()
    
def image_overlap(img_1, img_2, show_intermediate=True):
    img_1 = img_1 > 0
    img_2 = img_2 > 0

    overlap = np.logical_and(img_1, img_2)

    return overlap.sum()  # overlap area


def merge_small_rois(segmentation, size_thresholds = (0,1000), cluster_dist = 10):
    
    min_size, max_size = size_thresholds
    
    if label(segmentation).max() > 1 and segmentation.max() > 1: # check if segmentation is already labelled
        labelled = segmentation
    else:
        print("labelling mask")
        labelled = label(segmentation)
        
    props = regionprops(labelled)
    
    piece_ids = []
    piece_centroids = {}
    piece_areas = {}
    for region in props:
        lab = int(region.label)
        area = int(region.area)
        piece_areas[lab] = area
        if min_size < area < max_size:
            piece_ids.append(lab)
            piece_centroids[lab] = np.array(region.centroid, dtype=float)
    
    if not piece_ids: # no regions to merge
        mapping = {int(l): int(l) for l in np.unique(labelled)}
        return labelled, mapping, {}
    
    # check if piece centroids are within cluster distance to merge rois
    label_list = list(piece_ids)
    pts = np.vstack([piece_centroids[l] for l in label_list])
    kdt = cKDTree(pts)
    
    adjacency = defaultdict(list)
    neighbors = kdt.query_ball_tree(kdt, r=cluster_dist)
    
    adjacency = defaultdict(list)
    for i, nbrs in enumerate(neighbors):
        li = label_list[i]
        for j in nbrs:
            lj = label_list[j]
            if li == lj:
                continue
            adjacency[li].append(lj)
            adjacency[lj].append(li)
    
    visited = set()
    clusters = {}
     
    for lab in label_list:
        if lab in visited:
            continue
        queue = deque([lab])
        visited.add(lab)
        cluster = []
     
        while queue:
            current = queue.popleft()
            cluster.append(current)
            for nbr in adjacency[current]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
     
        clusters[lab] = cluster
    
    max_label = int(labelled.max())
    next_label = max_label + 1
    mapping = {}
    
    for root, members in clusters.items():
       if len(members) > 1:
           new_label = next_label
           next_label += 1
           for m in members:
               mapping[int(m)] = int(new_label)
       else:
           # singleton: keep original label
           single = members[0]
           mapping[int(single)] = int(single)

    # any label not in mapping maps to itself (including background)
    for lab in np.unique(labelled):
        lab = int(lab)
        if lab == 0: # background label 0
            mapping[lab] = 0
        elif lab not in mapping:
            mapping[lab] = lab
        
        # apply mapping to create merged segmentation
        merged = labelled.copy()
        for orig_lab, tgt_lab in mapping.items():
            if orig_lab == 0:
                continue
            if orig_lab == tgt_lab:
                continue
        
            merged[labelled == orig_lab] = tgt_lab

    return merged, mapping, clusters

from skimage.morphology import disk, ball, binary_dilation
from skimage.segmentation import relabel_sequential

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank.get(ra,0) < self.rank.get(rb,0):
            self.parent[ra] = rb
        elif self.rank.get(rb,0) < self.rank.get(ra,0):
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = self.rank.get(ra,0) + 1

def merge_small_rois_with_neighbors(
        labelled,
        size_thresholds=(0, 3000),
        dilate_radius=3,
        use_3d=None,
        force_merge=False,
        compact_output=False,
        resolve='min'):
    """
    Merge small ROIs using dilation-overlap with conditional glue behavior:
      - A small ROI s (min_size < area < max_size) is eligible to merge.
      - If s has exactly one neighbor in its dilated region -> merge s into that neighbor (neighbor may be large).
      - If s has multiple neighbors -> s only glues them together if at least TWO of those neighbors are small.
        In that case union {s} + those small neighbors.
      - force_merge: if s has no neighbors, merge into nearest non-background by centroid (optional).
    Returns: merged (uint16), final_mapping (orig->final), merge_events (list of sets), components (repr->members)
    """
    labelled = np.asarray(labelled)
    if use_3d is None:
        use_3d = (labelled.ndim == 3)

    min_size, max_size = size_thresholds
    props = regionprops(labelled)
    if not props:
        return labelled.astype(np.uint16), {int(l):int(l) for l in np.unique(labelled)}, [], {}

    # precompute areas and arrays
    area_by_label = {int(p.label): int(p.area) for p in props}
    all_labels = np.array([int(p.label) for p in props])
    all_centroids = np.vstack([np.array(p.centroid, dtype=float) for p in props])
    label_to_index = {lab: idx for idx, lab in enumerate(all_labels)}

    struct = ball(dilate_radius) if use_3d else disk(dilate_radius)

    merge_events = []  # list of sets that will be unioned

    for p in props:
        lab = int(p.label)
        area = int(p.area)
        # only small ROIs initiate merges
        if not (min_size < area < max_size):
            continue

        bbox = p.bbox
        if labelled.ndim == 3:
            minr, minc, mind, maxr, maxc, maxd = bbox
            slices = (slice(minr, maxr), slice(minc, maxc), slice(mind, maxd))
        else:
            minr, minc, maxr, maxc = bbox
            slices = (slice(minr, maxr), slice(minc, maxc))

        sub_mask = (labelled[slices] == lab)
        # dilate using modern keyword; fallback if necessary
        try:
            dilated = binary_dilation(sub_mask, footprint=struct)
        except TypeError:
            dilated = binary_dilation(sub_mask, selem=struct)

        labels_in_region = np.unique(labelled[slices][dilated])
        # filter out background and itself
        neighbor_labels = [int(x) for x in labels_in_region if (x != 0 and int(x) != lab)]

        if len(neighbor_labels) == 0:
            # no neighbors in dilated area
            if force_merge:
                # find nearest non-background by centroid
                s_idx = label_to_index[lab]
                if all_centroids.shape[0] > 1:
                    kdt = cKDTree(all_centroids)
                    k = min(5, all_centroids.shape[0])
                    dists, idxs = kdt.query(all_centroids[s_idx], k=k)
                    if np.isscalar(idxs):
                        idxs = np.array([idxs]); dists = np.array([dists])
                    chosen = None
                    for cand_idx in idxs:
                        if int(cand_idx) == int(s_idx):
                            continue
                        cand_lab = int(all_labels[int(cand_idx)])
                        if cand_lab == 0:
                            continue
                        chosen = cand_lab
                        break
                    if chosen is not None:
                        merge_events.append({lab, int(chosen)})
            # else leave alone
            continue

        if len(neighbor_labels) == 1:
            # exactly one neighbor -> merge into it regardless of its size
            merge_events.append({lab, neighbor_labels[0]})
            continue

        # multiple neighbors: glue only if at least TWO of those neighbors are small
        small_neighbors = [n for n in neighbor_labels if (min_size < area_by_label.get(n,0) < max_size)]
        if len(small_neighbors) >= 2:
            union_set = set([lab]) | set(small_neighbors)
            merge_events.append(union_set)
        else:
            # Otherwise (multiple neighbors but <2 small neighbors) do NOT glue.
            # Option: if you want to instead merge into the single nearest neighbor in this case,
            # uncomment the block below to merge into the nearest neighbor.
            #
            # # nearest neighbor fallback among neighbor_labels:
            # s_idx = label_to_index[lab]
            # pts = all_centroids[[label_to_index[n] for n in neighbor_labels], :]
            # dists = np.linalg.norm(pts - all_centroids[s_idx], axis=1)
            # best = neighbor_labels[int(np.argmin(dists))]
            # merge_events.append({lab, best})
            #
            # For now we do nothing (leave s unchanged) when multiple neighbors exist but <2 smalls.
            continue

    # If nothing recorded, return quickly (optionally compact)
    if not merge_events:
        merged = labelled.copy()
        if compact_output:
            merged, _, _ = relabel_sequential(merged)
        if merged.max() > np.iinfo(np.uint16).max:
            raise ValueError("Resulting labels exceed uint16 max. Set compact_output=True or use larger dtype.")
        return merged.astype(np.uint16), {int(l):int(l) for l in np.unique(labelled)}, [], {}

    # Build union-find, register labels
    uf = UnionFind()
    for lab in np.unique(labelled):
        uf.find(int(lab))

    # Union each merge event's members together
    for s in merge_events:
        s_list = sorted(int(x) for x in s)
        base = s_list[0]
        for other in s_list[1:]:
            uf.union(base, other)

    # Build components (exclude background 0)
    comps = defaultdict(list)
    for lab in np.unique(labelled):
        lab = int(lab)
        if lab == 0:
            continue
        root = uf.find(lab)
        comps[root].append(lab)

    # normalize components to deterministic representatives (min member)
    normalized_components = {}
    for root, members in comps.items():
        members_sorted = sorted(set(members))
        canonical = members_sorted[0]
        normalized_components[canonical] = members_sorted

    # Build final mapping depending on resolve strategy
    final_mapping = {}
    current_max = int(labelled.max())
    next_new = current_max + 1

    if resolve == 'min':
        for canonical, members in normalized_components.items():
            tgt = int(min(members))
            for m in members:
                final_mapping[int(m)] = tgt
    elif resolve == 'new':
        for canonical, members in normalized_components.items():
            tgt = next_new
            next_new += 1
            for m in members:
                final_mapping[int(m)] = int(tgt)
    else:
        raise ValueError("resolve must be 'min' or 'new'")

    # identity mapping for labels not in final_mapping and background
    for lab in np.unique(labelled):
        lab = int(lab)
        if lab == 0:
            final_mapping[0] = 0
        else:
            final_mapping.setdefault(lab, lab)

    # apply mapping safely
    merged = labelled.copy()
    max_key = int(max(final_mapping.keys()))
    if max_key <= 200000:
        lut = np.arange(max_key + 1, dtype=np.int64)
        for orig, tgt in final_mapping.items():
            if orig <= max_key:
                lut[orig] = tgt
        merged = lut[merged]
    else:
        for orig, tgt in final_mapping.items():
            if orig == tgt or orig == 0:
                continue
            merged[labelled == orig] = tgt

    if compact_output:
        merged, _, _ = relabel_sequential(merged)

    if merged.max() > np.iinfo(np.uint16).max:
        raise ValueError("Resulting labels exceed uint16 max. Use compact_output=True or larger dtype.")
    merged = merged.astype(np.uint16)

    components = {int(rep): sorted(members) for rep, members in normalized_components.items()}
    return merged, final_mapping, merge_events, components


def merge_overlapping_z(labels, z_spacing = 1, z_threshold = 2, xy_overlap = 0.5, preserve_original_ids = True):
    
    '''
    labels: 3D numpy array of ints, shape (Z, Y, X).
    z_spacing: float; physical spacing between z slices. Use 1.0 if using slice units.
    z_threshold: float; max allowed physical gap between label z-extents to merge.
    preserve_original_ids: if True, groups with a single member keep their original id
    xy_overlap: percent overlap of the smaller area to be considered overlapping
    '''
    
    Z, Y, X = labels.shape
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    if len(unique_labels) == 0:
        return labels.copy(), {}
    
    # 1) compute z-min and z-max per label
    z_min = {}
    z_max = {}
    for z in range(Z):
        slice_ids = np.unique(labels[z])
        slice_ids = slice_ids[slice_ids != 0]
        for lab in slice_ids:
            if lab not in z_min:
                z_min[lab] = z
            z_max[lab] = z
    
    # 2) compute 2D projection (XY) mask and area for each label
    projections = {}
    proj_area = {}
    # --- recommended robust method: direct per-label projection using np.any ---
    for lab in unique_labels:
        lab = int(lab)
        proj_mask = np.any(labels == lab, axis=0)  # shape (Y, X), boolean
        projections[lab] = proj_mask
        proj_area[lab] = int(proj_mask.sum())
    
    # convert z_threshold (physical) to slice-units for gap calc
    threshold_slices = z_threshold / float(z_spacing)
    
    def compute_gap_slices(amin, amax, bmin, bmax):
        if amax < bmin:
            return bmin - amax - 1
        elif bmax < amin:
            return amin - bmax - 1
        else:
            return -1  # intervals overlap
    
    # 3) build adjacency where both tests (z and xy overlap) pass
    adjacency = defaultdict(set)
    labs_list = [int(x) for x in unique_labels]
    for a, b in combinations(labs_list, 2):
        area_a = proj_area.get(a, 0)
        area_b = proj_area.get(b, 0)
        if area_a == 0 or area_b == 0:
            continue
    
        overlap_count = int(np.count_nonzero(projections[a] & projections[b]))
        if overlap_count == 0:
            continue
    
        smaller_area = min(area_a, area_b)
        required = int(np.ceil(xy_overlap * smaller_area))
    
        if overlap_count < required:
            continue
    
        amin, amax = z_min[a], z_max[a]
        bmin, bmax = z_min[b], z_max[b]
        gap = compute_gap_slices(amin, amax, bmin, bmax)
        if gap < 0:
            adjacency[a].add(b)
            adjacency[b].add(a)
        else:
            phys_gap = gap * z_spacing
            if phys_gap <= z_threshold + 1e-12:
                adjacency[a].add(b)
                adjacency[b].add(a)
    
    # ensure isolated labels appear in adjacency
    for lab in labs_list:
        adjacency.setdefault(lab, set())
    
    # 4) find connected components using iterative DFS
    visited = set()
    components = []
    for lab in adjacency.keys():
        if lab in visited:
            continue
        stack = [lab]
        comp = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in adjacency[node]:
                if nb not in visited:
                    stack.append(nb)
        components.append(sorted(comp))
    
    # 5) assign merged ids
    merged_id_map = {}
    next_new_id = int(max(labs_list)) + 1
    for comp in components:
        if len(comp) == 1 and preserve_original_ids:
            merged_id_map[comp[0]] = comp[0]
        else:
            new_id = next_new_id
            next_new_id += 1
            for old in comp:
                merged_id_map[old] = new_id
    
    # 6) remap volume
    max_lab_in_vol = int(labels.max())
    map_arr = np.zeros(max_lab_in_vol + 1, dtype=np.int32)
    for old in range(1, max_lab_in_vol + 1):
        if old in merged_id_map:
            map_arr[old] = merged_id_map[old]
        else:
            map_arr[old] = 0
    
    merged = map_arr[labels]
    mapping = {int(old): int(new) for old, new in merged_id_map.items()}
    
    return merged, mapping

def cell_stats_v1(BF_segmentation, FL_segmentation, area_threshold = 300, cell_prefix = "Series000", small_roi_merge = False, merge_overlapping_rois = False):
    
    if label(BF_segmentation).max() > 1 and BF_segmentation.max() > 1: # check if BF_segmentation is already labelled
        labelled = BF_segmentation
    else:
        labelled = label(BF_segmentation)
    
    if small_roi_merge:      
        labelled_img, _, _ = merge_small_rois(labelled)
    else:
        labelled_img = labelled
        
    if merge_overlapping_rois:
        labelled_img, _ = merge_overlapping_z(labelled_img)
    
    # number_of_cells = np.count_nonzero(np.unique(labelled_img))
    
    cell_stats = pd.DataFrame()
    kept_cells = np.zeros_like(labelled_img)
    deleted_cells = np.zeros_like(labelled_img)
    
    cellid_list = np.unique(labelled_img)
    cellid_list = cellid_list[cellid_list != 0].tolist() # get list of all cellids exclusing background
    
    for cell_num in tqdm(cellid_list):
        cell = (labelled_img == cell_num).astype(np.uint8)
        filled = (ndi.binary_fill_holes(cell)>0).astype(np.uint8)
        
        overlap_area = image_overlap(filled, FL_segmentation)
        
        props_table = regionprops_table(
            filled, 
            spacing=(0.49, 0.481, 0.481),
            properties = ["area", "centroid"] #, "axis_major_length", "axis_minor_length"]
            ) # pixel size in um
        
        props_df = pd.DataFrame(props_table)
        
        # make additional measurements
        calcs = pd.DataFrame({
            "label": [cell_prefix + "_" + str(cell_num)],
            #"aspect_ratio": [props_df["axis_major_length"][0] / props_df["axis_minor_length"][0]] # does not work with 3D
            })
        
        props_df = pd.concat([calcs, props_df], axis = 1) # add new columns
        
        # check if PXN signal overlaps
        if overlap_area > (0.8*props_df["area"][0]):
            props_df["PXN+"] = 1
        else:
            props_df["PXN+"] = 0
            
        if props_df["area"][0] > area_threshold:
            cell_stats = pd.concat([cell_stats, props_df])
            kept_cells = np.where(filled > 0, cell_num, kept_cells)
        
        else:
            deleted_cells = np.where(filled > 0, cell_num, deleted_cells)
            
    return cell_stats, labelled_img, deleted_cells

def cell_FL_count(BF_segmentation, FL_segmentation, overlap_fraction = 0.8, plot = False):
    
    if label(BF_segmentation).max() > 1 and BF_segmentation.max() > 1: # check if BF_segmentation is already labelled
        labelled = BF_segmentation
    else:
        labelled = label(BF_segmentation)
    
    # number_of_cells = np.count_nonzero(np.unique(labelled_img))
        
    cellid_list = np.unique(labelled)
    cellid_list = cellid_list[cellid_list != 0].tolist() # get list of all cellids exclusing background
    
    FL_count = 0
    noFL_count = 0
    
    FL_mask = np.zeros_like(labelled)
    noFL_mask = np.zeros_like(labelled)
    
    for cell_num in cellid_list:
        cell = (labelled == cell_num).astype(np.uint8)        
        overlap_area = image_overlap(cell, FL_segmentation)
        
        cell_area = np.sum(cell)
        # check if PXN signal overlaps
        if overlap_area > (overlap_fraction*cell_area):
            FL_count += 1
            FL_mask[cell == 1] = 1 
        else:
            noFL_count += 1
            noFL_mask[cell == 1] = 1
 
    return FL_count, noFL_count, FL_mask, noFL_mask, labelled

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

def isotropic_resampling_z(binary_image, spacing = (1.0,1.0,1.0)): 
    # spacing z,y,x
    z , y , x = spacing
    # compute zoom: (z, y, x)
    zoom_z = z / y  # ~5.09355
    zoom = (zoom_z, 1.0, 1.0)
    
    # resize (order=1 linear). For binary masks, we'll threshold afterwards.
    upsampled = ndi.zoom(binary_image.astype(np.float32), zoom=zoom, order=1)
    
    # optional smoothing (gaussian)
    # sigma in voxels (choose small value like [sigma_z, sigma_y, sigma_x])
    sigma = (1.0, 0.5, 0.5)   # tweak as needed
    smoothed = ndi.gaussian_filter(upsampled, sigma=sigma)
    # re-binarize
    mask_resampled = (smoothed > 0.5).astype(np.uint8)
    
    return mask_resampled

def remove_outliers_iqr(df, group_col, value_col):
    df = df.copy()

    Q1 = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(0.25))
    Q3 = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(0.75))
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[value_col] >= lower) & (df[value_col] <= upper)]

def overlapping_to_layers(rois, img_shape):
    """
    Groups overlapping ImageJ ROIs into separate non-overlapping layers.

    Parameters:
        rois (list of ImagejRoi): List of ROIs from roifile.
        img_shape (tuple): Shape of the binary image (height, width).

    Returns:
        list of np.ndarray: Each element is a binary image layer (boolean).
    """
    assigned = np.zeros(len(rois), dtype=bool)
    layers = []

    while not np.all(assigned):
        layer_mask = np.zeros(img_shape, dtype=np.uint8)
        this_layer = []

        for i, roi in enumerate(rois):
            if assigned[i]:
                continue

            coords = roi.coordinates()  # Returns (N, 2) array of (x, y)
            if coords is None or len(coords) < 3:
                continue  # Skip ROIs that can't form a valid polygon

            x = np.clip(np.round(coords[:, 0]).astype(int), 0, img_shape[1] - 1)
            y = np.clip(np.round(coords[:, 1]).astype(int), 0, img_shape[0] - 1)

            rr, cc = draw.polygon(y, x, img_shape)
            test_mask = np.zeros(img_shape, dtype=np.uint8)
            test_mask[rr, cc] = 1

            if np.any(layer_mask & test_mask):
                continue  # Overlaps, skip for this layer

            layer_mask |= test_mask
            this_layer.append(i)
            assigned[i] = True

        layers.append(layer_mask)

    return layers

def add_scale_bar(
    ax,
    px_size,
    length_um,
    color="white",
    lw=3,
    pad=100,
    location="lower right",
    fontsize=10,
    text = False
):
    """
    Draw a horizontal scale bar on an axes.

    Parameters
    ----------
    ax : matplotlib Axes
    px_size : float
        Physical pixel size (e.g. µm per pixel).
    length_um : float
        Length of scale bar in micrometers.
    color : str or tuple
    lw : int
        Line width.
    pad : int
        Padding in pixels from edge.
    location : str
        "lower right", "lower left", "upper right", "upper left"
    """
    # Convert micrometers → pixels
    length_px = length_um / px_size

    # Get image dimensions
    h, w = ax.images[0].get_array().shape[:2]

    # Compute bar anchor point (x0, y0)
    if location == "lower right":
        x0 = w - length_px - pad
        y0 = h - pad
        text_va = "top"
        text_y = y0 + 20  # below bar

    elif location == "lower left":
        x0 = pad
        y0 = h - pad
        text_va = "top"
        text_y = y0 + 20

    elif location == "upper right":
        x0 = w - length_px - pad
        y0 = pad
        text_va = "bottom"
        text_y = y0 - 20  # below bar (bar is at top)

    elif location == "upper left":
        x0 = pad
        y0 = pad
        text_va = "bottom"
        text_y = y0 - 20

    else:
        raise ValueError(
            "location must be 'lower right', 'lower left', 'upper right', or 'upper left'"
        )

    # Draw scale bar
    ax.plot(
        [x0, x0 + length_px],
        [y0, y0],
        color=color,
        linewidth=lw,
        zorder=10,
    )

    # Draw label underneath bar
    if text:
        ax.text(
            x0 + length_px / 2,
            text_y,
            f"{length_um:g} µm",
            ha="center",
            va=text_va,
            color=color,
            fontsize=fontsize,
        )

def crop_image(mask, image, padding=0, show_img=True):
    """Crops the image to the bounding box of the non-zero region in the mask."""
    coords = np.column_stack(np.where(mask > 0))
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    if padding > 0:
        y0 = max(0, y0 - padding)
        x0 = max(0, x0 - padding)
        y1 = min(mask.shape[0], y1 + padding)
        x1 = min(mask.shape[1], x1 + padding)
    
    cropped_image = image[y0:y1, x0:x1]

    if show_img:
        fig, ax = plt.subplots(figsize=(8, 8))

        plt.imshow(cropped_image, cmap="gray")

        add_scale_bar(
            ax=ax,
            px_size=constants.PX_SIZE_LR,
            length_um=1,
            color="white",
            lw=5,
            pad=10,
            location="lower right",
            fontsize=5,
        )

        plt.tight_layout()
        plt.axis("off")
        plt.show()


    return cropped_image, [y0,y1, x0,x1]

def normalize_to_unit(arr):
    """Normalize array to 0..1 safely (works with constant arrays)."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if np.isclose(mx, mn):
        return np.clip(arr - mn, 0, 1)  # will be all zeros
    return (arr - mn) / (mx - mn)

def apply_colormap(img, cmap_name='viridis'):
    """Map a 2D image (0..1) to an RGB (H,W,3) array using matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    mapped = cmap(img)[:, :, :3]  # drop alpha from RGBA -> RGB
    return mapped

def overlay_grayscale_images(img1, img2, cmap1='viridis', cmap2='plasma', alpha=0.5):
    """
    Convert two 2D grayscale images to RGB and overlay them.
    - img1, img2: 2D numpy arrays (same shape)
    - cmap1, cmap2: colormap names for each image
    - alpha: blending weight of img2 over img1 (0..1)
    Returns an (H, W, 3) float ndarray in [0,1].
    """
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape")
    # Normalize each to 0..1
    n1 = normalize_to_unit(img1)
    n2 = normalize_to_unit(img2)
    # Map to RGB with chosen colormaps
    rgb1 = apply_colormap(n1, cmap1)
    rgb2 = apply_colormap(n2, cmap2)
    # Simple linear blend (you can change blending model if desired)
    out = (1 - alpha) * rgb1 + alpha * rgb2
    out = np.clip(out, 0, 1)
    return out

def plot_overlays_masks_cell(
    base_img=None,
    outlines=None,
    masks=None,
    points=None,
    point_labels=None,
    return_image=False,
    legend=True,
    scale_bar=True,
    crop=False,
    title=None,
    show_base_alone=False,
    rotation = 0,
    fig_size = (6,6),
):
    """
    Plots overlays of outlines, masks, and points on a base image.

    :param base_img: np.ndarray or None
        Base image to plot overlays on. If None, a blank image is created.
    :param outlines: dict or None
        Dictionary of outlines {organelle: binary mask}
    :param masks: dict or None
        Dictionary of masks {organelle: binary mask}
    :param points: dict or None
        Dictionary of points {organelle: list of (x, y) coordinates}
    :param point_labels: dict or None
        Dictionary mapping organelle names to point colors {organelle: color}
    :param return_image: bool
        If True, returns the matplotlib figure object.
    :param legend: bool
        If True, displays a legend for the overlays.
    :param scale_bar: bool
        If True, adds a scale bar to the image.
    :return: fig, ax, (optional) final_img
    """
    # Rotate base image and masks if needed
    if rotation != 0:
        if base_img is not None:
            base_img = np.rot90(base_img, k=rotation // 90)

        if outlines is not None:
            for organelle in outlines:
                outlines[organelle] = np.rot90(outlines[organelle], k=rotation // 90)

        if masks is not None:
            for organelle in masks:
                masks[organelle] = np.rot90(masks[organelle], k=rotation // 90)

    h, w = base_img.shape[:2]
    dpi = 50

    #fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed axes
    ax.set_axis_off()

    if base_img is None:  # create blank image if no base image provided
        found = False
        if outlines:
            for outline in outlines.values():
                seg = np.asarray(outline) > 0
                if seg:
                    img_shape = seg.shape
                    found = True
                    break
        
        if (not found) and masks:
            for mask in masks.values():
                seg = np.asarray(mask) > 0
                if seg:
                    img_shape = seg.shape
                    found = True
                    break
            
        if found == False:
            print("No input images")
            return 
        
        base_img = np.ones(img_shape, dtype=float)
        
    else:
        img_shape = base_img.shape
    
    

    # show xray image
    #ax.imshow(base_img, cmap="gray")#, vmin=0, vmax=64000)
    ax.imshow(base_img)
    
    plotted_organelles = set()  # to track which organelles have been plotted in legend

    if outlines is not None:
        for item, outline in outlines.items():
            
            segmentation = outline > 0
            contours = measure.find_contours(segmentation, 0.5)
            color = constants.CMAP.get(item, "red")

            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=2)

            plotted_organelles.add(item) # add to legend only if not already added

    if masks is not None:
        for item, mask in masks.items():
            segmentation = mask > 0

            # overlay mask
            h, w = segmentation.shape
            overlay = np.zeros((h, w, 4), dtype=float)
            
            hex_color = constants.CMAP.get(item, "#008000") # default to green
            color = mcolors.to_rgb(hex_color)
        
            # set color (green) and alpha only where mask is True
            overlay[segmentation, :3] = color  # R,G,B in 0..1
            overlay[segmentation, 3] = 0.2  # alpha for masked pixels

            # show overlay; alpha channel is in the 4th channel, so set alpha=1 in imshow
            ax.imshow(overlay, interpolation="none", zorder=2)

    if points is not None:
        for item, point in points.items():
            color = point_labels[item]

            xs = [p[0] for p in point]
            ys = [p[1] for p in point]

            ax.scatter(
                xs,
                ys,
                s=5,
                facecolor=color,
                edgecolor=color,
                linewidths=0.5,
                zorder=50,
            )

    # add scale bar
    if scale_bar:
        add_scale_bar(
            ax=ax,
            px_size=constants.PX_SIZE_LR*0.001,
            length_um=10,
            color="black",
            lw=10,
            pad=10,
            location="lower right",
            fontsize=0,
        )

    # add legend
    if legend:
        legend_handles = []
        if masks is not None:
            handles_mask = [
                Patch(
                    facecolor=constants.CMAP[item], label=item
                )
                for item in masks.keys()
            ]

            legend_handles += handles_mask

        if outlines is not None:
            handles_outline = [
                Line2D(
                    [0],
                    [0],
                    color=constants.CMAP.get(item, "red"),
                    lw=10,
                    label=item,
                )
                for item in plotted_organelles
            ]
            legend_handles += handles_outline

        if points is not None:
            handles_points = [
                Line2D(
                    [0],
                    [0],
                    marker="o",  # or 'x', '*', etc.
                    color="none",  # no connecting line
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markersize=5,
                    linewidth=0,
                    label=label,
                )
                for label, color in point_labels.items()
            ]
            legend_handles += handles_points

        ax.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=True,
            facecolor="white",
            edgecolor="gray",
            fontsize=60,
        )

    # show plot
    if title is not None:
        plt.title(title, fontsize=20)

    plt.axis("off")
    #plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    final_img = np.asarray(buf, dtype=np.uint8).copy()  # H x W x 4
    # print(final_img.shape) for debugging

    plt.show()

    if show_base_alone and base_img is not None:
        fig_base, ax_base = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax_base.imshow(base_img, cmap="gray", vmin=0, vmax=100)
        # plt.title(title, fontsize=100)
        plt.title("X-ray", fontsize=100)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if crop:  # crop out each object
        if outlines is not None:
            for item, outline in outlines.items():
                segmentation = np.asarray(outline) > 0
                _ = crop_image(segmentation, final_img, padding=25, show_img=True)

        if masks is not None:
            for item, mask in masks.items():
                segmentation = np.asarray(mask) > 0
                _ = crop_image(segmentation, final_img, padding=25, show_img=True)

    if return_image:
        return fig, ax, final_img

    return fig, ax

def to_uint8_linear(img, vmin=None, vmax=None):
    """Linear scale to 0..255 using provided vmin/vmax or img min/max."""
    if vmin is None: vmin = float(img.min())
    if vmax is None: vmax = float(img.max())
    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - vmin) * (255.0 / (vmax - vmin))
    return np.clip(scaled, 0, 255).astype(np.uint8)

def to_uint8_percentile(img, low_p=1, high_p=99):
    """Clip to percentiles then linear scale to 0..255."""
    lo, hi = np.percentile(img.ravel(), [low_p, high_p])
    return to_uint8_linear(img, vmin=lo, vmax=hi)

def to_uint8_log(img, clip_min=None, clip_max=None):
    """Apply log1p then linear scale to 0..255. Useful for compressing dynamic range."""
    if clip_min is None: clip_min = float(img.min())
    if clip_max is None: clip_max = float(img.max())
    img_clipped = np.clip(img, clip_min, clip_max)
    logged = np.log1p(img_clipped - clip_min)  # shift so min->0
    return to_uint8_linear(logged, vmin=0.0, vmax=float(logged.max()))

def adaptive_threshold(
    image,
    method="gaussian",
    block_size=11,
    C=2
):
    """
    Apply adaptive thresholding to a grayscale image.

    Parameters:
        image (numpy.ndarray): Grayscale image.
        method (str): "mean" or "gaussian".
        block_size (int): Size of local neighborhood (must be odd and > 1).
        C (int): Constant subtracted from the computed mean.

    Returns:
        numpy.ndarray: Binary thresholded image.
    """

    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    if block_size % 2 == 0 or block_size <= 1:
        raise ValueError("block_size must be odd and > 1.")

    if method.lower() == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method.lower() == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError("method must be 'mean' or 'gaussian'.")

    thresh = cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )

    return thresh

def threshold_otsu(img8):
    # img8 must be uint8 2D
    if img8.dtype != np.uint8:
        raise ValueError("img8 must be uint8")
    _, binary_255 = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_01 = (binary_255 == 255).astype(np.uint8)
    
    return binary_255, binary_01

def create_table(df, body_fontsize=5, header_fontsize=5, fig_size=(5, 2.5)):

    '''
    df is a pandas DataFrame with columns = table columns and rows = table rows. The function creates a styled table figure with journal-style formatting and booktabs-like horizontal rules.
    '''

    # -----------------------
    # 2. Create Figure
    # -----------------------
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    # -----------------------
    # 3. Style the Table
    # -----------------------

    table.auto_set_font_size(False)
    #table.set_fontsize(9)

    # Remove all cell borders (journal style)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = 0.02

        if row == 0:
            # Header row
            cell.set_text_props(weight='bold', fontsize=header_fontsize)
        else:
            # Body rows
            cell.set_text_props(fontsize=body_fontsize)

    # Make header bold
    for col in range(len(df.columns)):
        table[(0, col)].set_text_props(weight='bold')


    # Left align first column (Group)
    #for row in range(1, len(df) + 1):
        #table[(row, 0)].get_text().set_ha('left')

    # -----------------------
    # 4. Add Horizontal Rules (Booktabs Style)
    # -----------------------
    plt.draw()

    # Get table bounding box
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transAxes.inverted()
    bbox = inv.transform_bbox(bbox)

    x0, x1 = bbox.x0, bbox.x1
    y_top = bbox.y1
    y_bottom = bbox.y0

    # Header bottom line
    header_height = table[(0, 0)].get_window_extent(fig.canvas.get_renderer())
    header_height = inv.transform_bbox(header_height)
    y_header_bottom = header_height.y0

    # Draw lines
    ax.plot([x0, x1], [y_top, y_top], linewidth=1, transform=ax.transAxes, color = "black")
    ax.plot([x0, x1], [y_header_bottom, y_header_bottom], linewidth=0.7, transform=ax.transAxes,  color = "black")
    ax.plot([x0, x1], [y_bottom, y_bottom], linewidth=1, transform=ax.transAxes,  color = "black")

    plt.tight_layout()

    return

def fill_holes(binary_object, kernel=None, gap_fill=0):
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

    return filled

import matplotlib.colors as mcolors
import numpy as np

def lighten_color(color, amount=0.45):
    """
    amount = 0 → original color
    amount = 1 → white
    0.4–0.5 works well for 4h
    """
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple(c + (white - c) * amount)

def calculate_pvals(df_clean, measure_plot, comparison_col, stat_test = "Welch", correction = True):
    alpha = 0.05
    lines = np.unique(df_clean[comparison_col])
    
    # ---- Pairwise Welch tests between lines ----
    pairs = []
    pvals = []
    
    if stat_test == "Welch":
        
        for aL, bL in itertools.combinations(lines, 2):
            a_vals = df_clean[df_clean[comparison_col].astype(str) == aL][measure_plot].astype(float)
            b_vals = df_clean[df_clean[comparison_col].astype(str) == bL][measure_plot].astype(float)
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            _, p = ttest_ind(a_vals, b_vals, equal_var=False)
            pairs.append((aL, bL))
            pvals.append(p)
            
    elif stat_test == "Anova-Tukey":
        
        groups = [
        df_clean[df_clean[comparison_col].astype(str) == g][measure_plot].astype(float)
        for g in lines
        ]
         
        # run ANOVA
        f_stat, p_anova = f_oneway(*groups)
         
        if p_anova < alpha:
            # prepare data
            values = df_clean[measure_plot].astype(float)
            groups = df_clean[comparison_col].astype(str)
             
            tuk = pairwise_tukeyhsd(values, groups, alpha=alpha)
             
            # extract pairs + p-values
            g1, g2 = tuk._multicomp.pairindices
            labels = tuk.groupsunique
             
            for i, j, p in zip(g1, g2, tuk.pvalues):
                pairs.append((labels[i], labels[j]))
                pvals.append(p)
        else:
            # ANOVA not significant
            return [], []
        
    if not pairs:
        raise ValueError("Not enough samples for pairwise comparisons.")
    
    if correction:
        # FDR correction
        _, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
    else:
        pvals_corr = pvals
    
    return pairs, pvals_corr