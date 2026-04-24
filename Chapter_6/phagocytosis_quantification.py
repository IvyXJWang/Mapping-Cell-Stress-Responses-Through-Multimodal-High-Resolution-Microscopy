from pathlib import Path
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import utils
from skimage.measure import label, regionprops, regionprops_table, perimeter
from scipy.optimize import linear_sum_assignment
import constants
import cv2
from skimage.morphology import remove_small_objects, convex_hull_image
import statistics
import math
#%% function definition 

def fill_holes_and_outline(binary_vol, pad_for_border_holes=False):
    """
    Parameters
    ----------
    binary_vol : (Z, Y, X) numpy bool array
        Input binary volume (True = foreground).
    pad_for_border_holes : bool
        If True, pad the volume by 1 voxel before filling so cavities
        that touch the original border are treated as internal and get filled.

    Returns
    -------
    filled : ndarray (bool)
        The hole-filled binary volume

    """
    vol = np.asarray(binary_vol, dtype=bool)

    if pad_for_border_holes:
        vol = np.pad(vol, pad_width=1, mode='constant', constant_values=False)

    # Fill holes (scipy.ndimage.binary_fill_holes handles N-D)
    filled = ndi.binary_fill_holes(vol)

    if pad_for_border_holes:
        # remove padding
        filled = filled[1:-1, 1:-1, 1:-1]

    return filled


def match_previous_frame(ref_labels, curr_binary,
                              min_overlap_fraction=0.1,
                              min_iou=0.05,):
    """
    Keep only the regions in curr_binary that best match reference labels (ref_labels).
    The output image is labeled with the reference labels for matched regions.
    
    Parameters
    ----------
    ref_labels : 2D int np.ndarray
        Label image at reference time (0 = background). Each label is a tracked object id.
    curr_binary : 2D bool/0-1 np.ndarray
        Binary image at later time. Regions will be labeled and compared to ref_labels.
    min_overlap_fraction : float
        Minimum fraction of the *reference* object's area that must overlap the candidate region
        to accept the match (O / area_ref >= min_overlap_fraction).
    min_iou : float
        Alternative threshold: minimum IoU required to accept a match.
    use_skimage_label : bool
        Prefer skimage.measure.label if available for connectivity; otherwise uses scipy.ndimage.label.
    
    Returns
    -------
    matched_label_image : 2D int np.ndarray
        Image the same shape as inputs. Contains only pixels from curr_binary that were matched,
        labeled with the corresponding reference label (0 = background / unmatched).
    mapping : dict
        Maps ref_label -> matched_curr_label (0 if no acceptable candidate found).
    """
    # Label current binary
    curr_labels = label(curr_binary)

    # Collect IDs (exclude background 0)
    ref_ids = np.unique(ref_labels)
    ref_ids = ref_ids[ref_ids != 0]
    curr_ids = np.unique(curr_labels)
    curr_ids = curr_ids[curr_ids != 0]
    
    # Prepare result image
    matched_label_image = np.zeros_like(ref_labels, dtype=int)
    mapping = {int(r): 0 for r in ref_ids}  # default: no match
    
    if len(ref_ids) == 0 or len(curr_ids) == 0:
        # nothing to match
        return matched_label_image, mapping
    
    # Precompute areas
    ref_areas = {int(r): int((ref_labels == r).sum()) for r in ref_ids}
    curr_areas = {int(c): int((curr_labels == c).sum()) for c in curr_ids}
    
    # Build overlap matrix: rows = ref_ids, cols = curr_ids
    overlap = np.zeros((len(ref_ids), len(curr_ids)), dtype=int)
    for j, cid in enumerate(curr_ids):
        mask = (curr_labels == cid)
        intersect_ids, counts = np.unique(ref_labels[mask], return_counts=True)
        for iid, cnt in zip(intersect_ids, counts):
            if iid == 0:
                continue
            i = int(np.where(ref_ids == iid)[0][0])
            overlap[i, j] = int(cnt)
    
    # If no overlaps at all -> no matches
    if overlap.sum() == 0:
        return matched_label_image, mapping
    
    # Hungarian: maximize overlap -> minimize -overlap
    n = max(len(ref_ids), len(curr_ids))
    cost = np.zeros((n, n), dtype=int)
    cost[:overlap.shape[0], :overlap.shape[1]] = -overlap
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Evaluate assignments and apply thresholds
    assigned_ref = set()
    assigned_curr = set()
    for r, c in zip(row_ind, col_ind):
        if r >= overlap.shape[0] or c >= overlap.shape[1]:
            continue  # padded assignment
        inter = int(overlap[r, c])
        if inter == 0:
            continue
        ref_id = int(ref_ids[r])
        curr_id = int(curr_ids[c])
        area_ref = ref_areas[ref_id]
        area_curr = curr_areas[curr_id]
        union = area_ref + area_curr - inter
        iou = inter / union if union > 0 else 0.0
        frac_of_ref = inter / area_ref if area_ref > 0 else 0.0
        
        # Accept if meets either threshold
        if (frac_of_ref >= min_overlap_fraction) or (iou >= min_iou):
            mapping[ref_id] = int(curr_id)
            assigned_ref.add(ref_id)
            assigned_curr.add(curr_id)
        else:
            mapping[ref_id] = 0  # explicitly no match
    
    # Compose matched_label_image: put pixels of matched curr regions, labeled by ref_id
    for ref_id, cid in mapping.items():
        if cid and cid in curr_ids:
            matched_label_image[curr_labels == cid] = int(ref_id)
    
    return matched_label_image, mapping

try:
    from skimage.measure import label as sklabel
except Exception:
    sklabel = None

def _bbox_of_mask(mask):
    """Return bbox as tuple of slices for ND mask (min..max+1 per axis)."""
    coords = np.nonzero(mask)
    if len(coords) == 0 or coords[0].size == 0:
        return None
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) + 1 for c in coords]
    return tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))

def _expand_bbox(bbox, shape, radius):
    """Expand bbox slices by radius (clamped to image shape)."""
    out = []
    for sl in bbox:
        start = max(0, sl.start - int(np.ceil(radius)))
        stop = min(shape[sl.ndim if hasattr(sl, "ndim") else 0] if False else None, sl.stop)  # placeholder
    # simpler: compute for each axis using slice objects passed in tuple with length = ndims
    return None  # unused; we'll implement expansion inline below

def _make_ball_structuring_element(ndim, radius):
    """Create an ND boolean structuring element approximating an Euclidean ball of given radius."""
    if radius <= 0:
        return np.array([1], dtype=bool)
    r = float(radius)
    # shape = (2*ceil(r)+1,) * ndim
    radi = int(np.ceil(r))
    grids = np.ogrid[tuple(slice(-radi, radi + 1) for _ in range(ndim))]
    dist2 = sum((g.astype(float) ** 2) for g in grids)
    selem = (dist2 <= (r + 1e-8) ** 2)
    return selem

def match_previous_frame_3D(
    ref_labels,
    curr_binary,
    min_overlap_fraction=0.1,
    min_iou=0.05,
    connectivity: int = 1,
    allow_new_objects: bool = True,
    search_radius: float = 0.0,
):
    """
    Faster ND matcher that uses bbox prefiltering and small dilations only where needed.

    Returns (matched_label_image, mapping, new_assignments)
    """

    if ref_labels.shape != curr_binary.shape:
        raise ValueError("ref_labels and curr_binary must have the same shape.")
    ndim = ref_labels.ndim
    shape = ref_labels.shape

    # Label current binary
    struct = ndi.generate_binary_structure(ndim, connectivity)
    curr_labels = ndi.label(curr_binary.astype(bool), structure=struct)[0]

    # collect ids
    ref_ids = np.unique(ref_labels)
    ref_ids = ref_ids[ref_ids != 0]
    curr_ids = np.unique(curr_labels)
    curr_ids = curr_ids[curr_ids != 0]

    matched_label_image = np.zeros_like(ref_labels, dtype=int)
    mapping = {int(r): 0 for r in ref_ids}
    if len(ref_ids) == 0 and len(curr_ids) == 0:
        return matched_label_image, mapping, {}
    if len(curr_ids) == 0:
        return matched_label_image, mapping, {}

    # Precompute bboxes and masks for refs and currs
    ref_bboxes = {}
    ref_masks = {}
    ref_areas = {}
    for rid in ref_ids:
        rid = int(rid)
        mask = (ref_labels == rid)
        bbox = _bbox_of_mask(mask)
        if bbox is None:
            continue
        ref_bboxes[rid] = bbox
        ref_masks[rid] = mask
        ref_areas[rid] = int(mask.sum())

    curr_bboxes = {}
    curr_masks = {}
    curr_areas = {}
    for cid in curr_ids:
        cid = int(cid)
        mask = (curr_labels == cid)
        bbox = _bbox_of_mask(mask)
        if bbox is None:
            continue
        curr_bboxes[cid] = bbox
        curr_masks[cid] = mask
        curr_areas[cid] = int(mask.sum())

    # Precompute structuring element for dilation if radius>0
    selem = _make_ball_structuring_element(ndim, search_radius) if search_radius > 0 else None

    # Candidate overlaps: only compute overlap for ref/curr pairs whose bboxes intersect
    # after expanding ref bbox by radius (in voxels)
    def expand_bbox(bbox, radius):
        radi = int(np.ceil(radius))
        out = []
        for ax, sl in enumerate(bbox):
            start = max(0, sl.start - radi)
            stop = min(shape[ax], sl.stop + radi)
            out.append(slice(start, stop))
        return tuple(out)

    # We'll build a sparse overlap dict keyed by (rid, cid) -> overlap_count (w.r.t dilated ref)
    overlap = {}  # (rid, cid) -> int
    # Also record original intersection for IoU calc: (rid, cid) -> int
    orig_inter = {}

    # For each ref, find candidate currs by bbox intersection
    for rid, rbbox in ref_bboxes.items():
        rbbox_exp = expand_bbox(rbbox, search_radius)
        # compute integer bbox extents to test intersection cheaply
        rstart = [s.start for s in rbbox_exp]
        rstop = [s.stop for s in rbbox_exp]
        # list candidate curr ids whose bbox intersects
        candidates = []
        for cid, cbbox in curr_bboxes.items():
            # check intersection per axis
            intersects = True
            for ax in range(ndim):
                if cbbox[ax].stop <= rstart[ax] or cbbox[ax].start >= rstop[ax]:
                    intersects = False
                    break
            if intersects:
                candidates.append(cid)
        if not candidates:
            continue

        # Crop a small window covering union of ref bbox_exp and candidate curr bboxes to speed ops
        # Compute a tight window to extract small boolean arrays
        # window start = min of rbbox_exp.start and min candidate cbbox.start
        win_start = [rstart[ax] for ax in range(ndim)]
        win_stop = [rstop[ax] for ax in range(ndim)]
        for cid in candidates:
            for ax in range(ndim):
                win_start[ax] = min(win_start[ax], curr_bboxes[cid][ax].start)
                win_stop[ax] = max(win_stop[ax], curr_bboxes[cid][ax].stop)
        win_slices = tuple(slice(s, e) for s, e in zip(win_start, win_stop))

        # extract local arrays once
        ref_local = ref_masks[rid][win_slices]
        # dilate local ref if needed
        if selem is not None:
            # binary_dilation preserves binary dtype
            ref_dilated_local = ndi.binary_dilation(ref_local, structure=selem)
        else:
            ref_dilated_local = ref_local

        # for each candidate curr compute intersections within window
        for cid in candidates:
            curr_local = curr_masks[cid][win_slices]
            inter = int(np.count_nonzero(ref_dilated_local & curr_local))
            if inter > 0:
                overlap[(rid, cid)] = inter
            # original overlap for IoU
            o_inter = int(np.count_nonzero(ref_local & curr_local))
            if o_inter > 0:
                orig_inter[(rid, cid)] = o_inter

    # If no overlaps at all, proceed to new object assignment later
    if len(overlap) == 0 and len(orig_inter) == 0:
        assigned_curr = set()
        assigned_ref = set()
    else:
        # Build dense overlap matrix for Hungarian using found keys
        ref_list = np.array(sorted(list(ref_ids)), dtype=int)
        curr_list = np.array(sorted(list(curr_ids)), dtype=int)
        R = len(ref_list); C = len(curr_list)
        overlap_mat = np.zeros((R, C), dtype=int)
        for (rid, cid), val in overlap.items():
            i = int(np.searchsorted(ref_list, rid))
            j = int(np.searchsorted(curr_list, cid))
            if i < R and ref_list[i] == rid and j < C and curr_list[j] == cid:
                overlap_mat[i, j] = val
        # fill in original intersections into overlap_mat if needed where overlap_mat==0
        for (rid, cid), val in orig_inter.items():
            i = int(np.searchsorted(ref_list, rid))
            j = int(np.searchsorted(curr_list, cid))
            if i < R and ref_list[i] == rid and j < C and curr_list[j] == cid:
                if overlap_mat[i, j] == 0:
                    overlap_mat[i, j] = val

        # Hungarian assignment
        n = max(R, C)
        cost = np.zeros((n, n), dtype=int)
        cost[:R, :C] = -overlap_mat
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_curr = set()
        assigned_ref = set()
        # Evaluate assignments with min_overlap_fraction (using dilated overlap) or IoU (orig overlap)
        for r, c in zip(row_ind, col_ind):
            if r >= R or c >= C:
                continue
            inter = int(overlap_mat[r, c])
            if inter == 0:
                continue
            ref_id = int(ref_list[r])
            curr_id = int(curr_list[c])
            area_ref = int(ref_areas.get(ref_id, 0))
            area_curr = int(curr_areas.get(curr_id, 0))
            # original intersection for IoU (may be 0)
            o_int = int(orig_inter.get((ref_id, curr_id), 0))
            union = int(area_ref + area_curr - o_int)
            iou = (o_int / union) if union > 0 else 0.0
            frac_of_ref = (inter / area_ref) if area_ref > 0 else 0.0

            if (frac_of_ref >= min_overlap_fraction) or (iou >= min_iou):
                mapping[int(ref_id)] = int(curr_id)
                assigned_ref.add(int(ref_id))
                assigned_curr.add(int(curr_id))
            else:
                mapping[int(ref_id)] = 0

    # Compose matched_label_image
    for ref_id, cid in list(mapping.items()):
        if cid and (cid in curr_masks):
            matched_label_image[curr_labels == cid] = int(ref_id)

    # Assign unmatched current regions new labels if requested
    all_curr_set = set(int(c) for c in curr_ids)
    unmatched_curr = sorted(list(all_curr_set - set(assigned_curr)))
    new_assignments = {}
    if allow_new_objects and unmatched_curr:
        next_label = int(max(ref_ids) + 1) if len(ref_ids) > 0 else 1
        for cid in unmatched_curr:
            new_ref = next_label
            new_assignments[new_ref] = int(cid)
            matched_label_image[curr_labels == cid] = new_ref
            mapping[new_ref] = int(cid)
            next_label += 1

    return matched_label_image, mapping, new_assignments

import numpy as np
import pandas as pd
from scipy import ndimage

# ---------- utilities ----------
def largest_inscribed_circle_center_2d(cell_mask2d):
    """Return center (row, col), radius, and distance_map for a 2D binary mask."""
    mask = np.asarray(cell_mask2d, dtype=bool)
    if not mask.any():
        dist_map = np.zeros_like(mask, dtype=float)
        return (np.nan, np.nan), 0.0, dist_map
    dist_map = ndimage.distance_transform_edt(mask)
    max_r = float(dist_map.max())
    max_pos = np.argwhere(dist_map == dist_map.max())[0]
    return (float(max_pos[0]), float(max_pos[1])), max_r, dist_map

def labeled_centroids_3d(label_img3d):
    """
    Given a 3D integer label array (z,y,x) return:
      - centroids_3d: list of (z,y,x) floats for each label>0 (in label order)
      - labels: array of labels in the same order
    """
    labels = np.unique(label_img3d)
    labels = labels[labels != 0]
    if labels.size == 0:
        return [], np.array([], dtype=int)
    # center_of_mass can compute for 3D label image
    centroids = ndimage.center_of_mass(np.ones_like(label_img3d), label_img3d, labels)
    centroids = [(float(z), float(y), float(x)) for (z,y,x) in centroids]
    return centroids, labels

def labeled_voxel_coords_for_label(label_img3d, label_value):
    """Return ndarray of voxel coordinates (n_vox, 3) for a given label (z,y,x order)."""
    coords = np.column_stack(np.nonzero(label_img3d == label_value))
    return coords  # ints

def euclidean(p, q, spacing=(1.0,1.0,1.0)):
    """Euclidean distance between p and q with spacing (dz, dy, dx) in same order as coordinates."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    diff = (p - q) * np.asarray(spacing, dtype=float)
    return float(np.linalg.norm(diff))

# ---------- main pipeline ----------
def compute_avg_distances_by_time_3d(df,
                                     time_col='time',
                                     bacteria_col='bacteria_labels_3d',
                                     cell_col='cell_mask_2d',
                                     mode='project',
                                     spacing=(1.0,1.0,1.0),
                                     z_ref=None,
                                     voxel_min_distance=False,
                                     return_series=False):
    """
    Compute per-time average distance between bacteria objects (3D labels) and the
    largest-inscribed-circle center of a 2D cell mask.

    Parameters
    ----------
    df : pandas.DataFrame with columns (time_col, bacteria_col, cell_col)
         bacteria_col entries are 3D numpy arrays with integer labels (z,y,x).
         cell_col entries are 2D binary arrays (y,x).
    mode : 'project' | '3d' | 'min_voxel'
        'project'   : compute 3D centroids for bacteria, drop z -> compare (y,x) to 2D circle center.
        '3d'        : compute full 3D centroid distances. cell's (y,x) is lifted to (z_ref,y,x).
                      You must provide z_ref (int) or a sequence mapping times to z_ref slices.
        'min_voxel' : compute minimum voxelwise distance from each labelled bacteria object's voxels
                      to the 2D circle center interpreted as a column at z_ref (or to the 2D point with z ignored if z_ref is None).
    spacing : tuple
        If mode == 'project' spacing should be (dy, dx) or if full 3D distances (dz,dy,dx).
        By default spacing=(1,1,1). For 'project' only the last two are used.
    z_ref : int or dict/time-indexed mapping or None
        Reference z slice index for the 2D cell mask when computing 3D distances. If None and mode=='3d' an error is raised.
        If you pass an int, same z_ref applied to all frames. If you pass a dict mapping time->z_ref that will be used per-time.
    voxel_min_distance : bool
        If True and mode == 'min_voxel', compute full 3D voxelwise min distance including z spacing (requires z_ref or uses voxel z positions).
    return_series : bool
        If True return pandas.Series indexed by time.

    Returns
    -------
    dict_or_series : mapping time -> average_distance (float; np.nan if no bacteria)
    """
    # Validate columns
    for col in (time_col, bacteria_col, cell_col):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    # Normalize spacing depending on mode
    spacing = tuple(spacing)
    if mode == 'project':
        # Expect spacing length 2 or 3; use last two as (dy,dx)
        if len(spacing) == 3:
            spacing_2d = (spacing[1], spacing[2])
        else:
            spacing_2d = (spacing[0], spacing[1])
    elif mode in ('3d','min_voxel'):
        if len(spacing) != 3:
            raise ValueError("For 3D computations spacing must be (dz,dy,dx)")

    grouped = {}

    for idx, row in df.iterrows():
        t = row[time_col]
        labels3d = row[bacteria_col]        # expected shape (Z,Y,X)
        cell2d = row[cell_col]             # expected shape (Y,X)

        # compute largest inscribed circle center in 2D cell mask
        (cy, cx), radius, _ = largest_inscribed_circle_center_2d(cell2d)

        # if no cell circle center (empty cell), set distances empty
        if np.isnan(cy):
            distances = []
        else:
            if mode == 'project':
                # compute 3D centroids and project to 2D (use y,x)
                centroids3d, labels = labeled_centroids_3d(labels3d)
                distances = []
                for (z,y,x) in centroids3d:
                    # use 2D image-plane distance (row=y, col=x)
                    d = euclidean((y,x), (cy,cx), spacing=spacing_2d)
                    distances.append(d)

            elif mode == '3d':
                # need a z_ref for cell mask
                # z_ref may be a single int or a mapping time->int
                if isinstance(z_ref, dict):
                    zref_t = z_ref.get(t, None)
                else:
                    zref_t = z_ref
                if zref_t is None:
                    raise ValueError("mode='3d' requires z_ref (int or dict mapping time->z_ref).")
                # compute 3D centroids and compute full 3D distances
                centroids3d, labels = labeled_centroids_3d(labels3d)
                distances = []
                for (z,y,x) in centroids3d:
                    d = euclidean((z,y,x), (zref_t, cy, cx), spacing=spacing)
                    distances.append(d)

            elif mode == 'min_voxel':
                # for each label compute min distance from any voxel to the point (cy,cx) optionally with z_ref
                labels = np.unique(labels3d)
                labels = labels[labels != 0]
                distances = []
                if labels.size == 0:
                    distances = []
                else:
                    if isinstance(z_ref, dict):
                        zref_t = z_ref.get(t, None)
                    else:
                        zref_t = z_ref
                    for labv in labels:
                        vox = labeled_voxel_coords_for_label(labels3d, labv)  # (n,3) ints (z,y,x)
                        if vox.size == 0:
                            continue
                        if zref_t is None:
                            # treat cell center as 2D column: compute 2D distance ignoring z (min over voxels)
                            # compute distance in y,x plane, since cell has only (y,x) center
                            # use spacing for (dy,dx)
                            dy = (vox[:,1] - cy) * spacing[1]  # spacing (dz,dy,dx)
                            dx = (vox[:,2] - cx) * spacing[2]
                            dmin = float(np.min(np.sqrt(dy*dy + dx*dx)))
                        else:
                            # compute full 3D Euclidean distance to (zref_t, cy, cx)
                            dz = (vox[:,0] - zref_t) * spacing[0]
                            dy = (vox[:,1] - cy) * spacing[1]
                            dx = (vox[:,2] - cx) * spacing[2]
                            dmin = float(np.min(np.sqrt(dz*dz + dy*dy + dx*dx)))
                        distances.append(dmin)
            else:
                raise ValueError("Unknown mode: choose 'project', '3d' or 'min_voxel'")

        # append to grouped
        if t not in grouped:
            grouped[t] = []
        grouped[t].extend(distances)

    # compute average per time
    result = {}
    for t, dlist in grouped.items():
        if len(dlist) == 0:
            result[t] = 0 #float('nan')
        else:
            result[t] = float(np.mean(dlist))

    if return_series:
        return pd.Series(result).sort_index()
    return result
#%%
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage as ndi
from tqdm import tqdm

# joblib is used for parallelization
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None


def _bbox_of_mask(mask):
    coords = np.nonzero(mask)
    if len(coords) == 0 or coords[0].size == 0:
        return None
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) + 1 for c in coords]
    return tuple(slice(mi, ma) for mi, ma in zip(mins, maxs))


def _make_ball_structuring_element(ndim, radius):
    if radius <= 0:
        return np.array([1], dtype=bool)
    r = float(radius)
    radi = int(np.ceil(r))
    grids = np.ogrid[tuple(slice(-radi, radi + 1) for _ in range(ndim))]
    dist2 = sum((g.astype(float) ** 2) for g in grids)
    selem = (dist2 <= (r + 1e-8) ** 2)
    return selem


def match_previous_frame_parallel(
    ref_labels,
    curr_binary,
    min_overlap_fraction=0.1,
    min_iou=0.05,
    connectivity: int = 1,
    allow_new_objects: bool = True,
    search_radius: float = 0.0,
    n_jobs: int = -1,
    backend: str = "threading",
):
    """
    Parallel ND matcher: prefilters by bboxes, dilates small windows only when needed,
    and computes overlap per-ref in parallel.

    Parameters
    ----------
    ref_labels : ND int np.ndarray
    curr_binary : ND bool/0-1 np.ndarray
    min_overlap_fraction : float
    min_iou : float
    connectivity : int
    allow_new_objects : bool
    search_radius : float
        Euclidean radius in voxels/pixels used to expand reference mask for overlap.
    n_jobs : int
        Number of parallel jobs for joblib. -1 means use all available cores.
    backend : str
        Joblib backend: 'threading' (default, avoids copying large arrays) or 'loky' (processes).

    Returns
    -------
    matched_label_image, mapping, new_assignments
    """
    if ref_labels.shape != curr_binary.shape:
        raise ValueError("ref_labels and curr_binary must have the same shape.")
    ndim = ref_labels.ndim
    shape = ref_labels.shape

    # Label current binary using scipy.ndimage.label
    struct = ndi.generate_binary_structure(ndim, connectivity)
    curr_labels = ndi.label(curr_binary.astype(bool), structure=struct)[0]

    # collect ids
    ref_ids = np.unique(ref_labels)
    ref_ids = ref_ids[ref_ids != 0]
    curr_ids = np.unique(curr_labels)
    curr_ids = curr_ids[curr_ids != 0]

    matched_label_image = np.zeros_like(ref_labels, dtype=int)
    mapping = {int(r): 0 for r in ref_ids}
    if len(ref_ids) == 0 and len(curr_ids) == 0:
        return matched_label_image, mapping, {}
    if len(curr_ids) == 0:
        return matched_label_image, mapping, {}

    # Precompute bboxes, masks, areas for refs and currs
    ref_bboxes = {}
    ref_masks = {}
    ref_areas = {}
    for rid in ref_ids:
        rid = int(rid)
        mask = (ref_labels == rid)
        bbox = _bbox_of_mask(mask)
        if bbox is None:
            continue
        ref_bboxes[rid] = bbox
        ref_masks[rid] = mask
        ref_areas[rid] = int(mask.sum())

    curr_bboxes = {}
    curr_masks = {}
    curr_areas = {}
    for cid in curr_ids:
        cid = int(cid)
        mask = (curr_labels == cid)
        bbox = _bbox_of_mask(mask)
        if bbox is None:
            continue
        curr_bboxes[cid] = bbox
        curr_masks[cid] = mask
        curr_areas[cid] = int(mask.sum())

    # structuring element for dilation (small)
    selem = _make_ball_structuring_element(ndim, search_radius) if search_radius > 0 else None

    # helper to expand bbox by radius (in voxels)
    def expand_bbox(bbox, radius):
        radi = int(np.ceil(radius))
        out = []
        for ax, sl in enumerate(bbox):
            start = max(0, sl.start - radi)
            stop = min(shape[ax], sl.stop + radi)
            out.append(slice(start, stop))
        return tuple(out)

    # Build list of candidate currs for each ref using bbox intersection (cheap test)
    ref_candidates = {}
    curr_list_items = list(curr_bboxes.items())  # list of (cid, bbox)
    for rid, rbbox in ref_bboxes.items():
        rbbox_exp = expand_bbox(rbbox, search_radius)
        rstart = [s.start for s in rbbox_exp]
        rstop = [s.stop for s in rbbox_exp]
        candidates = []
        for cid, cbbox in curr_list_items:
            intersects = True
            for ax in range(ndim):
                if cbbox[ax].stop <= rstart[ax] or cbbox[ax].start >= rstop[ax]:
                    intersects = False
                    break
            if intersects:
                candidates.append(cid)
        if candidates:
            ref_candidates[rid] = (rbbox_exp, candidates)

    # If no candidates at all, we can skip parallel work
    if len(ref_candidates) == 0:
        assigned_curr = set()
        assigned_ref = set()
        overlap = {}
        orig_inter = {}
    else:
        # Worker function computes overlap and original intersection for a single ref
        def _worker_ref(rid):
            rbbox_exp, candidates = ref_candidates[rid]
            # Compose a tight window covering union of rbbox_exp and candidate cbboxes
            win_start = [rbbox_exp[ax].start for ax in range(ndim)]
            win_stop = [rbbox_exp[ax].stop for ax in range(ndim)]
            for cid in candidates:
                for ax in range(ndim):
                    win_start[ax] = min(win_start[ax], curr_bboxes[cid][ax].start)
                    win_stop[ax] = max(win_stop[ax], curr_bboxes[cid][ax].stop)
            win_slices = tuple(slice(s, e) for s, e in zip(win_start, win_stop))

            # local arrays
            ref_local = ref_masks[rid][win_slices]
            if selem is not None:
                ref_dilated_local = ndi.binary_dilation(ref_local, structure=selem)
            else:
                ref_dilated_local = ref_local

            results_overlap = {}
            results_orig = {}
            for cid in candidates:
                curr_local = curr_masks[cid][win_slices]
                inter_dil = int(np.count_nonzero(ref_dilated_local & curr_local))
                if inter_dil > 0:
                    results_overlap[cid] = inter_dil
                o_inter = int(np.count_nonzero(ref_local & curr_local))
                if o_inter > 0:
                    results_orig[cid] = o_inter
            return rid, results_overlap, results_orig

        # run in parallel if joblib available, else sequential fallback
        if Parallel is not None and delayed is not None:
            # choose backend: threading avoids pickling big arrays (recommended)
            parallel = Parallel(n_jobs=n_jobs, backend=backend, prefer="threads")
            
            jobs = parallel(
                delayed(_worker_ref)(rid)
                for rid in tqdm(list(ref_candidates.keys()), desc="Matching references")
            )
            # aggregate
            overlap = {}
            orig_inter = {}
            for rid, rover, rorig in jobs:
                for cid, val in rover.items():
                    overlap[(rid, cid)] = val
                for cid, val in rorig.items():
                    orig_inter[(rid, cid)] = val
        else:
            # fallback sequential
            overlap = {}
            orig_inter = {}
            for rid in list(ref_candidates.keys()):
                rid, rover, rorig = _worker_ref(rid)
                for cid, val in rover.items():
                    overlap[(rid, cid)] = val
                for cid, val in rorig.items():
                    orig_inter[(rid, cid)] = val

        # After aggregation, if both empty then no candidate overlaps found
        if len(overlap) == 0 and len(orig_inter) == 0:
            assigned_curr = set()
            assigned_ref = set()

    # If we have overlaps or original intersections, build dense matrix for Hungarian
    assigned_curr = set()
    assigned_ref = set()
    if len(ref_candidates) > 0 and (len(overlap) > 0 or len(orig_inter) > 0):
        ref_list = np.array(sorted(list(ref_ids)), dtype=int)
        curr_list = np.array(sorted(list(curr_ids)), dtype=int)
        R = len(ref_list); C = len(curr_list)
        overlap_mat = np.zeros((R, C), dtype=int)

        # fill overlap_mat from overlap first
        for (rid, cid), val in overlap.items():
            i = int(np.searchsorted(ref_list, rid))
            j = int(np.searchsorted(curr_list, cid))
            if i < R and ref_list[i] == rid and j < C and curr_list[j] == cid:
                overlap_mat[i, j] = val
        # supplement with original_intersections where overlap_mat==0
        for (rid, cid), val in orig_inter.items():
            i = int(np.searchsorted(ref_list, rid))
            j = int(np.searchsorted(curr_list, cid))
            if i < R and ref_list[i] == rid and j < C and curr_list[j] == cid:
                if overlap_mat[i, j] == 0:
                    overlap_mat[i, j] = val

        # Hungarian
        n = max(R, C)
        cost = np.zeros((n, n), dtype=int)
        cost[:R, :C] = -overlap_mat
        row_ind, col_ind = linear_sum_assignment(cost)

        # Evaluate assignments using dilation-overlap fraction or IoU
        for r, c in zip(row_ind, col_ind):
            if r >= R or c >= C:
                continue
            inter = int(overlap_mat[r, c])
            if inter == 0:
                continue
            ref_id = int(ref_list[r])
            curr_id = int(curr_list[c])
            area_ref = int(ref_areas.get(ref_id, 0))
            area_curr = int(curr_areas.get(curr_id, 0))
            o_int = int(orig_inter.get((ref_id, curr_id), 0))
            union = int(area_ref + area_curr - o_int)
            iou = (o_int / union) if union > 0 else 0.0
            frac_of_ref = (inter / area_ref) if area_ref > 0 else 0.0

            if (frac_of_ref >= min_overlap_fraction) or (iou >= min_iou):
                mapping[int(ref_id)] = int(curr_id)
                assigned_ref.add(int(ref_id))
                assigned_curr.add(int(curr_id))
            else:
                mapping[int(ref_id)] = 0

    # Compose matched_label_image
    for ref_id, cid in list(mapping.items()):
        if cid and (cid in curr_masks):
            matched_label_image[curr_labels == cid] = int(ref_id)

    # Assign unmatched current regions new labels if requested
    all_curr_set = set(int(c) for c in curr_ids)
    unmatched_curr = sorted(list(all_curr_set - set(assigned_curr)))
    new_assignments = {}
    if allow_new_objects and unmatched_curr:
        next_label = int(max(ref_ids) + 1) if len(ref_ids) > 0 else 1
        for cid in unmatched_curr:
            new_ref = next_label
            new_assignments[new_ref] = int(cid)
            matched_label_image[curr_labels == cid] = new_ref
            mapping[new_ref] = int(cid)
            next_label += 1

    return matched_label_image, mapping, new_assignments
#%% IO
lines_uninfected = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]
lines_infected = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]

binary_masks_PXN_uninfected = {}
fl_PXN_uninfected = {}

for line in lines_infected:
    parentdir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset4D/4D_{line}_uninfected")
    
    # load in bacteria and macrophage masks 
    bin_dir_PXN = parentdir / "binary_volumes" / "PXN_binvol" / "watershed_2D"    
    binary_masks_PXN_uninfected[line] = utils.load_path_into_dict(bin_dir_PXN, keywordregex=[r"t\d{3}"], keyword="separated")

binary_masks_ecoli_infected = {}
binary_masks_PXN_infected = {}
bf_infected = {}
PXN_infected = {}
ecoli_infected = {}

for line in lines_infected:
    parentdir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset4D/4D_{line}_infected")
    
    # load in bacteria and macrophag emasks 
    bin_dir_PXN = parentdir / "binary_volumes" / "PXN_binvol" / "watershed_2D"
    bin_dir_ecoli = parentdir / "binary_volumes" / "ecoli_binvol" 
    
    binary_masks_ecoli_infected[line] = utils.load_path_into_dict(bin_dir_ecoli, keywordregex=[r"t\d{3}"], keyword="binary")
    binary_masks_PXN_infected[line] = utils.load_path_into_dict(bin_dir_PXN, keywordregex=[r"t\d{3}"], keyword="separated")

for line in ["W1118"]: #, "AnxB9rnai"]:
    parentdir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset4D/4D_{line}_infected")
    rawdir = parentdir / "raw" 
    
    bf_infected[line] = utils.load_path_into_dict(rawdir / "timepoint_z_bf", keywordregex=[r"t\d{3}"], keyword="BF")
    PXN_infected[line] = utils.load_path_into_dict(rawdir / "timepoint_z_PXN", keywordregex=[r"t\d{3}"], keyword="PXN")
    ecoli_infected[line] = utils.load_path_into_dict(rawdir / "timepoint_z_ecoli", keywordregex=[r"t\d{3}"], keyword="ecoli")
    
figdir = constants.FIG_DIR / "phagocytosis"
figdir.mkdir(parents=True, exist_ok=True)

vardir = Path(r"C:/Users/IvyWork/Desktop/projects/dataset4D/4D_Phagocytosis_df")
vardir.mkdir(parents=True, exist_ok=True)


#%% prepare binary masks - uninfected

macrophage_labeled_uninfected = {}

for line in lines_uninfected: 
    final_labeled_macrophages = {}
    
    for t, filepath in binary_masks_PXN_uninfected[line].items():
        img = tiff.imread(filepath)
    
        binary = img.astype(bool)
        labeled = label(binary, connectivity=2)  # 2 = 8-connectivity in 2D
        cleaned = remove_small_objects(labeled, min_size=150)
        
        cleaned_binary = (cleaned > 0).astype(np.uint8)
        final_labeled_macrophages[t] = label(cleaned_binary, connectivity=2)
    
    # timepoint matching all macrophage images - match to previous timepoint labelling
    time_matched_macrophages_labeled = {}
    t_prev = "t001"
    
    for t, image in final_labeled_macrophages.items():
        
        if t == t_prev:
            time_matched_macrophages_labeled[t] = image
            continue # no need to time match first timepoint
            
        time1 = time_matched_macrophages_labeled[t_prev]
        time2 = final_labeled_macrophages[t]
        matched_label_image, mapping = match_previous_frame(time1, time2)
        time_matched_macrophages_labeled[t] = matched_label_image
    
        t_prev = t # updated reference timeframe
    
    macrophage_labeled_uninfected[line] = time_matched_macrophages_labeled

#%% plot time matched

selected_line = "W1118"
selected_cell = [3,8,11,16,21,26,31,36,41,46,51]
for t, image in macrophage_labeled_uninfected[selected_line].items():
    
    #mask = (image == selected_cell).astype(np.uint8)  # boolean mask of shape (H, W)
    #mask = image
    relabel = np.zeros_like(image)

    for i, lab in enumerate(selected_cell, start=1):
        relabel[image == lab] = i
    
    mask = np.isin(image, selected_cell).astype(np.uint8) 
    #relabel = np.where(np.isin(image, selected_cell), image, 0)
    
    #mask2D = np.max(mask, axis = 0)
    mask2D = relabel.copy()
    #vol = np.sum(mask)*0.481*0.481*0.49
    
    fig, ax = plt.subplots()
    ax.axis("off")

    im = ax.imshow(mask2D, cmap="tab20")  # good for discrete labels
    ax.set_title(f"Timepoint {t}")

    cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label("Label ID")

    plt.show()

#%%
# color bar separately
mask2D = macrophage_labeled_uninfected[selected_line].items()  # or any representative image

labels = np.unique(mask2D)

cmap = plt.get_cmap("tab20")
norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())

# Create standalone figure for colorbar
fig, ax = plt.subplots(figsize=(2, 6))

cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    ticks=labels,
    orientation='vertical'
)

cb.set_label("Label ID")

plt.show()

#%% prepare binary masks - infected

macrophage_labeled_infected = {}
ecoli_binary_infected = {}

for line in lines_infected: 
    final_labeled_macrophages = {}
    final_ecoli = {}
    
    for t, filepath in binary_masks_PXN_infected[line].items():
        img = tiff.imread(filepath)
    
        binary = img.astype(bool)
        labeled = label(binary, connectivity=2)  # 2 = 8-connectivity in 2D
        cleaned = remove_small_objects(labeled, min_size=150)
        
        cleaned_binary = (cleaned > 0).astype(np.uint8)
        final_labeled_macrophages[t] = label(cleaned_binary, connectivity=2)
    
    # timepoint matching all macrophage images - match to previous timepoint labelling
    time_matched_macrophages_labeled = {}
    t_prev = "t001"
    
    for t, image in final_labeled_macrophages.items():
        
        if t == t_prev:
            time_matched_macrophages_labeled[t] = image
            continue # no need to time match first timepoint
            
        time1 = time_matched_macrophages_labeled[t_prev]
        time2 = final_labeled_macrophages[t]
        matched_label_image, mapping = match_previous_frame(time1, time2, min_overlap_fraction=0.5)
        time_matched_macrophages_labeled[t] = matched_label_image
    
        t_prev = t # updated reference timeframe
        
    
    for t, filepath in binary_masks_ecoli_infected[line].items():
        img_ecoli = tiff.imread(filepath)
        z,y,x = img_ecoli.shape
        
        filtered_ecoli = np.zeros_like(img_ecoli, dtype=np.uint8)
        
        for slice_idx in list(range(z)):
            
            slice_img = img_ecoli[slice_idx,:,:]
            binary = slice_img.astype(bool)
            output = np.zeros_like(binary, dtype=np.uint8)
    
            labeled = label(binary, connectivity=2)  # 2 = 8-connectivity in 2D
            for region in regionprops(labeled):
                if 5 <= region.area <= 30:
                    output[labeled == region.label] = 1
            
            filtered_ecoli[slice_idx,:,:] = output
        
        final_ecoli[t] = filtered_ecoli[3:20,:,:]
        
    macrophage_labeled_infected[line] = time_matched_macrophages_labeled
    ecoli_binary_infected[line] = final_ecoli

#%% label E. coli over time
#selected_line = "W1118"

ecoli_labelled = {}
for selected_line in lines_infected: 
    print(f"Processing {selected_line}")
    t_prev = "t001"
    time_matched_ecoli_labeled = {}
    for t, image in tqdm(ecoli_binary_infected[selected_line].items()):
        
    
        if t == t_prev:
            labelled = label(image.astype(np.uint8))
            time_matched_ecoli_labeled[t] = labelled
            
            continue # no need to time match first timepoint
            
        time1 = time_matched_ecoli_labeled[t_prev]
        time2 = image
        matched_label_image, mapping, new_assignments = match_previous_frame_3D(time1, time2, min_overlap_fraction=0.1, search_radius = 4)
        time_matched_ecoli_labeled[t] = matched_label_image
    
        t_prev = t # updated reference timeframe                                 
    
    ecoli_labelled[selected_line] = time_matched_ecoli_labeled

#%% save labelled E.coli masks

for line in lines_infected:
    np.savez(vardir / f"ecoli_tracked_{line}.npz", **ecoli_labelled[selected_line])

#%% load in E. coli masks again 
ecoli_labelled_loaded = {}

for line in lines_infected:
    ecoli_labelled_loaded[line] = dict(np.load(vardir / f"ecoli_tracked_{line}.npz"))

#%% validate time-matched E.coli

selected_line = "W1118"
ecoli_label = 27

for t, image in ecoli_labelled[selected_line].items():
    
    mask = (image == ecoli_label).astype(np.uint8)  # boolean mask of shape (H, W)
    mask2D = np.max(mask, axis = 0)
    vol = np.sum(mask)*0.481*0.481*0.49
    
    plt.figure()
    plt.imshow(mask2D)
    plt.title(f"Cell {ecoli_label} - {t}, {round(vol,2)}")
    
#%% overlay images

ecoli_label = 27
selected_line = "W1118"

bbox = False
time = 0
for t, image in ecoli_labelled[selected_line].items():
    
    mask = (image == ecoli_label).astype(np.uint8)  # boolean mask of shape (H, W)
    area = mask.sum() * 0.481 * 0.481
    
    if area == 0:
        continue
    
    mask_2D = np.max(mask, axis = 0).astype(np.uint8)
    base_img = tiff.imread(bf_infected[selected_line][t])
    
    PXN_img = tiff.imread(PXN_infected[selected_line][t])
    PXN_sum = PXN_img.max(axis = 0)
    PXN_uint8 = utils.to_uint8_percentile(PXN_sum)
    PXN_binary, _ = utils.threshold_otsu(PXN_uint8) 
    
    ecoli_img = tiff.imread(ecoli_infected[selected_line][t])
    ecoli_sum = ecoli_img.max(axis = 0)
    ecoli_uint8 = utils.to_uint8_percentile(ecoli_sum)
    ecoli_binary, _ = utils.threshold_otsu(ecoli_uint8) 
    
    z,y,x = base_img.shape
    
    if bbox:
        seg_cropped = mask_2D[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    else:
        seg_cropped, bbox = utils.crop_image(mask_2D, mask_2D, padding = 30, show_img = False) # binary map

    base_img_cropped = base_img[round(z/2)][bbox[0]:bbox[1], bbox[2]:bbox[3]]
    PXN_binary_cropped = PXN_binary[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    ecoli_binary_cropped = ecoli_binary[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    
    fig,ax = utils.plot_overlays_masks_cell(#base_img = base_img_cropped, 
                                   masks = {"Macrophage": PXN_binary_cropped, "Ecoli": ecoli_binary_cropped}, 
                                   legend  = False, 
                                   fig_size = (5,5), 
                                   scale_bar=False,
                                   title = f"{time}min: measured area {round(area)}\u00B5m\u00B2"
                                   )
    time += 10    
#%% E. coli lifetime 

def quantify_label_lifetimes(label_dict, fpm=None):
    """
    Quantify how long each label exists in a time series.

    Parameters
    ----------
    label_dict : dict
        Dictionary mapping timepoints -> labeled images
        e.g. {"t001": img1, "t002": img2, ...}

    fps : float, optional
        Frames per second. If provided, duration will be returned in seconds.

    Returns
    -------
    lifetimes : dict
        label_id -> {
            'first_frame': str,
            'last_frame': str,
            'frames_present': int,
            'frames_list': list[str],
            'duration_seconds': float or None
        }
    """

    lifetimes = {}

    # ensure frames processed in time order
    sorted_times = sorted(label_dict.keys())

    frames_seen = {}

    for t in sorted_times:
        img = label_dict[t]

        labels = np.unique(img)
        labels = labels[labels != 0]

        for lab in labels:
            lab = int(lab)

            if lab not in frames_seen:
                frames_seen[lab] = []

            frames_seen[lab].append(t)

    for lab, frames in frames_seen.items():

        first = frames[0]
        last = frames[-1]
        count = len(frames)

        lifetimes[lab] = {
            "first_frame": first,
            "last_frame": last,
            "frames_present": count,
            "frames_list": frames,
            "duration_seconds": count / fpm if fpm else None
        }

    return lifetimes

all_lifetimes = {}
for selected_line in lines_infected:
    all_lifetimes[selected_line] = quantify_label_lifetimes(ecoli_labelled[selected_line], fpm=0.1)

#%% histogram of lifetimes

for selected_line in lines_infected:
    
    lifetimes = all_lifetimes[selected_line]
    # extract lifetimes
    lifetimes_frames = [v["frames_present"] for v in lifetimes.values()]
    
    plt.figure()
    plt.hist(lifetimes_frames, bins=100)
    plt.xlabel("Lifetime (frames)")
    plt.ylabel("Number of labels")
    plt.title(f"Distribution of label lifetimes {selected_line}")
    plt.show()
#%% validate time-matched macrophages

selected_line = "AnxB9rnai"
macrophage_label = 9

for t, image in macrophage_labeled_infected[selected_line].items():
    
    mask = (image == macrophage_label).astype(np.uint8)  # boolean mask of shape (H, W)
    area = np.sum(mask)*0.481*0.481
    
    plt.figure()
    plt.imshow(mask)
    plt.title(f"Cell {macrophage_label} - {t}, {area}")

#%% measure cell properties - whole image
regions = regionprops(final_labeled_macrophages["t001"])
props = regionprops_table(final_labeled_macrophages["t001"], properties=[
    "label", "area", "centroid", "solidity", "major_axis_length", "minor_axis_length", "perimeter"
    ])

cell_properties = pd.DataFrame(props)
cell_properties.to_excel(parentdir / "cell_properties.xlsx")

#%% cell properties uninfected

cell_properties_uninfected = {}
timepoint_properties_uninfected = {}

for line in lines_uninfected:
    
    t = "001"
    timepoint_measures = []
    cell_measures = []
    
    time = 0
    for t, image in macrophage_labeled_uninfected[line].items():
        
        # Precompute unique A labels (skip background 0)
        labeled_macrophage = image
        a_labels = np.unique(labeled_macrophage)
        a_labels = a_labels[a_labels != 0]
                
        for a_lab in a_labels:
            mask = (labeled_macrophage == a_lab)  # boolean mask of shape (H, W)
            n_pixels = mask.sum()
            
            labeled_mask_single_cell = label(mask)
            regions = regionprops(labeled_mask_single_cell)
            region = regions[0]
            
            #plt.figure()
            #plt.imshow(mask)
            #plt.title(f"Cell {a_lab} - {t}, {n_pixels}")
            
            if n_pixels == 0:
                continue
            
            # calculate circularity
            perim = perimeter(mask, neighborhood=8)
            circ = 4.0 * math.pi * (n_pixels) / (perim * perim)
            
            # calculate solidity
            hull = convex_hull_image(mask)             
            convex_area_pixels = float(hull.sum())
            solidity = float(n_pixels) / convex_area_pixels
            ar = region.axis_major_length / region.axis_minor_length
            
            cell_property = {
                "CellID": f"CELL{a_lab}",
                "Timepoint_min": time,
                "CellArea_um": n_pixels * 0.481 * 0.481,
                "CellCircularity": circ,
                "CellSolidity": solidity,
                "CellAspectRatio": ar
                }
            
            cell_measures.append(cell_property)
            
        time += 10 # each iteration 10minutes pass
    
        timepoint_measure = {
            "Timepoint_min": time,
            "Cell_Num": len(a_labels),
            }
        
        timepoint_measures.append(timepoint_measure)
        
    cell_properties_df = pd.DataFrame(cell_measures)
    timepoint_properties_df = pd.DataFrame(timepoint_measures)
    
    cell_properties_uninfected[line] = cell_properties_df
    timepoint_properties_uninfected[line] = timepoint_properties_df

#%% cell properties infected

#co-localization and count of bacteria per cell per timepoint 

plot = False
cell_properties_infected = {}
timepoint_properties_infected = {}
ecoli_cells = {}

for line in lines_infected:
    
    t = "001"
    timepoint_measures = []
    cell_measures = []
    
    min_overlap_pixels = 5
    time = 0
    for t, image in macrophage_labeled_infected[line].items():
        
        filtered_ecoli = ecoli_binary_infected[line][t]
        # Precompute unique A labels (skip background 0)
        labeled_macrophage = image
        a_labels = np.unique(labeled_macrophage)
        a_labels = a_labels[a_labels != 0]
        
        matched_labels = []
        # For speed, make boolean arrays for B nonzero per slice (Z, H, W)
        B_nonzero = (filtered_ecoli != 0)
        
        
        for a_lab in a_labels:
            mask = (labeled_macrophage == a_lab)  # boolean mask of shape (H, W)
            kernel = np.ones((3,3), np.uint8)
            mask_eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations = 1) > 0 # okay to touch the edge of the cell boundary but a threshold has to be within the cell 
            n_pixels = mask.sum()
            
            labeled_mask_single_cell = label(mask)
            regions = regionprops(labeled_mask_single_cell)
            region = regions[0]
            
            if len(regions) > 0:
                region = regions[0]
            #plt.figure()
            #plt.imshow(mask)
            #plt.title(f"Cell {a_lab} - {t}, {n_pixels}")
            
            if n_pixels == 0:
                continue
        
            # Count how many (z, y, x) positions across the stack have B_nonzero == True
            # at the mask positions. This gives the total overlapping pixels across all slices.
            # B_nonzero[:, mask] has shape (Z, n_pixels)
            overlap = int(np.count_nonzero(B_nonzero[:, mask_eroded]))
            
            B_internal = np.logical_and(B_nonzero, mask_eroded)
            B_labeled = label(B_internal, connectivity = 3) # label in 3D
            B_regions = regionprops(B_labeled)
            
            if overlap >= int(min_overlap_pixels):
                matched_labels.append(int(a_lab))
            
            num_internal = sum(
                1 for r in regionprops(B_labeled)
                if r.area >= 5
            )
            
            labels_to_keep = [r.label for r in B_regions if r.area >= 5]
            ecoli_mask = np.where(
                np.isin(B_labeled, labels_to_keep),
                B_labeled,
                0
            )
            
            # calculate circularity
            perim = perimeter(mask, neighborhood=8)
            circ = 4.0 * math.pi * (n_pixels) / (perim * perim)
            
            # calculate solidity
            hull = convex_hull_image(mask)             
            convex_area_pixels = float(hull.sum())
            solidity = float(n_pixels) / convex_area_pixels
            ar = region.axis_major_length / region.axis_minor_length
            
            
            if plot: 
                if num_internal > 5:
    
                    # Project overlap across Z (so we can visualize in 2D)
                    overlap_proj = np.any(B_internal, axis=0)  # (H, W) boolean
                
                    H, W = mask_eroded.shape
                
                    # Create RGB image for overlay
                    overlay = np.zeros((H, W, 3), dtype=np.uint8)
                
                    # Cell mask in green
                    overlay[..., 1] = mask_eroded.astype(np.uint8) * 120
                
                    # Overlap (internal ecoli) in red
                    overlay[..., 0] = overlap_proj.astype(np.uint8) * 255
                
                    # 4) Compute bounding box for the (eroded) cell mask
                    props = regionprops(mask_eroded.astype(np.uint8))
                    if len(props) == 0:
                        # fallback if something odd happens
                        minr, minc, maxr, maxc = 0, 0, H, W
                    else:
                        # if there are multiple regions (rare), pick the largest by area
                        props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
                        minr, minc, maxr, maxc = props_sorted[0].bbox  # (min_row, min_col, max_row, max_col)
                
                    # apply padding and clip to image bounds
                    minr_p = max(minr - 10, 0)
                    minc_p = max(minc - 10, 0)
                    maxr_p = min(maxr + 10, H)
                    maxc_p = min(maxc + 10, W)
                
                    # 5) Crop overlay to bbox
                    cropped_overlay = overlay[minr_p:maxr_p, minc_p:maxc_p]
                
                    # 6) Side-by-side display with matplotlib
                    fig, axes = plt.subplots(1, 2, figsize=(6,6))
                    ax_full, ax_crop = axes
                
                    ax_full.imshow(overlay)
                    ax_full.set_title(f"Full image — Cell {a_lab} | Time {time} min | n_internal={num_internal}")
                    ax_full.axis("off")
                
                    ax_crop.imshow(cropped_overlay)
                    ax_crop.set_title(f"Cropped to bbox (pad={10}px)")
                    ax_crop.axis("off")
                    
                    ecoli_regions = regionprops(B_labeled)
                    for i, r in enumerate(ecoli_regions, start=1):
                        if r.area < 5:
                            continue
                
                        # r.centroid gives (z, y, x) because B_labeled is 3D
                        zc, yc, xc = r.centroid
                
                        # FULL IMAGE LABEL
                        ax_full.text(
                            xc, yc,
                            str(i),
                            color='yellow',
                            fontsize=10,
                            weight='bold'
                        )
                
                        # CROPPED IMAGE LABEL
                        if (minr_p <= yc <= maxr_p) and (minc_p <= xc <= maxc_p):
                            ax_crop.text(
                                xc - minc_p,
                                yc - minr_p,
                                str(i),
                                color='yellow',
                                fontsize=10,
                                weight='bold'
                            )
                
                    plt.tight_layout()
                    plt.show()
                        
            ecoli_size = [r.area for r in regionprops(B_labeled) if r.area > 5]
            ecoli_size_physical = [(size*0.481*0.481*0.49) for size in ecoli_size]
            
            cell_property = {
                "CellID": f"CELL{a_lab}",
                "Timepoint_min": time,
                "Ecoli_Num": num_internal,
                "CellArea_um": n_pixels * 0.481 * 0.481,
                "Ecoli_size_um3_list": ecoli_size_physical,
                "Ecoli_size_avg": statistics.mean(ecoli_size_physical) if len(ecoli_size_physical) > 1 else 0,
                "Ecoli_total_vol_um3": (sum(ecoli_size))*0.481*0.481*0.49,
                "Ecoli_labeled": ecoli_mask,
                "CellCircularity": circ,
                "CellSolidity": solidity,
                "CellMask": mask,
                "CellAspectRatio": ar
                }
            
            cell_measures.append(cell_property)
            
        time += 10 # each iteration 10minutes pass
    
        timepoint_measure = {
            "Ecoli+_cells": len(matched_labels),
            "Ecoli+_proportion": len(matched_labels) / len(a_labels),
            "Timepoint_min": time,
            "Cell_Num": len(a_labels),
            }
        
        timepoint_measures.append(timepoint_measure)
        
    cell_properties_df = pd.DataFrame(cell_measures)
    timepoint_properties_df = pd.DataFrame(timepoint_measures)
    
    cell_properties_infected[line] = cell_properties_df
    timepoint_properties_infected[line] = timepoint_properties_df
    ecoli_cells[line] = matched_labels
    
#%% save calculation variables
for line in lines_infected:
    cell_properties_infected[line].to_pickle(vardir / f"cell_properties_infected_{line}.pkl")
    cell_properties_uninfected[line].to_pickle(vardir / f"cell_properties_uninfected_{line}.pkl")

    timepoint_properties_infected[line].to_pickle(vardir / f"timepoint_properties_infected_{line}.pkl")
    timepoint_properties_uninfected[line].to_pickle(vardir / f"timepoint_properties_uninfected_{line}.pkl")

#%% read in calculation variables
cell_properties_infected_loaded = {}
cell_properties_uninfected_loaded = {}

for line in lines_infected:
    cell_properties_infected_loaded[line] = pd.read_pickle(vardir / f"cell_properties_infected_{line}.pkl")
    cell_properties_uninfected_loaded[line] = pd.read_pickle(vardir / f"cell_properties_uninfected_{line}.pkl")

#%% per cell e. coli tracking
#selected_line = "W1118"
ecoli_labelled_percell_df = {}
cell_properties_infected_ecoli = {}

for selected_line in lines_infected:
    
    print(f"Processing {selected_line}")
    
    cells = list(np.unique(cell_properties_infected[selected_line]["CellID"]))
    line_df = cell_properties_infected[selected_line]
    ecoli_labelled_percell_list = []
            
    for selected_cell in tqdm(cells):
        per_cell_df = line_df[line_df["CellID"] == selected_cell].sort_values(by = "Timepoint_min")
        time_matched_ecoli_percell = {}
        t_prev = 0
    
        for row in per_cell_df.itertuples(index = False):
            
            #print(f" Processing {row.Timepoint_min} - tprev {t_prev}")
            
            ecoli_mask = row.Ecoli_labeled
            cell_mask = (row.CellMask > 0).astype(np.uint8)

            if row.Timepoint_min == 0: # first timepoint
                cell_mask_cropped, bbox = utils.crop_image(cell_mask, cell_mask, padding = 50, show_img = False)
            
            else:
                cell_mask_cropped = cell_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            
            ecoli_cropped = ecoli_mask[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]
            #ecoli_mask_2D = np.max(ecoli_mask, axis = 0)
            
            #plt.figure()
            #plt.imshow(ecoli_mask_2D)
            #plt.axis("off")
                        
            if row.Timepoint_min == 0:
                time_matched_ecoli_percell[row.Timepoint_min] = ecoli_cropped
                time_matched_ecoli_cell = {
                    "CellID": row.CellID,
                    "Timepoint_min": row.Timepoint_min,
                    "Labelled_Ecoli_time": ecoli_cropped,
                    "CellMask": cell_mask_cropped,
                    "ImageCrop": bbox,
                    }
                
                t_prev = row.Timepoint_min
                ecoli_labelled_percell_list.append(time_matched_ecoli_cell)
                continue # no need to time match first timepoint
            
            time1 = time_matched_ecoli_percell[t_prev]
            time2 = ecoli_cropped
            
            matched_label_image, mapping, new_assignments = match_previous_frame_3D(time1, time2, min_overlap_fraction=0.1, search_radius = 10)
            time_matched_ecoli_percell[row.Timepoint_min] = matched_label_image
        
            t_prev = row.Timepoint_min # updated reference timeframe                                 
            
            time_matched_ecoli_cell = {
                "CellID": row.CellID,
                "Timepoint_min": row.Timepoint_min,
                "Labelled_Ecoli_time": matched_label_image,
                "CellMask": cell_mask_cropped,
                "ImageCrop": bbox,
                }
            
            ecoli_labelled_percell_list.append(time_matched_ecoli_cell)
        
    #lifetimes_percell = quantify_label_lifetimes(time_matched_ecoli_percell, fpm=0.1)
    ecoli_labelled_percell_df[selected_line] = pd.DataFrame(ecoli_labelled_percell_list)
    cell_properties_infected_ecoli[selected_line] = cell_properties_infected[selected_line].merge(ecoli_labelled_percell_df[selected_line], on=["CellID", "Timepoint_min"], how = "inner")

#%%
cell_properties_infected_ecoli = {}
for selected_line in lines_infected:
    cell_properties_infected_ecoli[selected_line] = cell_properties_infected[selected_line].merge(ecoli_labelled_percell_df[selected_line], on=["CellID", "Timepoint_min"], how = "inner")

#%% save dataframe
for line in lines_infected:
    cell_properties_infected_ecoli[line].to_pickle(vardir / f"cell_properties_infected_ecoli_{line}.pkl")

#%% load in previously calculated e. coli time matching
cell_properties_infected_ecoli_loaded = {}

for line in lines_infected:
    cell_properties_infected_ecoli_loaded[line] = pd.read_pickle(vardir / f"cell_properties_infected_ecoli_{line}.pkl")

#%% calculate lifetimes + distances

timepoints = np.unique(ecoli_labelled_percell_df["W1118"]["Timepoint_min"])

lifetime_measures_all = {}

for selected_line in lines_infected:
    lifetimes_percell = {}
    lifetime_measures = []
    line_df = ecoli_labelled_percell_df[selected_line]
    
    cells = np.unique(line_df["CellID"])
    print(f"Processing {selected_line}")
    
    for cell in tqdm(cells):
        
        cell_df = line_df[line_df["CellID"] == cell]
        # calculate bacteria distance from centroid
        dist_avg_by_time = compute_avg_distances_by_time_3d(cell_df,
                                              time_col='Timepoint_min',
                                              bacteria_col='Labelled_Ecoli_time',
                                              cell_col='CellMask',
                                              mode='project',
                                              spacing=(0.49,0.481,0.481))
        
        avg_spread_list = [v for v in dist_avg_by_time.values() if v != 0] # only use distances where E. coli are present
        dist_avg_alltime = np.mean(avg_spread_list) if len(avg_spread_list) != 0 else 0 # ignore cells that never have E. coli
                                              
        new_labels_per_frame = {t:0 for t in timepoints}
        
        df = line_df[line_df["CellID"] == cell]
        time_matched_ecoli = dict(zip(df["Timepoint_min"], df["Labelled_Ecoli_time"]))
        lifetime_stats = quantify_label_lifetimes(time_matched_ecoli, fpm = 0.1)
        lifetimes_frames = [v["frames_present"] for v in lifetime_stats.values()]
    
        if len(lifetimes_frames) == 0:
            lifetime_avg = 0.0
            lifetime_upper = 0.0
            lifetime_sem = 0.0
        
        else:
            lifetime_avg = np.mean(lifetimes_frames)
            lifetime_upper = np.percentile(lifetimes_frames, 75)
            lifetime_std = np.std(lifetimes_frames)
            lifetime_sem = lifetime_std / np.sqrt(len(lifetimes_frames))
        
        
        for ecoli, info in lifetime_stats.items():
            first_frame = info["first_frame"]
            new_labels_per_frame[first_frame] += 1
        
        internalization_rate = [v/10 for v in new_labels_per_frame.values()]
        internalization_rate_overall = np.sum([v for v in new_labels_per_frame.values()]) / 240
        
        if len(internalization_rate) == 0:
            internalisation_avg = 0.0
            internalisation_upper = 0.0
            internalisation_sem = 0.0
        
        else:
            internalisation_avg = np.mean(internalization_rate)
            internalisation_upper = np.percentile(internalization_rate, 75)
            internalisation_std = np.std(internalization_rate)
            internalisation_sem = internalisation_std / np.sqrt(len(internalization_rate))
        
        lifetime_measure = {
            "CellID": cell,
            "LifetimeAvg_min": lifetime_avg * 10, # lifetime in minutes
            "Lifetime75_min": lifetime_upper * 10, # violin plot
            "Lifetime_std": lifetime_std,
            "Lifetime_sem": lifetime_sem,
            "Lifetimes_min": [life*10 for life in lifetimes_frames], # violin plot
            "NewEcoli_perFrame": new_labels_per_frame,
            "RateNewEcoli_perMin":internalization_rate, # violin plot
            "RateNewEcoli_avg": internalisation_avg,
            "RateNewEcoli_std": internalisation_std,
            "RateNewEcoli_sem": internalisation_sem,
            "RateNewEcoli_overall": internalization_rate_overall,
            "Spread_Avg_um": dist_avg_alltime,
            "Spread_perFrame": dist_avg_by_time,
            }
        
        lifetimes_percell[cell] = lifetime_stats
        lifetime_measures.append(lifetime_measure)
        
    lifetime_measures_df = pd.DataFrame(lifetime_measures)
    lifetime_measures_all[selected_line] = lifetime_measures_df

#%% visualize e. coli lifetime - image (overlay fluorescence + segmentation over time)
from matplotlib.colors import LinearSegmentedColormap
import plotting as plot

selected_line = "W1118"
cell_label = 10

magenta_cmap = LinearSegmentedColormap.from_list(
    "magenta_map",
    [
        (0.0, (0, 0, 0)),      # black
        (0.4, (1, 0, 1)),      # magenta appears earlier
        (1.0, (1, 0.8, 1))     # lighter magenta at high values
    ]
)

cyan_cmap = LinearSegmentedColormap.from_list(
    "cyan_map",
    [(0, 0, 0), (0, 1, 1)]  # black → cyan
)

line_df = ecoli_labelled_percell_df[selected_line]

parentdir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset4D/4D_{selected_line}_infected")

fl_PXN_infected = utils.load_path_into_dict(parentdir / "ilastik_segmentation_input" / "timepoint_z_PXN", keywordregex=[r"t\d{3}"], keyword="PXN")
fl_ecoli_infected = utils.load_path_into_dict(parentdir / "ilastik_segmentation_input" / "timepoint_z_ecoli", keywordregex=[r"t\d{3}"], keyword="ecoli")

timepoints = np.unique(line_df["Timepoint_min"])
iteration = 1

for time in timepoints:
    
    cell_mask = line_df.loc[
        (line_df["CellID"] == f"CELL{cell_label}") & (line_df["Timepoint_min"] == time),
        "CellMask"
    ].iloc[0]
    
    # e. coli segmentation 
    ecoli_mask = line_df.loc[
        (line_df["CellID"] == f"CELL{cell_label}") & (line_df["Timepoint_min"] == time),
        "Labelled_Ecoli_time"
    ].iloc[0]
    
    bbox_display = line_df.loc[
        (line_df["CellID"] == f"CELL{cell_label}") & (line_df["Timepoint_min"] == time),
        "ImageCrop"
    ].iloc[0]
    
    ecoli_mask_2D = np.max(ecoli_mask, axis = 0)
    ecoli_bin = (ecoli_mask_2D > 0).astype(np.uint8)
    
    # fluorescence masks
    t = f"{iteration:03d}"

    original_fl_PXN = tiff.imread(fl_PXN_infected[f"t{t}"])
    original_fl_ecoli = tiff.imread(fl_ecoli_infected[f"t{t}"])
    PXN_proj = np.max(original_fl_PXN, axis = 0) # mean intensity PXN
    ecoli_proj = np.max(original_fl_ecoli, axis = 0) # mean intensity E.coli
    
    PXN_proj_cropped = PXN_proj[bbox_display[0]:bbox_display[1], bbox_display[2]:bbox_display[3]]
    ecoli_proj_cropped = ecoli_proj[bbox_display[0]:bbox_display[1], bbox_display[2]:bbox_display[3]]
    
    rgb = utils.overlay_grayscale_images(ecoli_proj_cropped, PXN_proj_cropped, cmap1=magenta_cmap, cmap2=cyan_cmap, alpha=0.4)
    
    fig,ax = utils.plot_overlays_masks_cell(base_img = rgb, 
                                   outlines = {"ecoli": ecoli_bin, "PXN": cell_mask}, 
                                   legend  = False, 
                                   fig_size = (5,5), 
                                   scale_bar=False,
                                   title = f"{time}min"
                                   )
    iteration += 1

# visualize e. coli lifetimes - chart
df = line_df[line_df["CellID"] == f"CELL{cell_label}"]
time_matched_ecoli = dict(zip(df["Timepoint_min"], df["Labelled_Ecoli_time"]))
lifetime_stats = quantify_label_lifetimes(time_matched_ecoli, fpm = 0.1)

fig, ax, df_used = plot.plot_lifetimes_gantt(
    lifetime_stats,
    #fps=0.1,            # optional: convert frames -> seconds
    max_labels=50,      # plot at most 300 rows
    sort_by="start",
    show_label_ticks=False
)
plt.show()
#%% 
a = lifetime_measures_all["W1118"]["Spread_Avg_um"]
#%% plot measure values table


# combine into single dataframe
lifetime_measures_all_table = pd.concat(
    df.assign(Line=key)
    for key, df in lifetime_measures_all.items()
).reset_index(drop=True)

table_measures = ["RateNewEcoli_avg"]

lifetime_measures_all_table["RateNewEcoli_avg"] *= 60

lifetime_measures_all_table["Line"] = pd.Categorical(
    lifetime_measures_all_table["Line"],
    categories=lines_infected,
    ordered=True
)

avg_per_line = (
    lifetime_measures_all_table
    .groupby("Line", sort = False)[table_measures]
    .mean()
    .round(2)
    .reset_index()
).rename(columns = {"LifetimeAvg_min": "Lifetime x\u0304 \u00B1 SEM (min)", "Lifetime_std": "Lifetime \u03C3"})

utils.create_table(avg_per_line, body_fontsize=8, header_fontsize=8, fig_size=(5,1.5))

#%% SEM

new_col = "Internalisation Rate x\u0304 \u00B1 SEM (per hour)"
# combine into single dataframe
lifetime_measures_all_table = pd.concat(
    df.assign(Line=key)
    for key, df in lifetime_measures_all.items()
).reset_index(drop=True)

table_measures = ["RateNewEcoli_avg", "Lifetime_std"]

lifetime_measures_all_table["RateNewEcoli_avg"] *= 60

lifetime_measures_all_table["Line"] = pd.Categorical(
    lifetime_measures_all_table["Line"],
    categories=lines_infected,
    ordered=True
)

avg_per_line = (
    lifetime_measures_all_table
    .groupby("Line", sort=False, observed=False)
    .agg(
        Lifetime_mean=("RateNewEcoli_avg", "mean"),
        Lifetime_sem=("RateNewEcoli_avg", "sem"),
        Lifetime_std=("RateNewEcoli_avg", "std"),
    )
    .round(2)
    .reset_index()
)

# Create formatted column
avg_per_line["Internalisation Rate x\u0304 \u00B1 SEM (per hour)"] = (
    avg_per_line["Lifetime_mean"].astype(str)
    + " ± "
    + avg_per_line["Lifetime_sem"].astype(str)
)

# Rename std column
avg_per_line = avg_per_line.rename(
    columns={"Lifetime_std": "Lifetime \u03C3"}
)

# Optional: keep only nice output columns
avg_per_line = avg_per_line[
    ["Line", new_col]#, "Lifetime \u03C3"]
]

utils.create_table(avg_per_line, body_fontsize=8, header_fontsize=8, fig_size=(6,1.5))

#%% histogram of lifetimes
import seaborn as sns
plot_wt = False
selected_line = "W1118"

for selected_line in lines_infected:
    measure = "Spread_Avg_um"
    df = lifetime_measures_all[selected_line]
    df_expanded = df.explode(measure).reset_index(drop=True)
    
    plt.figure(figsize=(4,4))
    sns.histplot(data=df_expanded, 
                 x=measure, 
                 bins = 15, 
                 color = constants.CMAP[selected_line], 
                 alpha = 0.8, 
                 edgecolor = constants.CMAP[selected_line],
                 stat="probability")
    if plot_wt:
        df_ref = lifetime_measures_all["W1118"]
        df_expanded_ref =  df_ref.explode(measure).reset_index(drop=True)
        sns.histplot(df_expanded_ref[measure], 
                     stat="probability", 
                     bins=25, 
                     color=constants.CMAP["W1118"], 
                     alpha=0.4, 
                     edgecolor = constants.CMAP["W1118"])

    plt.xlabel("")
    plt.ylabel("")
    plt.ylim(top = 0.5)
    plt.xlim(left = 0, right = 15)
    plt.show()

#%% combine lifetime spread measures with cell_properties - individual

cell_properties_infected_ecoli_full2 = cell_properties_infected_ecoli.copy()

measure = "NewEcoli_perFrame"
for selected_line in lines_infected:
    
    lifetime_df = lifetime_measures_all[selected_line]
    lifetime_df["dict_items"] = lifetime_df[measure].apply(lambda x: list(x.items()))
    lifetime_measures_df_timeseparated = lifetime_df.explode("dict_items")

    lifetime_measures_df_timeseparated[["Timepoint_min", measure]] = pd.DataFrame(
        lifetime_measures_df_timeseparated["dict_items"].tolist(),
        index=lifetime_measures_df_timeseparated.index
    )

    lifetime_measures_df_timeseparated = lifetime_measures_df_timeseparated.drop(columns=["dict_items"])
    
    lifetime_measures_df_timeseparated['Spread_perFrame'] = (
        lifetime_measures_df_timeseparated['Spread_perFrame']
        .replace("nan", np.nan)
        .pipe(pd.to_numeric, errors='coerce')
        .fillna(0)
    ) 

    cell_prop_df = cell_properties_infected_ecoli_full2[selected_line]
    cell_properties_infected_ecoli_full2[selected_line] = lifetime_measures_df_timeseparated.merge(cell_prop_df, on=['CellID','Timepoint_min'], how='inner')

#%% combine two dictionary column measures with cell_properties
measures = ["Spread_perFrame", "NewEcoli_perFrame"]

cell_properties_infected_ecoli_full = cell_properties_infected_ecoli.copy()

for measure in measures:
    
    for selected_line in lines_infected: 
        lifetime_df = lifetime_measures_all[selected_line]
        lifetime_df["dict_items"] = lifetime_df[measure].apply(lambda x: list(x.items()))
        lifetime_measures_df_timeseparated = lifetime_df.explode("dict_items")

        lifetime_measures_df_timeseparated[["Timepoint_min", measure]] = pd.DataFrame(
            lifetime_measures_df_timeseparated["dict_items"].tolist(),
            index=lifetime_measures_df_timeseparated.index
        )

        lifetime_measures_df_timeseparated = lifetime_measures_df_timeseparated.drop(columns=["dict_items"])
        
        lifetime_measures_df_timeseparated[measure] = (
            lifetime_measures_df_timeseparated[measure]
            .replace("nan", np.nan)
            .pipe(pd.to_numeric, errors='coerce')
            .fillna(0)
        ) 
        
        to_merge = lifetime_measures_df_timeseparated[["CellID", "Timepoint_min", measure]]
        cell_prop_df = cell_properties_infected_ecoli_full[selected_line]
        cell_properties_infected_ecoli_full[selected_line] = to_merge.merge(
        cell_prop_df, on=["CellID", "Timepoint_min"], how="inner"
        )
#%% compare properties - all properties
#%% check e. coli time correction
selected_line = "W1118"
ecoli_label = 25

bbox = False
for t, image in ecoli_labelled_percell[selected_line].items():
    
    cell_mask = cell_properties_infected[selected_line].loc[cell_properties_infected[selected_line]['Timepoint_min'] == t, 'CellMask'].values[0]
    
    if t == 0: 
        continue

    if bbox:
        cell_mask_cropped = cell_mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(np.uint8)
    
    mask = (image == ecoli_label).astype(np.uint8)  # boolean mask of shape (H, W)
    #mask = image

    mask2D = np.max(mask, axis = 0)
    vol = np.sum(mask)*0.481*0.481*0.49
    
    plt.figure()
    plt.imshow(cell_mask_cropped, cmap="gray")
    red_overlay = np.zeros((cell_mask_cropped.shape[0], cell_mask_cropped.shape[1], 4), dtype=float)
    red_overlay[..., 0] = 1.0   # R
    red_overlay[..., 3] = 0.6 * mask2D.astype(float)  # alpha where overlap

    plt.imshow(red_overlay)
    plt.title(f"Cell {ecoli_label} - {t}, {round(vol,2)}")
    plt.axis("off")

#%% plot E.coli number in every cell over time
selected_line = "AnxB9rnai"
cell_feature_list = list(cell_property.keys())
feature_plot = "CellCircularity"
plot.line_plot_over_time(cell_properties_infected_protrusions[selected_line], 
                         feature_plot, "Timepoint_min", "CellID", selected_line = selected_line, 
                         plot_all = True, separate_plots = True, 
                         full_tracks = False, plot_one = False)

#%% plot binary map of cell shape over time overlaid on BF

selected_cell = 65
selected_line = "W1118"

iteration = 0
time = 0
for t, image in macrophage_labeled_infected[selected_line].items():
    
    mask = (image == selected_cell).astype(np.uint8)  # boolean mask of shape (H, W)
    area = mask.sum() * 0.481 * 0.481
    
    base_img = tiff.imread(bf_infected[selected_line][t])
    PXN_img = tiff.imread(PXN_infected[selected_line][t])
    PXN_sum = PXN_img.max(axis = 0)
    PXN_uint8 = utils.to_uint8_percentile(PXN_sum)
    PXN_binary, _ = utils.threshold_otsu(PXN_uint8) 

    z,y,x = base_img.shape
    
    if iteration == 0:
        seg_cropped, bbox = utils.crop_image(mask, mask, padding = 30, show_img = False) # binary map
    else:
        seg_cropped = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        
    base_img_cropped = base_img[round(z/2)][bbox[0]:bbox[1], bbox[2]:bbox[3]]
    PXN_binary_cropped = PXN_binary[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    
    fig,ax = utils.plot_overlays_masks_cell(base_img_cropped, 
                                   outlines = {"Cell": seg_cropped}, 
                                   masks = {"PXN": PXN_binary_cropped}, 
                                   legend  = False, 
                                   fig_size = (5,5), 
                                   scale_bar=True,
                                   title = f"{time}min: measured area {round(area)}\u00B5m\u00B2"
                                   )
    time += 10    
    iteration += 1
#%% overlay both fly lines - infected

cell_feature_list = list(cell_properties_infected_ecoli.keys())
feature_plot = "NewEcoli_count"

    
plt.figure()

for line in lines_infected:  
    mean_df = (
        cell_properties_infected_ecoli[line]
        .groupby("Timepoint_min")[feature_plot]
        .mean()
        .reset_index()
    )

    plt.plot(
        mean_df["Timepoint_min"],
        mean_df[feature_plot],
        linewidth=1,
        label=line, 
        color = constants.CMAP[line]
    )

plt.xlabel("Timepoint (min)")
plt.ylabel(f"Mean {feature_plot}")
plt.title(f"Mean {feature_plot} over Time")
plt.legend()
plt.show()
    
#%% plot cell area over time - all cell lines
labels = False

feature_plot = "NewEcoli_count"

plt.figure(figsize = (7,7))

# Uninfected → dashed lines
for line in lines_uninfected:
    mean_df = (
        cell_properties_uninfected[line]
        .groupby("Timepoint_min")[feature_plot]
        .mean()
        .reset_index()
    )

    plt.plot(
        mean_df["Timepoint_min"],
        mean_df[feature_plot],
        linewidth=1.5,
        linestyle='--',
        label=f"{line} (uninfected)",
        color = constants.CMAP[line]
    )

# Infected → solid lines
for line in lines_infected:
    mean_df = (
        cell_properties_infected[line]
        .groupby("Timepoint_min")[feature_plot]
        .mean()
        .reset_index()
    )

    plt.plot(
        mean_df["Timepoint_min"],
        mean_df[feature_plot],
        linewidth=1.5,
        linestyle='-',
        label=f"{line} (infected)",
        color = constants.CMAP[line]
    )

if labels:
    plt.xlabel("Timepoint (min)")
    plt.ylabel(f"Mean {feature_plot}")
    plt.title(f"Mean {feature_plot} over Time")
else:
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")

plt.legend()
plt.show()

#%% normalized to time 0 

# normalize each cell to baseline 
# try deviation from baseline (percent change from baseline value - baseline / baseline)

plt.figure()

# Uninfected
for line in lines_uninfected:
    mean_df = (
        cell_properties_uninfected[line]
        .groupby("Timepoint_min")[feature_plot]
        .mean()
        .reset_index()
        .sort_values("Timepoint_min")
    )

    t0_value = mean_df.loc[
        mean_df["Timepoint_min"] == 0, feature_plot
    ].values[0]

    mean_df["normalized"] = mean_df[feature_plot] / t0_value

    plt.plot(
        mean_df["Timepoint_min"],
        mean_df["normalized"],
        linewidth=1.5,
        linestyle='--',
        label=f"{line} (uninfected)", 
        color = constants.CMAP[line]
    )
    
# Infected
for line in lines_infected:
    mean_df = (
        cell_properties_infected[line]
        .groupby("Timepoint_min")[feature_plot]
        .mean()
        .reset_index()
        .sort_values("Timepoint_min")
    )

    # Get value at time 0
    t0_value = mean_df.loc[
        mean_df["Timepoint_min"] == 0, feature_plot
    ].values[0]

    # Normalize
    mean_df["normalized"] = mean_df[feature_plot] / t0_value

    plt.plot(
        mean_df["Timepoint_min"],
        mean_df["normalized"],
        linewidth=1.5,
        linestyle='-',
        label=f"{line} (infected)",
        color = constants.CMAP[line]
    )

plt.xlabel("Timepoint (min)")
plt.ylabel(f"{feature_plot} (normalized to t=0)")
plt.title(f"Normalized {feature_plot} over Time")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

#%% plot sem
import plotting as plot
#all_feat = list(cell_properties_infected_ecoli_full["W1118"].columns)
all_feat = list(cell_properties_infected["W1118"].columns)

feature_plot = "CellCircularity"
cell_prop_inf_expanded = {}
cell_prop_uninf_expanded = {}

'''
for line in lines_infected:
    cell_prop_inf_expanded[line] = cell_properties_infected[line].explode(feature_plot).reset_index(drop=True) # only do if feature is recorded in a list
    cell_prop_uninf_expanded[line] = cell_properties_uninfected[line].explode(feature_plot).reset_index(drop=True)
 '''   
lines_uninfected_plot = lines_uninfected
lines_uninfected_plot = ["W1118", "AnxB11rnai"] # only compare two at a time
lines_infected_plot = lines_uninfected_plot

plot.plot_groups_with_band(lines_uninfected_plot, lines_infected_plot,
                       cell_properties_uninfected_protrusions, cell_properties_infected_protrusions,
                       feature_plot, band="sem", cmap_map=constants.CMAP, labels = False, 
                       figsize = (5,5), normalize = False, all_conditions = False, 
                       include_zeros = False, ylimit = 1)

# To plot the range:
# plot_groups_with_band(..., band="range")

#%% measure of morphodynamic speed
from tqdm import tqdm 

morphodynamic_speed_inf = {}
morphodynamic_speed_uninf = {}

for line in tqdm(lines_infected):
    df_inf = cell_properties_infected[line]
    df_inf = df_inf.sort_values(['CellID', 'Timepoint_min']).copy()
    df_inf['CircChange'] = df_inf.groupby('CellID')['CellCircularity'].diff().abs()
    df_inf['CircChange'] = df_inf['CircChange'].fillna(0)
    
    morphodynamic_speed_inf[line] = (
        df_inf.groupby('CellID', as_index=False)['CircChange']
          .mean()
          .rename(columns={'CircChange': 'CircChange_avg'})
    )
    
    df_uninf = cell_properties_uninfected[line]
    df_uninf = df_uninf.sort_values(['CellID', 'Timepoint_min']).copy()
    df_uninf['CircChange'] = df_uninf.groupby('CellID')['CellCircularity'].diff().abs()
    df_uninf['CircChange'] = df_uninf['CircChange'].fillna(0)
    
    morphodynamic_speed_uninf[line] = (
        df_uninf.groupby('CellID', as_index=False)['CircChange']
          .mean()
          .rename(columns={'CircChange': 'CircChange_avg'})
    )

# modify original dataframe
cell_properties_infected_protrusions = {}
cell_properties_uninfected_protrusions = {}

for line in lines_infected:
    
    df = cell_properties_infected[line]
    df = df.sort_values(['CellID', 'Timepoint_min']).copy()
    df['CircularityChangeAbs'] = df.groupby('CellID')['CellCircularity'].diff().abs()
    df['CircularityChangeAbs'] = df['CircularityChangeAbs'].fillna(0)
    df['CircularityChange'] = df.groupby('CellID')['CellCircularity'].diff()
    df['CircularityChange'] = df['CircularityChange'].fillna(0)
    
    cell_properties_infected_protrusions[line] = df
    
    df_uninf = cell_properties_uninfected[line]
    df_uninf = df_uninf.sort_values(['CellID', 'Timepoint_min']).copy()
    df_uninf['CircularityChangeAbs'] = df_uninf.groupby('CellID')['CellCircularity'].diff().abs()
    df_uninf['CircularityChangeAbs'] = df_uninf['CircularityChangeAbs'].fillna(0)
    df_uninf['CircularityChange'] = df_uninf.groupby('CellID')['CellCircularity'].diff()
    df_uninf['CircularityChange'] = df_uninf['CircularityChange'].fillna(0)
    
    cell_properties_uninfected_protrusions[line] = df_uninf
#%%
# 1) Sort so "previous timepoint" is correct within each id
df = df.sort_values(['id', 'timepoint']).copy()

# 2) Absolute change from the previous timepoint within each id
df['change'] = df.groupby('id')['measure'].diff().abs()

# 3) New dataframe with average change per id
avg_change_df = (
    df.groupby('id', as_index=False)['change']
      .mean()
      .rename(columns={'change': 'avg_change'})
)


#%% plot E. coli - normalized to group time 0 

def plot_single_group_with_band(lines,
                          cell_properties,
                          feature_plot,
                          band="range",          # "range" or "sem"
                          percentile=(10,90),   # future-proofing if you want percentiles
                          cmap_map=None,
                          figsize=(8,5),
                          legend_outside=True,
                          labels = True, 
                          normalize = True
                         ):
    """
    band: "range" (min->max) or "sem" (mean +/- sem)
    cmap_map: dict mapping line -> color (e.g., constants.CMAP). If None, matplotlib default cycle used.
    """
    plt.figure(figsize=figsize)
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_line(agg, label, linestyle, color):
        plt.plot(agg['Timepoint_min'], agg['mean_pct'], linewidth=1.5, linestyle=linestyle, label=label, color=color)
        if band == "range":
            lower = agg['min_pct']
            upper = agg['max_pct']
        elif band == "sem":
            lower = agg['mean_pct'] - agg['sem_pct']
            upper = agg['mean_pct'] + agg['sem_pct']
        else:
            raise ValueError("band must be 'range' or 'sem'")

        plt.fill_between(agg['Timepoint_min'], lower, upper, alpha=0.2, color=color, linewidth=0)

    # Infected (solid)
    offset = len(lines)
    for j, line in enumerate(lines):
        df_line = cell_properties[line]
        agg = agg_pct_change(df_line, value_col=feature_plot, time_col="Timepoint_min", normalize = normalize)
        if cmap_map and line in cmap_map:
            color = cmap_map[line]
        else:
            color = prop_cycle[(j + offset) % len(prop_cycle)]
        plot_line(agg, f"{line} (infected)", linestyle='-', color=color)
    
    if labels:
        plt.xlabel("Timepoint (min)")
        plt.ylabel(f"Percent change in {feature_plot} from baseline (%)")
        plt.title(f"Percent change from baseline ({feature_plot}) — per-cell normalized, then averaged")
    else:
        plt.xlabel("")
        plt.ylabel("")
        plt.title("")

    if legend_outside:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        plt.legend()

    plt.show()

feature_plot = "NewEcoli_count"
plot_single_group_with_band(lines_infected, cell_properties_infected_ecoli,
                       feature_plot, band="sem", cmap_map=constants.CMAP, labels = False, figsize = (8,5), normalize = False)

#%% plot individual cell E.coli number 

feature_plot = "Ecoli_Num"
cell_list = list(np.unique(cell_properties_df["CellID"]))

for cell_to_plot in cell_list:
    
    cell_df = cell_properties_df[cell_properties_df["CellID"] == cell_to_plot].sort_values("Timepoint_min")
    
    plt.figure()
    plt.plot(cell_df["Timepoint_min"], cell_df[feature_plot], marker="o", color = constants.CMAP[line])
    
    plt.xlabel("Timepoint (min)")
    plt.ylabel(f"{feature_plot}")
    plt.title(f"{feature_plot} over Time – {cell_to_plot}")
    plt.show()

#%% FOV ecoli+ cells over time
label = False
time_feature_list = list(timepoint_measure.keys())
feature_plot = "Ecoli+_proportion"

plt.figure()

for line in lines_infected:  
    plt.plot(timepoint_properties_infected[line]["Timepoint_min"], timepoint_properties_infected[line][feature_plot], color = constants.CMAP[line])

if label:
    plt.ylim(bottom=0, top=100)
    plt.xlabel("Timepoint (min)")
    plt.ylabel(f"{feature_plot} cells")
    plt.title(f"{feature_plot} cells over Time")
else:
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")

plt.legend(lines_infected)
plt.show()

#%% plot overlap
line = "W1118"
t = "012"
lab_idx = 10

fl_PXN_infected = utils.load_path_into_dict(parentdir / "ilastik_segmentation_input" / "timepoint_z_PXN", keywordregex=[r"t\d{3}"], keyword="PXN")
fl_ecoli_infected = utils.load_path_into_dict(parentdir / "ilastik_segmentation_input" / "timepoint_z_ecoli", keywordregex=[r"t\d{3}"], keyword="ecoli")

labeled_A = macrophage_labeled_infected[line][f"t{t}"]  # alias
original_fl_PXN = tiff.imread(fl_PXN_infected[f"t{t}"])
original_fl_ecoli = tiff.imread(fl_ecoli_infected[f"t{t}"])

# 2D projection: True where any slice had an object
filtered_ecoli = ecoli_binary_infected[line][f"t{t}"]
B_proj = np.any(filtered_ecoli != 0, axis=0)  # shape (H, W), dtype=bool # ecoli segmentation
C_proj = np.mean(original_fl_PXN, axis = 0) # mean intensity PXN
D_proj = np.mean(original_fl_ecoli, axis = 0) # mean intensity E.coli

from matplotlib.colors import LinearSegmentedColormap

magenta_cmap = LinearSegmentedColormap.from_list(
    "magenta_map", [(0, 0, 0), (1, 0, 1)]
)

cyan_cmap = LinearSegmentedColormap.from_list(
    "cyan_map",
    [(0, 0, 0), (0, 1, 1)]  # black → cyan
)

# Build regionprops for A to get bboxes and centroids
regions = {r.label: r for r in regionprops(labeled_A)}
labelB = {}
# show results — one figure per matched label

if lab_idx is None:
    ecoli_cells_plot = ecoli_cells[line]
else:
    ecoli_cells_plot = [lab_idx]

for a_lab in ecoli_cells_plot:
    
    if a_lab not in regions:
        continue  # defensive

    r = regions[a_lab]
    minr, minc, maxr, maxc = r.bbox  # (min_row, min_col, max_row, max_col)

    # crop A mask and B projection to bbox (add a small pad for context)
    pad = 5
    minr_p = max(minr - pad, 0)
    minc_p = max(minc - pad, 0)
    maxr_p = min(maxr + pad, labeled_A.shape[0])
    maxc_p = min(maxc + pad, labeled_A.shape[1])

    A_crop = (labeled_A[minr_p:maxr_p, minc_p:maxc_p] == a_lab)
    B_crop = B_proj[minr_p:maxr_p, minc_p:maxc_p]
    C_crop = C_proj[minr_p:maxr_p, minc_p:maxc_p]
    D_crop = D_proj[minr_p:maxr_p, minc_p:maxc_p]
    
    labelB[a_lab] = label(B_crop)
    # overlap within the crop
    overlap_crop = A_crop & B_crop

    # Plot: left = A mask, right = A mask with overlap overlay (red)
    fig, axes = plt.subplots(2, 3, figsize=(8, 4))
    axes = axes.ravel()

    ax = axes[0]
    ax.imshow(A_crop, cmap="gray")
    ax.set_title(f"Macrophage segmentation - {a_lab}")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(A_crop, cmap="gray")
    # overlay overlaps in red (use alpha)
    # create an RGBA red overlay where overlap is True
    red_overlay = np.zeros((A_crop.shape[0], A_crop.shape[1], 4), dtype=float)
    red_overlay[..., 0] = 1.0   # R
    red_overlay[..., 3] = 0.6 * overlap_crop.astype(float)  # alpha where overlap
    ax.imshow(red_overlay)
    ax.set_title("Internal E. coli")
    ax.axis("off")
    
    ax = axes[2]
    ax.imshow(C_crop, cmap = cyan_cmap)
    ax.set_title("PXN signal")
    ax.axis("off")
    
    ax = axes[3]
    ax.imshow(D_crop, cmap = magenta_cmap)
    ax.set_title("E. coli signal")
    ax.axis("off")
    
    ax = axes[4]
    ax.imshow(C_crop, cmap = cyan_cmap)
    ax.imshow(D_crop, cmap = magenta_cmap, alpha = 0.5)
    ax.set_title("Merge")
    ax.axis("off")

    plt.tight_layout()
    plt.show()

#%% save cropped image

A = A_crop.astype(float)
A = (A - A.min()) / (A.max() - A.min())

# Convert grayscale to RGB
rgb = np.stack([A, A, A], axis=-1)

# Add red overlay where overlap is True
alpha = 0.6
mask = overlap_crop.astype(bool)

rgb[mask, 0] = (1 - alpha) * rgb[mask, 0] + alpha * 1.0  # Red channel
rgb[mask, 1] = (1 - alpha) * rgb[mask, 1]
rgb[mask, 2] = (1 - alpha) * rgb[mask, 2]

# Convert to uint8 for saving
rgb_uint8 = (rgb * 255).astype(np.uint8)
rgb_uint16 = (rgb * 65535).astype(np.uint16)

tiff.imwrite(figdir / "internal_ecoli.tiff", rgb_uint8)

tiff.imwrite(figdir / "macrophage_segmentation.tiff", A_crop)

D_norm = D_crop - D_crop.min()
D_norm = D_norm / D_norm.max()
D_uint16 = (D_norm * 65535).astype(np.uint16)

C_norm = C_crop - C_crop.min()
C_norm = C_norm / C_norm.max()
C_uint16 = (C_norm * 65535).astype(np.uint16)

tiff.imwrite(figdir / "PXN_signal.tiff", C_uint16)
tiff.imwrite(figdir / "ecoli_signal.tiff", D_uint16)


#%% histogram - compare distribution of internalization rate

#%% violin plots - compare between lines
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.patches as mpatches

# ---------------- settings ----------------
comparison_col = "Line"
measure_plot = "CircChange_avg"
# ------------------------------------------

# expand out  (only works if column is a list)
'''
if isinstance(lifetime_measures_all["W1118"][measure_plot].iloc[0], list):
    
    lifetime_measures_all_expanded = pd.concat(
        df.assign(Line=key).explode(measure_plot)
        for key, df in lifetime_measures_all.items()
    ).reset_index(drop=True)
    
    df_clean = lifetime_measures_all_expanded.copy()

    
elif isinstance(lifetime_measures_all["W1118"][measure_plot].iloc[0], dict):
    df_clean = pd.concat(cell_properties_infected_ecoli_full, names=['Line']).reset_index(level=0)

else:
    df_clean = pd.concat(lifetime_measures_all, names=['Line']).reset_index(level=0)
'''

df_clean = pd.concat(morphodynamic_speed_inf, names=['Line']).reset_index(level=0)
ylimit = np.max(df_clean[measure_plot])*1.5
# Prepare data
#df_clean = utils.remove_outliers_iqr(lifetime_measures_all_expanded, [comparison_col], measure).copy()

lines = lines_infected

# Generate color palette per line
palette = dict(zip(lines, sns.color_palette("colorblind", n_colors=len(lines))))
palette = constants.CMAP

pairs, pvals_corr = utils.calculate_pvals(df_clean, measure_plot, comparison_col, stat_test = "Welch")

# ---- Plot ----
plt.figure(figsize=(max(8, len(lines)*1.2), 6))
ax = sns.violinplot(
    data=df_clean,
    x=comparison_col,
    y=measure_plot,
    order=lines,
    palette=palette,
    cut=0
)

sns.stripplot(
    data=df_clean,
    x=comparison_col,
    y=measure_plot,
    order=lines,
    color="k",
    size=3,
    jitter=True,
    alpha=0.6
)

# Annotate comparisons
annot = Annotator(ax, pairs, data=df_clean, x=comparison_col, y=measure_plot, order=lines)
annot.configure(text_format="star", loc="outside")
annot.set_pvalues_and_annotate(list(pvals_corr))

# ---- Custom legend showing Lines ----
legend_handles = [mpatches.Patch(color=palette[line], label=line) for line in lines]
ax.legend(handles=legend_handles, title="Line", bbox_to_anchor=(1.02, 1), loc="upper left")

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(top = ylimit)
#ax.set_title(f"{measure_plot} at Time {tp_label} — Between-Line Comparisons")

plt.tight_layout()
plt.show()

#%% table summary
# combine into single dataframe
table_measures = ["RateNewEcoli_perMin"]
lifetime_measures_all_table = df_clean.copy()

lifetime_measures_all_table["Line"] = pd.Categorical(
    lifetime_measures_all_table["Line"],
    categories=lines_infected,
    ordered=True
)

avg_per_line = (
    lifetime_measures_all_table
    .groupby("Line", sort = False)[table_measures]
    .mean()
    .mul(60)
    .reset_index()
).rename(columns = {"RateNewEcoli_perMin": "Internalization Rate (per hour)"})

avg_per_line["Internalization Rate (per hour)"] = (
    avg_per_line["Internalization Rate (per hour)"].round(1)
)

utils.create_table(avg_per_line, body_fontsize=8, header_fontsize=8, fig_size=(4.5,1.25))

#%% protrusions

