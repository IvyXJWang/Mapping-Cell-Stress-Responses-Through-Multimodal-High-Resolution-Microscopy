import segmentation_utils as utils
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from skan import csr  # requires numpy 2.2 or less
import networkx as nx

from pathlib import Path
from joblib import Parallel, delayed
import cv2
from skimage.measure import find_contours
from roifile import ImagejRoi
from tqdm import tqdm
from skimage import draw
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.morphology import reconstruction
import tifffile as tiff
from feature_extraction_utils import collapse_multilayer


def shape_complexity(binary_image, solidity_threshold=0.9):
    """
    Input
        binary_image (2D array): binary image of shapes to be separated by complexity
        solidity (float): solidity threshold (0-1)
    Output
        shapes_complex (2D array): binary image of only the simple shapes (high solidity)
        shapes_simple (2D array): binary image of only the complex shapes (low solidity)
    """
    # Label objects
    labeled = label(binary_image)
    roi_properties = regionprops(labeled)

    shapes_simple = np.zeros_like(binary_image, dtype=bool)
    shapes_complex = np.zeros_like(binary_image, dtype=bool)

    for roi in roi_properties:
        solidity = roi.solidity
        if solidity >= solidity_threshold:
            shapes_simple[labeled == roi.label] = True
        else:
            shapes_complex[labeled == roi.label] = True

    return shapes_complex, shapes_simple


def grow_branch(
    branch, imgBin, imgBorders, count, max_radius=200, method="dilate", DEBUG = False,
):
    distance_map = ndi.distance_transform_edt(imgBin)
    branch_mask = np.zeros_like(imgBin, dtype=bool) 
    coords = np.array(branch)
    branch_mask[coords[:, 0], coords[:, 1]] = True

    if method == "dilate":
        radii = [distance_map[y, x] for y, x in branch]
        radius = int(np.round(np.mean(radii) *1.5)) # * 1.25))

        if radius > max_radius:
            print(f"Skipping branch {count} - too large")
            return None

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        branch_uint8 = (branch_mask * 255).astype(np.uint8)
        dilated = cv2.dilate(branch_uint8, kernel)
        grown = (dilated > 0) & imgBorders

        if DEBUG:
            return grown, imgBorders

    elif method == "reconstruct":
        # should only have a single marker (single branch input)
        branch_mask = branch_mask.astype(np.uint8)
        marker = branch_mask.copy()
        distance = distance_transform_edt(imgBin == 0)
        watershed_label = watershed(
            -distance, markers=marker, mask=imgBorders.astype(bool)
        )

        recovered = np.zeros_like(imgBin, dtype=np.uint8)
        rec = reconstruction(marker, watershed_label, method="dilation")
        recovered = np.logical_or(recovered, rec > 0)

        grown = recovered.astype(np.uint8)

    return grown, count


def branch_graph(skeletonGraph):
    # Build branch graph manually
    endpoints = {}  # maps endpoint coords to branch indices
    G = nx.Graph()

    for i in range(len(skeletonGraph.paths_list())):
        coords = np.array(skeletonGraph.path_coordinates(i))
        if coords.size == 0:
            continue
        start = tuple(coords[0])
        end = tuple(coords[-1])

        G.add_node(i)  # each branch is a node

        for pt in [start, end]:
            key = tuple(np.round(pt).astype(int))
            if key in endpoints:
                for j in endpoints[key]:
                    G.add_edge(i, j)
                endpoints[key].append(i)
            else:
                endpoints[key] = [i]

    return G


def merge_linear_branch_chains(G):
    visited = set()
    merged_chains = []

    for node in G.nodes:
        if node in visited or G.degree[node] != 1:  # start from endpoints
            continue

        chain = [node]
        visited.add(node)
        current = node
        prev = None

        while True:
            neighbors = [
                n for n in G.neighbors(current) if n != prev and n not in visited
            ]
            if len(neighbors) != 1:
                break  # junction or dead end

            next_node = neighbors[0]
            if G.degree[next_node] != 2:
                chain.append(next_node)
                visited.add(next_node)
                break

            chain.append(next_node)
            visited.add(next_node)
            prev = current
            current = next_node

        merged_chains.append(chain)

    return merged_chains


def split_mito(
    imgBin, imgBorders=None, minbranchLength=0, merge_branches=True, DEBUG=False
):
    """
    Input
        imgBin (2D array): binary image of mitochondria to be split
        imgBorders (2D array): binary image of borders to be used for growing branches
        minbranchLength (int): minimum length of branches to be kept
    Output
        recovered_shapes (list): list of recovered shapes as binary masks
        split_rois (list): list of ImagejRoi objects for each recovered shape
    """
    if (
        imgBorders is None
    ):  # if no border image input use original binary image as border image
        imgBorders = imgBin

    skeleton = skeletonize(imgBin)

    # Create a Skan Skeleton graph
    skeleton_graph = csr.Skeleton(skeleton)

    # verify coverage of skeleton graph
    # Number of branches (paths)
    n_paths = len(skeleton_graph.paths_list())

    # Get coordinates of all branches
    branches_coords = [skeleton_graph.path_coordinates(i) for i in range(n_paths)]

    # Initialize an empty mask to store all branch pixels
    branch_mask = np.zeros_like(skeleton, dtype=bool)

    # Fill in the mask with pixels from each branch
    for coords in branches_coords:
        # Convert coordinates to integer row/col
        y = coords[:, 0].astype(int)
        x = coords[:, 1].astype(int)
        branch_mask[y, x] = True
    

    # Calculate coverage
    # total_skeleton_pixels = np.sum(skeleton)
    # covered_pixels = np.sum(branch_mask)
    # coverage = covered_pixels / total_skeleton_pixels * 100

    # print(f"Skeleton pixel coverage by branches: {coverage:.2f}%")

    if merge_branches:
        # Merge branches
        G = branch_graph(skeleton_graph)
        merged_chains = merge_linear_branch_chains(G)
        # print(f"Merged {len(merged_chains)} chains from {len(G.nodes)} branches")

        merged_coords_list = []

        for chain in merged_chains:
            all_coords = []
            for branch_idx in chain:
                coords = skeleton_graph.path_coordinates(branch_idx)
                all_coords.extend(coords)
            merged_coords_list.append(np.array(all_coords))

        if len(merged_coords_list) == 0:
            merged_coords_list = branches_coords

    else:
        merged_coords_list = branches_coords

    if minbranchLength > 0:
        branches_coords_final = [
            coords for coords in merged_coords_list if len(coords) >= minbranchLength
        ]

    else:
        branches_coords_final = merged_coords_list

    #print(len(merged_coords_list))

    recovered_shapes = []
    branches_coords_final_processing = branches_coords_final
    
    if DEBUG:
        return branches_coords_final_processing, n_paths
    
    #if DEBUG:
        #branches_coords_final_processing = branches_coords_final_processing[
          #  :5
       # ]  # do not dilate every branch if debugging (bottleneck)

    # start_total = time.time()

    # Parallel execution
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(grow_branch)(
            branch, imgBin, imgBorders, count, method="dilate"
        )
        for count, branch in enumerate(branches_coords_final_processing, 1)
    )

    # Filter out skipped branches
    recovered_shapes = [(mask, idx) for mask, idx in results if mask is not None]


    print(f"\n{len(recovered_shapes)} overlapping mitochondria separated")

    split_rois = []
    for mask, i in recovered_shapes:
        contours = find_contours(mask.astype(float), level=0.5)
        if not contours:
            continue

        # Use the largest contour
        contour = max(contours, key=len)
        contour = np.round(contour).astype(np.int16)  # (y, x)
        xy = contour[:, ::-1]  # Convert to (x, y)

        roi = ImagejRoi.frompoints(xy)
        roi.name = f"region_{i}"

        split_rois.append(roi)

    return recovered_shapes, split_rois


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


def is_tiff(file: Path) -> bool:
    return file.is_file() and (file.suffix == ".tif" or file.suffix == ".tiff")


def mitochondria_processing(
    cellid: str,
    probmap_path: Path,
    cell_path: Path,
    padding=50,
    saveintermediate=False,
    outputdir=None,
    DEBUG=False,
    threshold=(60, 90),
    internal_first=True,
    local_threshold=True,
    empty_threshold = 0.01,
):
    if outputdir is None and saveintermediate != False:
        outputdir = probmap_path
        print("Saving intermediate images to probability map directory")

    probmap = np.pad(
        tiff.imread(probmap_path), pad_width=padding, mode="constant", constant_values=0
    )
    probmap_filtered = np.where(probmap < empty_threshold, 0, probmap)
    probmap_objectvals = probmap_filtered[probmap_filtered > 0].tolist()

    binary_cell = (
        np.pad(
            tiff.imread(cell_path),
            pad_width=padding,
            mode="constant",
            constant_values=0,
        )
        > 0
    )

    # Get image shape from binary cell
    imgshape = binary_cell.shape

    if type(threshold) == float and threshold > 0 and threshold < 1:
        threshold_list = [threshold]
    elif type(threshold) == list and type(threshold[0]) == float:
        threshold_list = threshold
    elif threshold == "std":
        threshold_list = [
            np.mean(probmap_objectvals) + 0.5 * np.std(probmap_objectvals)
        ]
    elif isinstance(threshold, tuple):
        percent_above_skel, percent_above_fill = threshold
        threshold_skel = np.percentile(probmap_objectvals, 100 - percent_above_skel)
        threshold_fill = np.percentile(probmap_objectvals, 100 - percent_above_fill)
        threshold_list = [threshold_skel, threshold_fill]
    else:
        threshold_list = [0.25]
        print(
            "Using default confidence threshold of 0.25 for mitochondria segmentation"
        )

    # Extract segmentation above confidence threshold
    binary_mito_dict = utils.segmentation_threshold_2D(
        probmap, confidence_thresholds=threshold_list
    )

    threshold_list_label = list(binary_mito_dict.keys())
    # Filter object size (fill)
    binary_mito_size, _ = utils.filter_object_size(
        binary_mito_dict[threshold_list_label[-1]], min_size=2500
    )
    
    # Internal mask (fill)
    if internal_first:
        binary_mito_size_internal, _, _ = utils.keep_internal_rois(
            binary_mito_size,
            binary_cell,
            min_overlap=0.3,
            separate_connecting_iter=0,
            largest_only=False,
        )
    else:
        binary_mito_size_internal = binary_mito_size

    # Smoothen object shapes soft thresholding
    # Split into large and small shapes for different smoothening
    binary_mito_size_internal_small, binary_mito_size_internal_large = utils.split_binary_size(binary_mito_size_internal, area_threshold=7500)
    binary_mito_size_internal_small_smooth = utils.smooth_shape_fft(
        binary_mito_size_internal_small, FD_retained=0.05, morphological_opening=0
    ) # FD_retained percentage of Fourier descriptors retained (lower means more smoothing, 0.01)
    # shape dependent smoothening (small shapes get smoothened away)
    binary_mito_size_internal_large_smooth = utils.smooth_shape_fft(
        binary_mito_size_internal_large, FD_retained=0.01, morphological_opening=0
    )
    binary_mito_size_internal_smooth = np.logical_or(binary_mito_size_internal_small_smooth, binary_mito_size_internal_large_smooth).astype(np.uint8)

    # Separate objects by complexity (skeleton)
    binary_mito_complex, binary_mito_simple = shape_complexity(
        binary_mito_size_internal_smooth, solidity_threshold=0.90
    )
    binary_mito_complex = np.logical_and(
        binary_mito_complex > 0, binary_mito_dict[threshold_list_label[0]] > 0
    ).astype(np.uint8)

    # ****Add watershed separation to shape_complexity function to separate and isolate round mito
    
    if len(threshold_list) > 1:
        skel_mask = binary_mito_complex
        fill_mask = binary_mito_size
    else:
        skel_mask = binary_mito_complex
        fill_mask = skel_mask

    if DEBUG:
        utils.save_tiff(
            skel_mask, outputdir, f"{cellid}_mito_skelmask_{round(threshold_list[0], 5)}"
        )

        if len(threshold_list) > 1:
            utils.save_tiff(
            fill_mask, outputdir, f"{cellid}_mito_fillmask_{round(threshold_list[-1], 5)}"
        )
            print(f"Skeleton threshold: {threshold_list[0]}, Fill threshold: {threshold_list[-1]}")
        
        return skel_mask, fill_mask
    
    
    if skel_mask.sum() == 0:
        print(f"No complex mitochondria detected for {cellid}")
        next_roi_idx = 1
        split_rois_complex = []
    else: 
        # Separate overlapping complex mito in complex mito population
        recovered_shapes, split_rois_complex = split_mito(
            skel_mask, imgBorders=fill_mask, minbranchLength=50, merge_branches = False
        )
        next_roi_idx = len(split_rois_complex) + 1
    
    if DEBUG:
        return recovered_shapes

    # Convert simple mito population into rois
    split_rois_simple = utils.binary_to_rois(
        binary_mito_simple, region_idx_start=next_roi_idx
    )

    # Combined all rois together + save as .zip
    all_rois = split_rois_complex + split_rois_simple

    # Separate all rois into multiple binary layers
    mito_layers = overlapping_to_layers(all_rois, imgshape)

    # Keep only internal mitochondria
    binary_mito_internal_list = []
    for layer_split in mito_layers:
        mito_layer_internal, _, _ = utils.keep_internal_rois(
            layer_split,
            binary_cell,
            min_overlap=0.7,
            largest_only=False,
        )
        binary_mito_internal_list.append(mito_layer_internal)

    # Smoothen internal mitochondria - incorporate size filter (prevent oversmoothening)
    mito_smoothened_list = []
    for layer_internal in binary_mito_internal_list:
        mito_layer_smooth = utils.smooth_shape_fft(
            layer_internal, FD_retained=0.05, morphological_opening=0
        )
        mito_smoothened_list.append(mito_layer_smooth)

    # Filter object size after separation
    binary_mito_final_list = []
    for layer_smooth in mito_smoothened_list:
        mito_layer_final, _ = utils.filter_object_size(layer_smooth, min_size=5000)
        binary_mito_final_list.append(mito_layer_final)

    binary_mito_labelled_list = []
    for layer_labelled in binary_mito_final_list:
        mito_layer_labelled = label(layer_labelled)
        binary_mito_labelled_list.append(mito_layer_labelled)

    if saveintermediate:
        # utils.save_tiff(binary_nucleus[cellid], outputdir, f"{cellid}_nucleus_thresholded_py")# thresholded probability map nucleus
        utils.save_tiff(
            binary_mito_internal_list[0], outputdir, f"{cellid}_mito_noexternal"
        )  # external mitochondria removed
        # utils.save_tiff(
        # binary_mito_size, outputdir, f"{cellid}_mito_sizefiltered"
        # )  # small artifacts removed
        utils.save_tiff(
            mito_smoothened_list[0], outputdir, f"{cellid}_mito_smooth"
        )  # smoothened
        count = 0
        for layer_split in mito_layers:
            utils.save_tiff(
                layer_split, outputdir, f"{cellid}_mito_separated_layer{count}"
            )  # overlapping separated
            count += 1

    if DEBUG:
        import matplotlib.pyplot as plt

        flattened = collapse_multilayer(binary_mito_labelled_list)
        mito_fill_skel = skeletonize(binary_mito_dict[threshold_list_label[-1]])
        mito_skel_skel = skeletonize(binary_mito_dict[threshold_list_label[0]])

        variables = {
            "Mitochondria probability map": probmap,
            "Mitochondria boundary threshold": binary_mito_dict[
                threshold_list_label[-1]
            ],
            "Mitochondria skeleton threshold": binary_mito_dict[
                threshold_list_label[0]
            ],
            "Mitochondria size threshold": binary_mito_size,
            "Internal mitochondria": binary_mito_size_internal,
            "Complex mitochondria": binary_mito_complex,
            "Simple mitochondria": binary_mito_simple,
            # "recovered mitochondria": recovered_shapes[0][0],
            "All recovered mitochondria combined": flattened,
        }
        for t, v in variables.items():
            plt.imshow(v, cmap="gray")
            plt.title(
                # f" skel thresh {percent_above_skel} : fill thresh {percent_above_fill} {cellid} {t}"
                f"{t}"
            )
            plt.axis("off")
            plt.show()

        return binary_mito_complex

    # Label objects
    return {cellid: binary_mito_labelled_list}


def postprocess_mitochondria(
    organelle_paths: dict, padding=50, savedir="", DEBUG=False, threshold=(60,90)
):
    probmap_dict = organelle_paths["mitochondria"]
    cell_processed = organelle_paths["cell"]

    results = []
    all_CellID = list(cell_processed.keys())
    for cellid in tqdm(all_CellID, desc="Processing", total=len(all_CellID)):
        results.append(
            mitochondria_processing(
                cellid,
                probmap_dict[cellid],
                cell_processed[cellid],
                saveintermediate="CELL012",
                outputdir=savedir,
                padding=padding,
                DEBUG=DEBUG,
                threshold=threshold,
                empty_threshold=0.01,
            )
        )

    if DEBUG:
        return results

    # results: [{key:value},{key:value}]
    binary_mitochondria_labelled = {k: v for d in results for k, v in d.items()}

    if savedir != "":
        # save mitochondria images (layer by layer)
        for cell in binary_mitochondria_labelled.keys():
            for i, layer in enumerate(binary_mitochondria_labelled[cell]):
                mito_layer = f"{cell}_layer{i}"
                filename = mito_layer + "_mitochondria_labelled.tif"
                if (
                    layer.max() < 255
                ):  # save as smallest file type while preserving labels
                    tiff.imwrite(savedir / filename, layer.astype(np.uint8))
                else:
                    tiff.imwrite(savedir / filename, layer.astype(np.uint16))
        print("\nProcessed mitochondria segmentation saved as tiff")
        return

    else:
        return binary_mitochondria_labelled


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
            if inner_k in ["CELL046"]
        }
        for outer_k, outer_v in allpaths.items()
    }

    allpaths_processed = {
        outer_k: {
            inner_k: [inner_v]
            for inner_k, inner_v in outer_v.items()
            if inner_k in ["CELL046"]
        }
        for outer_k, outer_v in allpaths_processed.items()
    }

    binary_mitochondria_labelled = postprocess_mitochondria(
        allpaths, allpaths_processed, padding=50, savedir=thesis_figs
    )
