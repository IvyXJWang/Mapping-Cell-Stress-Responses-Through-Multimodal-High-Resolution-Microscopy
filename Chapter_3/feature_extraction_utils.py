import pandas as pd

pd.DataFrame.iteritems = pd.DataFrame.items  # iteritems removed in pandas version 2.0

from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.draw import polygon
import math
from shapely.geometry import Polygon, MultiPolygon
from scipy.stats import skew
from pointpats import random
import geopandas as gpd
from skimage import measure
from shapely import minimum_bounding_circle
import cv2
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from shapely.plotting import plot_polygon
from matplotlib.patches import Wedge
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt

# project modules
import constants
import segmentation_utils as utils


def load_segmentation(roidir: Path, organelle=[]):
    all_segmentation = {}

    # read in binary maps
    if "debris" in organelle or organelle == "all":
        all_segmentation["debris"] = utils.load_tiff_into_dict(
            roidir, [r"CELL\d{3}", "debris"]
        )

    if "au" in organelle or organelle == "all":
        all_segmentation["au"] = utils.load_tiff_into_dict(roidir, [r"CELL\d{3}", "au"])

    if "cell" in organelle or organelle == "all":
        all_segmentation["cell"] = utils.load_tiff_into_dict(
            roidir, [r"CELL\d{3}", "cell"]
        )

    if "nucleus" in organelle or organelle == "all":
        all_segmentation["nucleus"] = utils.load_tiff_into_dict(
            roidir, [r"CELL\d{3}", "nucleus"]
        )

    if "lipiddroplets" in organelle or organelle == "all":
        all_segmentation["lipiddroplets"] = utils.load_tiff_into_dict(
            roidir, [r"CELL\d{3}", "lipiddroplet"]
        )

    if "cytoplasm" in organelle or organelle == "all":
        all_segmentation["cytoplasm"] = utils.load_tiff_into_dict(
            roidir, [r"CELL\d{3}", "cytoplasm"]
        )

    if "mitochondria" in organelle or organelle == "all":
        mitochondria_segmentation_unsorted = pd.DataFrame()
        layer = 0
        while True:
            layer_string = str(layer)
            pattern = [r"CELL\d{3}", f"_layer{layer_string}", "mitochondria"]
            image_dict = utils.load_tiff_into_dict(roidir, pattern)

            if len(image_dict) == 0:
                break

            mitochondria_segmentation_unsorted_dict = {
                "layer": layer,
                "Metadata_CellID": image_dict.keys(),
                "image": image_dict.values(),
            }
            mitochondria_segmentation_unsorted_df = pd.DataFrame(
                mitochondria_segmentation_unsorted_dict
            )

            mitochondria_segmentation_unsorted = pd.concat(
                [
                    mitochondria_segmentation_unsorted,
                    mitochondria_segmentation_unsorted_df,
                ],
                ignore_index=True,
            )

            layer += 1

        # add image number to segmentation dataframe
        image_numbers = list(range(1, len(mitochondria_segmentation_unsorted) + 1))
        mitochondria_segmentation = mitochondria_segmentation_unsorted.sort_values(
            by=["Metadata_CellID", "layer"]
        )
        mitochondria_segmentation["ImageNumber"] = image_numbers
        all_segmentation["mitochondria"] = mitochondria_segmentation

    return all_segmentation


def ImageNumber_to_CellID_dict(allfiles_dict):
    # Mapping ImageNumber to CellID
    image_sheet = allfiles_dict["Image"]
    image_to_cellid = {
        k: v for k, v in zip(image_sheet["ImageNumber"], image_sheet["Metadata_CellID"])
    }

    return image_to_cellid


def map_columns(allfiles_dict, col1, col2):
    # Mapping col1 to col2
    image_sheet = allfiles_dict["Image"]
    col1_to_col2 = {k: v for k, v in zip(image_sheet[col1], image_sheet[col2])}

    return col1_to_col2


def ImageNumber_to_imagesize_dict(allfiles_dict):
    # Mapping ImageNumber to imagesize
    imagesize_df = pd.DataFrame()

    image_sheet = allfiles_dict["Image"]
    image_to_size = {
        k: v
        for k, v in zip(
            image_sheet["ImageNumber"],
            zip(image_sheet["Height_xray"], image_sheet["Width_xray"]),
        )
    }

    imagesize_df["ImageNumber"] = image_to_size.keys()
    imagesize_df["ImageSize"] = image_to_size.values()

    return imagesize_df


def map_to_image_number(df_dict, mapdf):
    # map column to dataframe based on image number
    df_mapped = {}

    for sheet_name, df in df_dict.items():
        if "ImageNumber" in df.columns:
            df["Metadata_CellID"] = df["ImageNumber"].map(mapdf)

            df_mapped[sheet_name] = df

    return df_mapped


def convexhull_props_allroi(rois_dict):
    """
    Calculates convex hull area and perimeter for each ROI and returns a DataFrame.

    Parameters:
        rois_dict (dict): Dictionary of ROIs as ImagejRoi objects

    Returns:
        pd.DataFrame: With columns ['roi_name', 'hull_area', 'hull_perimeter']
    """

    results = []

    for name, roi in rois_dict.items():
        coords = roi.coordinates()

        if len(coords) < 3:
            # Convex hull needs at least 3 points
            results.append(
                {"roi_name": name, "hull_area": np.nan, "hull_perimeter": np.nan}
            )
            continue

        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices]

        # Compute perimeter (Euclidean length around the hull)
        diffs = np.diff(np.vstack([hull_coords, hull_coords[0]]), axis=0)
        perimeter = np.sum(np.linalg.norm(diffs, axis=1))

        results.append(
            {
                "roi_name": name,
                "hull_area": hull.volume,  # area in 2D
                "hull_perimeter": perimeter,
            }
        )

    return pd.DataFrame(results)


def circularity(area, perimeter):
    import math

    circularity = (4 * math.pi * area) / (perimeter**2)

    return circularity


def convexhull_props_singleroi(single_roi):
    """
    Calculates convex hull area and perimeter for each ROI and returns a DataFrame.

    Parameters:
        rois_dict (dict): Dictionary of ROIs as ImagejRoi objects

    Returns:
        pd.DataFrame: With columns ['roi_name', 'hull_area', 'hull_perimeter']
    """

    results = []
    coords = np.column_stack(np.nonzero(single_roi))

    if len(coords) < 3:
        # Convex hull needs at least 3 points
        results.append({"hull_area": np.nan, "hull_perimeter": np.nan})

    hull = ConvexHull(coords)
    hull_coords = coords[hull.vertices]

    # Compute perimeter (Euclidean length around the hull)
    diffs = np.diff(np.vstack([hull_coords, hull_coords[0]]), axis=0)
    perimeter = np.sum(np.linalg.norm(diffs, axis=1))

    results.append(
        {
            "hull_area": hull.volume,  # area in 2D
            "hull_perimeter": perimeter,
            "hull_circularity": circularity(hull.volume, perimeter),
        }
    )

    return pd.DataFrame(results)


def largest_distance_between_two_points(single_roi, plot=False):
    results = []
    coords = np.column_stack(np.nonzero(single_roi))

    dist_matrix = cdist(coords, coords, metric="euclidean")  # pair-wise distances
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    pt1, pt2 = coords[i], coords[j]
    max_dist = dist_matrix[i, j]

    results.append(
        {
            "maximum_extension": max_dist,
            "maximum_extension_points": (pt1, pt2),
        }
    )

    if plot:
        plt.figure(figsize=(6, 6))
        plt.plot(coords[:, 1], coords[:, 0], "ko-", label="Contour")
        plt.plot(
            [pt1[1], pt2[1]],
            [pt1[0], pt2[0]],
            "r-",
            linewidth=2,
            label=f"Max Dist = {max_dist:.2f}",
        )
        plt.plot(pt1[1], pt1[0], "ro", label="Point 1")
        plt.plot(pt2[1], pt2[0], "bo", label="Point 2")
        plt.gca().invert_yaxis()  # Flip y-axis for image-like display
        plt.axis("equal")
        plt.legend()
        plt.title("Farthest Points on ROI")
        plt.show()

    return pd.DataFrame(results)


def drop_duplicate_columns_df(dataframe1, dataframe2, keep_cols):
    # remove duplicate columns in new dataframe

    dataframe_filtered = dataframe1.drop(
        columns=[
            col
            for col in dataframe1.columns
            if col in dataframe2.columns and col not in keep_cols
        ]
    )

    return dataframe_filtered


def roi_to_mask(shape, x, y):
    """Convert traced ROI with absolute coordinates to full-size binary mask."""
    mask = np.zeros(shape, dtype=bool)
    rr, cc = polygon(y, x, shape)  # row = y, col = x
    mask[rr, cc] = True

    return mask


def mask_list(rois_dict, image_shape):
    """Load ROIs from a zip file and return list of masks and ROI names."""

    masks = []
    roi_names = []

    for name, roi in rois_dict.items():
        x = np.array(roi["x"])
        y = np.array(roi["y"])
        mask = roi_to_mask(image_shape, x, y)
        masks.append(mask)
        roi_names.append(name)

    return masks, roi_names


def binary_to_polygon(binary_image, simplify_tolerance=0.0):
    """
    Convert a binary mask into shapely Polygon(s).

    Parameters
    ----------
    binary_image : 2D array (dtype=0/1)
        Binary mask image.
    simplify_tolerance : float
        If >0, simplifies polygon geometry (reduces vertices).

    Returns
    -------
    shapely Polygon or MultiPolygon
    """

    # Find contours at value 0.5 (boundary between 0 and 1)
    contours = measure.find_contours(binary_image, 0.5)

    polygons = []
    for contour in contours:
        # Coordinates are (row, col), so flip to (x, y)
        poly = Polygon(contour[:, ::-1])
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance)
        if poly.is_valid:
            polygons.append(poly)

    if not polygons:
        return None
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return MultiPolygon(polygons)


def farthest_point_from_edge(mask: np.ndarray):
    """
    mask: 2D boolean or 0/1 array where True/1 = inside the object.
    Returns: (x, y, r) where (x,y) is the farthest point, r is distance to edge (pixels).
    """
    mask = mask.astype(bool)

    # Distance to the nearest background pixel (i.e., object edge)
    dist = distance_transform_edt(mask)

    # Index of maximum distance within the object
    y, x = np.unravel_index(np.argmax(dist), dist.shape)

    r = dist[y, x]

    return x, y, r


def distance_pt(point, center):
    """
    Calculate distance between a point and a center.

    Parameters
    ----------
    point  : list, two numbers [x,y]
    center : list, two numbers [x,y]

    Returns
    -------
    _      : float
             distance.
    """

    dist = math.sqrt(
        math.pow(point[0] - center[0], 2) + math.pow(point[1] - center[1], 2)
    )

    return dist


def std_distance_pt(points, center, show_intermediate=False, center_roi=None):
    """
    Calculate standard distance of a point array.

    Parameters
    ----------
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.
    center : list, two numbers [x,y]

    Returns
    -------
    _      : float
             standard distance.
    """
    if len(points) == 0:
        return np.nan

    if len(center) != 2:
        return np.nan

    points = np.asarray(points)
    n, p = points.shape

    sum_of_sq_diff_x = 0.0
    sum_of_sq_diff_y = 0.0

    for x, y in points:
        diff_x = math.pow(x - center[0], 2)
        diff_y = math.pow(y - center[1], 2)
        sum_of_sq_diff_x += diff_x
        sum_of_sq_diff_y += diff_y

    sum_of_results = (sum_of_sq_diff_x / n) + (sum_of_sq_diff_y / n)

    if show_intermediate:
        plt.figure(figsize=(6, 6))

        # Show binary image
        if center_roi is not None:
            plt.imshow(center_roi, cmap="gray", origin="upper")

        plt.scatter(
            points[:, 0], points[:, 1], c="lime", s=10, label="Lipid droplet centroid"
        )

        plt.scatter([center[0]], [center[1]], c="blue", s=50, label="Nucleus centroid")

        plt.title("Standard Distance From Nucleus")
        plt.legend(loc="lower right")
        # plt.axis('off')

        plt.show()

    return math.sqrt(sum_of_results)


def std_distance_pt_summary_measures(points, center):
    """
    Calculate variation metrics of distances from each point to a center.

    Parameters
    ----------
    points : arraylike
        (n,2) coordinates of points
    center : list or array
        [x, y] coordinates of center

    Returns
    -------
    dict
        Dictionary containing:
        - distances: array of distances
        - mean_distance: mean of distances
        - std_distance: standard deviation of distances
        - skew_distance: skewness of distances
        - cv_distance: coefficient of variation (std / mean)
    """
    points = np.asarray(points)

    # Compute distances of each point to center
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))

    mean_dist = distances.mean()
    std_dist = distances.std(ddof=0)  # population std
    skew_dist = skew(distances)
    cv_dist = std_dist / mean_dist if mean_dist != 0 else 0.0

    return {
        "distances": distances,
        "mean_distance": mean_dist,  # should be equivalent to standard distance returned by above function
        "std_distance": std_dist,
        "skew_distance": skew_dist,
        "cv_distance": cv_dist,
    }


def median_distance_pt(points, center):
    """
    Compute median absolute deviation of distances from center.

    Parameters
    ----------
    points : arraylike
        (n,2) array of coordinates
    center : arraylike, optional
        Center to compute distances from. If None, use geometric median.

    Returns
    -------
    float
        MAD of distances
    """
    points = np.asarray(points)

    distances = np.linalg.norm(points - center, axis=1)

    return np.median(np.abs(distances - np.median(distances)))


def gini_coefficient(
    values,
):  # Gini coefficient measures inequality in the distances. 0 = all equal, 1 = maximum inequality
    """
    Compute Gini coefficient of an array.

    Parameters
    ----------
    values : arraylike
        Non-negative values (e.g., distances)

    Returns
    -------
    float
        Gini coefficient
    """
    values = np.asarray(values)
    if np.any(values < 0):
        raise ValueError("Gini coefficient requires non-negative values.")

    sorted_vals = np.sort(values)
    n = len(values)
    cumvals = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return gini


def gini_distance(points, center):
    """
    Compute Gini coefficient of distances from center.

    Parameters
    ----------
    points : arraylike
        (n,2) array of coordinates
    center : arraylike, optional
        Center to compute distances from. If None, use geometric median.

    Returns
    -------
    float
        Gini coefficient
    """
    points = np.asarray(points)

    distances = np.linalg.norm(points - center, axis=1)
    return gini_coefficient(distances)


def ellipse_pt(points, center):
    """
    Calculate parameters of standard deviational ellipse for a point pattern.

    Parameters
    ----------
    points : arraylike
             (n,2), (x,y) coordinates of a series of event points.

    center : list, two numbers [x,y]


    Returns
    -------
    _      : float
             semi-major axis.
    _      : float
             semi-minor axis.
    theta  : float
             clockwise rotation angle of the ellipse.

    Notes
    -----
    Implements approach from:

    https://www.icpsr.umich.edu/CrimeStat/files/CrimeStatChapter.4.pdf
    """

    points = np.asarray(points)
    x = points[:, 0] - center[0]
    y = points[:, 1] - center[1]

    # covariance elements
    xss = (x**2).sum()
    yss = (y**2).sum()
    xy = (x * y).sum()

    # orientation
    theta = 0.5 * np.arctan2(2 * xy, xss - yss)

    # rotated coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # semi-axis lengths
    a = np.sqrt(np.sum(x_rot**2) / len(points))  # semi-major
    b = np.sqrt(np.sum(y_rot**2) / len(points))  # semi-minor

    aspect_ratio = a / b

    return a, b, theta, aspect_ratio


def object_polarization(points, center, show_intermediate=False, outline=None):
    points = np.asarray(points)
    center = np.asarray(center)

    # measure polarity
    vectors = points - center

    # Compute angles (in radians) from reference to each point
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles_deg = np.degrees(angles) % 360

    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))

    R = np.sqrt(sin_sum**2 + cos_sum**2)
    R_bar = R / len(angles)

    if R_bar > 0:
        s = np.sqrt(-2 * np.log(R_bar))  #
    else:
        s = np.inf  # Undefined when R_bar = 0

    if show_intermediate:
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        bin_edges = np.deg2rad(np.arange(0, 360 + 10, 10))
        ax.hist(np.deg2rad(angles_deg), bins=bin_edges, edgecolor="black")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.title("Directionality from Nucleus Centroid")

        n_bins = 12
        bin_edges = np.linspace(0, 360, n_bins + 1)
        bin_counts, _ = np.histogram(angles_deg, bins=bin_edges)
        bin_counts = bin_counts / bin_counts.max()  # normalize for display

        # Add cell
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot outline
        if outline is not None:
            plt.imshow(outline, cmap="gray", origin="upper", alpha=1)

        # Plot points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=20,
            color="lime",
            label="Lipid droplet centroid",
        )

        # Plot reference point
        ax.scatter(*center, s=80, color="red", label="Nucleus centroid")

        # Plot rose diagram at reference point
        r_max = 750  # maximum wedge length
        for i in range(n_bins):
            theta1 = bin_edges[i]
            theta2 = bin_edges[i + 1]
            count = bin_counts[i]
            if count > 0:
                wedge = Wedge(
                    center=center,
                    r=r_max * count,
                    theta1=theta1,
                    theta2=theta2,
                    facecolor="steelblue",
                    edgecolor="black",
                    alpha=0.6,
                )
                ax.add_patch(wedge)

        # Adjust plot
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
        ax.set_title("Directionality From Nucleus")
        # plt.grid(True)
        ax.axis("off")
        plt.show()

    return R_bar


def boundary_random_point_dist(boundary_shape, center, number_points, n_sim=999):
    # simulate random distribution of points across cell shape
    R_sim = []
    for _ in range(n_sim):
        rand_points = random.poisson(boundary_shape, size=number_points)
        vectors_sim = rand_points - center
        angles_sim = np.arctan2(vectors_sim[:, 1], vectors_sim[:, 0])
        R = np.sqrt(
            np.sum(np.cos(angles_sim)) ** 2 + np.sum(np.sin(angles_sim)) ** 2
        ) / len(angles_sim)
        R_sim.append(R)

    return np.array(R_sim)


def polarity_binary(p_thresh, R_sim, R_obs):
    p_thresh = p_thresh  # p-value threshold

    p_val = np.mean(
        R_sim >= R_obs
    )  # proportion of simulations with greater polarization than observed polarization

    if p_val <= p_thresh:
        polarized = 1
    else:
        polarized = 0  # not polarized (no significant polarity)

    return polarized


def degree_nonrandom_polarization(R_bar, R_sim):
    # calculate degree of polarisation beyond polarity imposed by cell shape
    mu_null = np.mean(R_sim)
    std_null = np.std(R_sim)

    polarity_z = (R_bar - mu_null) / std_null

    relative_polarity = (R_bar - mu_null) / (1 - mu_null) if mu_null < 1 else 0

    return relative_polarity


def minimum_enclosing_circle(binary_img, method="opencv", shape="circle"):
    """
    Compute the minimum enclosing circle of a binary mask.

    Parameters
    ----------
    binary_image : 2D np.array
        Binary mask (0/1 or 0/255) of the object.
    method : str
        'opencv' for OpenCV method, 'shapely' for Shapely method.

    Returns
    -------
    center : tuple of floats
        (x, y) coordinates of circle center.
    radius : float
        Radius of the circle.
    """
    binary_image = np.array(binary_img, dtype=np.uint8)
    if method.lower() == "opencv":
        # OpenCV expects contours
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            return None, None
        cnt = max(contours, key=cv2.contourArea)  # largest contour

        if shape == "circle":
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            return (x, y), radius

        elif shape == "ellipse":
            ellipse = cv2.fitEllipse(cnt)
            ((x, y), (major, minor), angle) = ellipse
            radius = minor / 2
            return (x, y), radius

    elif method.lower() == "shapely":
        # convert binary image to polygon
        contours = measure.find_contours(binary_image, 0.5)
        if len(contours) == 0:
            return None, None
        poly = Polygon(contours[0][:, ::-1])
        circle_poly = minimum_bounding_circle(poly)
        center = circle_poly.centroid
        radius = np.linalg.norm(
            [
                center.x - list(circle_poly.exterior.coords)[0][0],
                center.y - list(circle_poly.exterior.coords)[0][1],
            ]
        )

        return (center.x, center.y), radius

    else:
        raise ValueError("Method must be 'opencv' or 'shapely'")


def binary_mask_transform(binary_mask, pixels=1, kernel_size=3, method="dilate"):
    """
    Expand or erode a binary mask using OpenCV

    Parameters
    ----------
    binary_mask : 2D np.array
        Binary mask (0/1 or 0/255)
    pixels : int
        Number of pixels to expand
    kernel_size : int
        Size of the structuring element (default 3)

    Returns
    -------
    expanded_mask : 2D np.array
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if method == "dilate":
        transformed_mask = cv2.dilate(
            binary_mask.astype(np.uint8), kernel, iterations=pixels
        )

    elif method == "erode":
        transformed_mask = cv2.erode(
            binary_mask.astype(np.uint8), kernel, iterations=pixels
        )

    return transformed_mask


def split_internal_external_objects(object_labelled, masked_area):
    """
    Returns labelled image with objects within the mask area and labelled image with objects not in mask area
    """
    perinuclear_label_image = np.zeros_like(object_labelled)
    non_perinuclear_label_image = np.zeros_like(object_labelled)

    labels = np.unique(object_labelled)
    labels = labels[labels != 0]  # remove background

    for label in labels:
        mask = object_labelled == label

        # Does ANY voxel of this object fall inside the perinuclear mask?
        if np.any(masked_area & mask):
            # Put the WHOLE object into the perinuclear image
            perinuclear_label_image[mask] = label
        else:
            # Otherwise place the WHOLE object into the non-perinuclear image
            non_perinuclear_label_image[mask] = label

    return perinuclear_label_image, non_perinuclear_label_image


def perinuclear_region_obj(
    nucleus_binary, object_labelled, show_intermediate=False, outline=None, radius=None,
):
    
    if radius is None:
        _, radius = minimum_enclosing_circle(nucleus_binary)
        radius = 0.5 * radius
    
    perinuclear_binary = binary_mask_transform(
        nucleus_binary, pixels=math.ceil(radius), method="dilate"
    )

    perinuclear_mask = perinuclear_binary > 0

    masked_labels = object_labelled[perinuclear_mask]
    object_labels = np.unique(masked_labels)
    object_labels = object_labels[object_labels != 0]  # exclude background
    object_count = len(object_labels)

    perinuclear_area = np.sum(perinuclear_binary > 0)

    if show_intermediate:
        perinuclear_label_image, non_perinuclear_label_image = (
            split_internal_external_objects(object_labelled, perinuclear_mask)
        )
        # display with a colormap suitable for labels
        plt.figure(figsize=(6, 6))
        plt.title(f"Perinuclear objects (count={object_count})")
        plt.title("Perinuclear objects")
        # using 'tab20' / 'nipy_spectral' can give distinct colors for labels

        # plt.imshow(perinuclear_label_image, cmap="nipy_spectral")
        # black background
        black_bg = np.zeros_like(nucleus_binary, dtype=float)
        plt.imshow(black_bg, cmap="gray", vmin=0, vmax=1)

        # nucleus - white
        nuc_mask = nucleus_binary > 0
        nuc_rgba = np.zeros((*nuc_mask.shape, 4))
        nuc_rgba[nuc_mask] = [1, 1, 1, 1]  # white
        plt.imshow(nuc_rgba)

        # perinuclear - green
        peri_mask = perinuclear_label_image > 0
        peri_rgba = np.zeros((*peri_mask.shape, 4))
        peri_rgba[peri_mask] = [0, 1, 0, 1]  # (R,G,B,A) → solid lime green
        plt.imshow(peri_rgba)

        # non-perinuclear - gray
        nonperi_mask = non_perinuclear_label_image > 0
        nonperi_rgba = np.zeros((*nonperi_mask.shape, 4))
        nonperi_rgba[nonperi_mask] = [0.6, 0.6, 0.6, 1]  # solid gray
        plt.imshow(nonperi_rgba)

        # perinuclear boundary - red dash
        peri_img = perinuclear_mask.astype(np.uint8) * 255
        peri_contours, _ = cv2.findContours(
            peri_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for pc in peri_contours:
            pc_xy = pc[:, 0, :]
            plt.plot(pc_xy[:, 0], pc_xy[:, 1], color="red", linewidth=1, linestyle="--")

        if outline is not None:
            outline_img = (outline > 0).astype(np.uint8) * 255
            o_contours, _ = cv2.findContours(
                outline_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Plot each outline contour
            for oc in o_contours:
                if oc.shape[0] < 2:
                    continue
                oc_xy = oc[:, 0, :]
                # cv2 uses (x, y) order for points; matplotlib expects (x, y) pairs

                plt.plot(
                    oc_xy[:, 0],
                    oc_xy[:, 1],
                    color="blue",
                    linewidth=2,
                    label="Cell Outline"
                    if "Cell Outline" not in plt.gca().get_legend_handles_labels()[1]
                    else "",
                )

        legend_patches = [
            Patch(
                facecolor="#00FF00",
                edgecolor="black",
                label="Perinuclear Lipid Droplets",
            ),  # lime green
            Patch(
                facecolor="#888888",
                edgecolor="black",
                label="Non-perinuclear lipid droplets",
            ),  # gray
            Patch(
                facecolor="#FFFFFF", edgecolor="black", label="Nucleus"
            ),  # light gray
        ]
        outline_label = Line2D(
            [0], [0], color="blue", linewidth=1.5, label="Cell outline"
        )
        perinuclear_label = Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.5,
            linestyle="--",
            label="Perinuclear region outline",
        )

        plt.legend(
            handles=legend_patches + [outline_label, perinuclear_label],
            loc="lower right",
            framealpha=1,
            facecolor="white",
            edgecolor="gray",
            labelcolor="black",
            fontsize=8,
        )
        plt.axis("off")
        plt.show()

    object_labels_string = [str(x) for x in list(object_labels)]

    return object_count, perinuclear_area, object_labels_string


def mitochondria_region_obj(
    mitochondria_all_binary,
    object_labelled,
    radius = 30,
    show_intermediate=False,
    outline=None,
):
    mitochondria_binary = binary_mask_transform(
        mitochondria_all_binary, pixels=radius, method="dilate"
    )

    perimitochondria_mask = mitochondria_binary > 0

    masked_labels = object_labelled[perimitochondria_mask]
    object_labels = np.unique(masked_labels)
    object_labels = object_labels[object_labels != 0]  # exclude background
    object_count = len(object_labels)

    perimitochondria_area = np.sum(mitochondria_binary > 0)

    if show_intermediate:
        perimitochondria_label_image, non_perimitochondria_label_image = (
            split_internal_external_objects(object_labelled, perimitochondria_mask)
        )
        # display with a colormap suitable for labels
        plt.figure(figsize=(6, 6))
        plt.title(f"Mitochondria Neighbourhood Objects (count={object_count})")
        plt.title("Mitochondria Neighbourhood Objects")
        # using 'tab20' / 'nipy_spectral' can give distinct colors for labels

        # plt.imshow(perinuclear_label_image, cmap="nipy_spectral")
        # black background
        black_bg = np.zeros_like(mitochondria_binary, dtype=float)
        plt.imshow(black_bg, cmap="gray", vmin=0, vmax=1)

        # nucleus - white
        mito_mask = mitochondria_all_binary > 0
        mito_rgba = np.zeros((*mito_mask.shape, 4))
        mito_rgba[mito_mask] = [1, 1, 1, 1]  # white
        plt.imshow(mito_rgba)

        # perinuclear - green
        peri_mask = perimitochondria_label_image > 0
        peri_rgba = np.zeros((*peri_mask.shape, 4))
        peri_rgba[peri_mask] = [0, 1, 0, 1]  # (R,G,B,A) → solid lime green
        plt.imshow(peri_rgba)

        # non-perinuclear - gray
        nonperi_mask = non_perimitochondria_label_image > 0
        nonperi_rgba = np.zeros((*nonperi_mask.shape, 4))
        nonperi_rgba[nonperi_mask] = [0.6, 0.6, 0.6, 1]  # solid gray
        plt.imshow(nonperi_rgba)

        # perimitochondria boundary - red dash
        peri_img = perimitochondria_mask.astype(np.uint8) * 255
        peri_contours, _ = cv2.findContours(
            peri_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for pc in peri_contours:
            pc_xy = pc[:, 0, :]
            plt.plot(pc_xy[:, 0], pc_xy[:, 1], color="red", linewidth=1, linestyle="--")

        if outline is not None:
            outline_img = (outline > 0).astype(np.uint8) * 255
            o_contours, _ = cv2.findContours(
                outline_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Plot each outline contour
            for oc in o_contours:
                if oc.shape[0] < 2:
                    continue
                oc_xy = oc[:, 0, :]
                # cv2 uses (x, y) order for points; matplotlib expects (x, y) pairs

                plt.plot(
                    oc_xy[:, 0],
                    oc_xy[:, 1],
                    color="blue",
                    linewidth=2,
                    label="Cell Outline"
                    if "Cell Outline" not in plt.gca().get_legend_handles_labels()[1]
                    else "",
                )

        legend_patches = [
            Patch(
                facecolor="#00FF00",
                edgecolor="black",
                label="Mitochondria neighborhood lipid droplets",
            ),  # lime green
            Patch(
                facecolor="#888888",
                edgecolor="black",
                label="Non-mitochondria neighborhood lipid droplets",
            ),  # gray
            Patch(
                facecolor="#FFFFFF", edgecolor="black", label="Mitochondria"
            ),  # light gray
        ]
        outline_label = Line2D(
            [0], [0], color="blue", linewidth=1.5, label="Cell outline"
        )
        perinuclear_label = Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.5,
            linestyle="--",
            label="Mitochondria neighborhood region outline",
        )

        plt.legend(
            handles=legend_patches + [outline_label, perinuclear_label],
            loc="lower right",
            framealpha=1,
            facecolor="white",
            edgecolor="gray",
            labelcolor="black",
            fontsize=6,
        )
        plt.axis("off")
        plt.show()

    object_labels_string = [str(x) for x in list(object_labels)]

    return object_count, perimitochondria_area, object_labels_string


def cell_edge_obj(
    cell_binary,
    object_labelled,
    isolate=None,
    show_intermediate=False,
    other_objects=None,
):
    cell_binary = (cell_binary > 0).astype(np.uint8)
    _, radius = minimum_enclosing_circle(cell_binary, shape="ellipse")
    cell_internal = binary_mask_transform(
        cell_binary, pixels=math.ceil(0.05 * radius), method="erode"
    )
    cell_edge = cv2.subtract(cell_binary, cell_internal)

    if isolate != None:
        _, radius = minimum_enclosing_circle(isolate)
        isolate_region = binary_mask_transform(
            isolate, pixels=math.ceil(0.5 * radius), method="dilate"
        )

        cell_edge = cv2.subtract(cell_edge, isolate_region)

    edge_mask = cell_edge > 0

    masked_labels = object_labelled[edge_mask]
    object_labels = np.unique(masked_labels)
    object_labels = object_labels[object_labels != 0]  # exclude background
    object_count = len(object_labels)

    cell_edge_area = np.sum(cell_edge > 0)

    if show_intermediate:
        edge_label_image, non_edge_label_image = split_internal_external_objects(
            object_labelled, edge_mask
        )
        # display with a colormap suitable for labels
        plt.figure(figsize=(6, 6))
        plt.title(f"Cell edge objects (count={object_count})")
        plt.title("Cell edge objects")
        # using 'tab20' / 'nipy_spectral' can give distinct colors for labels

        # plt.imshow(perinuclear_label_image, cmap="nipy_spectral")
        # 1. Black background
        black_bg = np.zeros_like(object_labelled, dtype=float)
        plt.imshow(black_bg, cmap="gray", vmin=0, vmax=1)

        # show any other objects - white
        if other_objects is not None:
            object_mask = other_objects > 0
            nuc_rgba = np.zeros((*object_mask.shape, 4))
            nuc_rgba[object_mask] = [1, 1, 1, 1]  # white
            plt.imshow(nuc_rgba)

        # edge - green
        edge_objmask = edge_label_image > 0
        edge_rgba = np.zeros((*edge_objmask.shape, 4))
        edge_rgba[edge_objmask] = [0, 1, 0, 1]  # (R,G,B,A) → solid lime green
        plt.imshow(edge_rgba)

        # non-edge - gray
        nonedge_objmask = non_edge_label_image > 0
        nonedge_rgba = np.zeros((*nonedge_objmask.shape, 4))
        nonedge_rgba[nonedge_objmask] = [0.6, 0.6, 0.6, 1]  # solid gray
        plt.imshow(nonedge_rgba)

        # cell edge boundary - red dash
        edge_img = cell_internal.astype(np.uint8) * 255
        edge_contours, _ = cv2.findContours(
            edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for ec in edge_contours:
            ec_xy = ec[:, 0, :]
            plt.plot(
                ec_xy[:, 0],
                ec_xy[:, 1],
                color="red",
                linewidth=1,
                linestyle="--",
                zorder=50,
            )

        # cell boundary - blue line
        outline_img = (cell_binary > 0).astype(np.uint8) * 255
        o_contours, _ = cv2.findContours(
            outline_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # Plot each outline contour
        for oc in o_contours:
            if oc.shape[0] < 2:
                continue
            oc_xy = oc[:, 0, :]
            # cv2 uses (x, y) order for points; matplotlib expects (x, y) pairs

            plt.plot(
                oc_xy[:, 0],
                oc_xy[:, 1],
                color="blue",
                linewidth=2,
                label="Cell Outline"
                if "Cell Outline" not in plt.gca().get_legend_handles_labels()[1]
                else "",
            )

        legend_patches = [
            Patch(
                facecolor="#00FF00",
                edgecolor="black",
                label="Perinuclear Lipid Droplets",
            ),  # lime green
            Patch(
                facecolor="#888888",
                edgecolor="black",
                label="Non-perinuclear lipid droplets",
            ),  # gray
            Patch(
                facecolor="#FFFFFF", edgecolor="black", label="Nucleus"
            ),  # light gray
        ]
        outline_label = Line2D(
            [0], [0], color="blue", linewidth=1.5, label="Cell outline"
        )
        perinuclear_label = Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.5,
            linestyle="--",
            label="Cell edge region outline",
        )

        plt.legend(
            handles=legend_patches + [outline_label, perinuclear_label],
            loc="lower right",
            framealpha=1,
            facecolor="white",
            edgecolor="gray",
            labelcolor="black",
            fontsize=8,
        )
        plt.axis("off")
        plt.show()

        """
    if show_plot:
        edge_label_image = np.where(edge_mask, object_labelled, 0)

        # display with a colormap suitable for labels
        plt.figure(figsize=(6, 6))
        plt.title(f"Edge objects (count={object_count})")
        # using 'tab20' / 'nipy_spectral' can give distinct colors for labels
        plt.imshow(edge_label_image, cmap="nipy_spectral")
        plt.axis("off")
        plt.show()
        """
    object_labels_string = [str(x) for x in list(object_labels)]

    return object_count, cell_edge_area, object_labels_string


def cluster_number_points(
    points, eps=100, min_samples=3, outline=None, show_intermediate=False
):
    """
    ******Assumes points are in order i.e. index matches object ID******

    Cluster 2D points with DBSCAN and (optionally) plot results.

    Parameters
    ----------
    coords : list of tuples
        [(x1, y1), (x2, y2), ...]
    cellshape : shapely Polygon, optional
        Polygon representing the cell boundary for context
    eps : float
        DBSCAN neighborhood radius
    min_samples : int
        Minimum samples for DBSCAN cluster
    plot : bool
        If True, plot the results
    """

    points = np.array(points)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(points)
    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    cluster_labels = labels[labels != -1]
    unique, counts = np.unique(cluster_labels, return_counts=True)
    avg_cluster_size = counts.mean() if counts.size != 0 else 0

    point_ids = np.arange(len(points))
    cluster_member_ids = point_ids[labels != -1]
    noise_ids = point_ids[labels == -1]

    cluster_member_ids_string = [str(x + 1) for x in list(cluster_member_ids)]
    noise_ids_string = [str(x + 1) for x in list(noise_ids)]

    if show_intermediate:
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the cell shape if provided
        """
        if outline is not None:
            gpd.GeoSeries([outline]).plot(
                ax=ax,
                edgecolor="green",
                facecolor="green",
                alpha=0.2,
                label="Cell Outline",
            )
        """

        if outline is not None:
            # black background
            black_bg = np.zeros_like(outline, dtype=float)
            plt.imshow(black_bg, cmap="gray", vmin=0, vmax=1)

            outline_mask = outline > 0
            outline_rgba = np.zeros((*outline_mask.shape, 4))
            outline_rgba[outline_mask] = [1, 1, 1, 1]  # white
            plt.imshow(outline_rgba)

        # Noise points (label = -1)
        noise_mask = labels == -1
        ax.scatter(
            points[noise_mask, 0],
            points[noise_mask, 1],
            c="grey",
            s=10,
            linewidth=0,
            label="Lipid droplets",
        )

        # Clustered points
        ax.scatter(
            points[~noise_mask, 0],
            points[~noise_mask, 1],
            c="red",
            s=10,
            linewidth=0,
            label="Clustered lipid droplets",
        )

        ax.set_axis_off()
        plt.legend(loc="lower right")
        plt.tight_layout()
        # plt.title("Lipid Droplet Clusters")
        plt.show()

    return n_clusters, avg_cluster_size, cluster_member_ids_string, noise_ids_string


def collapse_multilayer(img_list):
    stack = np.stack(img_list, axis=0)
    combined = (np.any(stack > 0, axis=0)).astype(np.uint8)

    return combined


# functions for calculating summary statistics (mean, median, standard deviation, skewness) of specified features grouped by cell ID from cellProfiler output / several xlsx/csv files


def add_metadata_measurementfile(
    measurement_df, mapping_df, columns=[], mapping_column=""
):
    if columns == [] or mapping_column == "":
        print("No columns specified - merging all columns")
        measurement_df_updated = measurement_df.merge(mapping_df)

    else:
        measurement_df_updated = measurement_df.merge(
            mapping_df[columns], on=mapping_column, how="left"
        )

    return measurement_df_updated


def combine_files_to_sheets(directory, keywords=[], keyword_method="all", save=False):
    """
    Combine all CSV files in a directory into a single Excel file with each file as a separate sheet.

    Parameters
    ----------
    directory : str
        Path to the directory containing the CSV files.
    keyword: list of keywords to specify files

    Returns
    -------
    allsheets: dataframe
        Path to the saved Excel file.
    """
    allsheets = {}

    for file in Path(directory).iterdir():
        filename = file.stem  # filename without extension

        include = True
        if keywords != []:
            if keyword_method == "any":
                include = any(k in filename for k in keywords)
            elif keyword_method == "all":
                include = all(k in filename for k in keywords)
            elif keyword_method == "none":
                include = not any(k in filename for k in keywords)
        else:
            include = True

        if not include:
            continue

        if file.suffix == ".csv":
            df = pd.read_csv(file)
        elif file.suffix == ".xlsx":
            df = pd.read_excel(file)  # default reads first sheet
        else:
            continue  # skip non-csv/xlsx files

        allsheets[filename] = df

    if save:
        savePath = Path(directory) / "all_combined.xlsx"

        # Create an Excel writer object
        with pd.ExcelWriter(savePath, engine="openpyxl") as writer:
            for sheetname, df in allsheets.items():
                df.to_excel(writer, sheet_name=sheetname, index=False)

        print(f"Combined files saved to {savePath}")

    return allsheets


def add_CellID(df, mapdf):
    # Add CellID to ImageNumber in invididual organelle sheets
    df_cellID = {}

    for sheet_name, df in df.items():
        if "ImageNumber" in df.columns:
            df["Metadata_CellID"] = df["ImageNumber"].map(mapdf)

            df_cellID[sheet_name] = df

    return df_cellID


def drop_duplicate_cell_data(sheet_dictionary, ignore_cols=""):
    """
    Drop duplicate cell data from the sheets in the dictionary.

    Parameters
    ----------
    sheet_dictionary : dict
        Dictionary containing DataFrames for each sheet.

    Returns
    -------
    dict
        Dictionary with duplicates removed.
    """

    columns_to_check = ["AreaShape_Area", "AreaShape_Perimeter", "Metadata_CellID"]

    extracted_dataframe_dict = {}

    for sheet_name, df in sheet_dictionary.items():
        if "ImageNumber" in df.columns and "AreaShape_Area" in df.columns:
            extracted_df = df.drop_duplicates(subset=columns_to_check, keep="first")
            extracted_dataframe_dict[sheet_name] = extracted_df

    return extracted_dataframe_dict


def image_overlap(img_1, img_2, show_intermediate=True):
    img_1 = img_1 > 0
    img_2 = img_2 > 0

    overlap = np.logical_and(img_1, img_2)

    return overlap.sum()  # overlap area


def add_CellID_to_cellprofiler_measurements(filedir, organelle_list: list = []):
    # if organelle_list = []: # add in feature to find organelle list from naming convention "organelle_"

    # create combined dataframe for each organelle
    combined_dataframes = {}
    for organelle in organelle_list:
        combined_dataframes[organelle] = combine_files_to_sheets(
            filedir, keywords=[organelle]
        )

    # update sheets with CellID
    updated_dataframes = {}
    for organelle, organelle_df in combined_dataframes.items():
        mapping_df = organelle_df[f"{organelle}_Image"]
        measurement_df = organelle_df[f"{organelle}_{organelle}"]

        updated_sheet = add_metadata_measurementfile(
            measurement_df,
            mapping_df,
            columns=["ImageNumber", "Metadata_CellID"],
            mapping_column="ImageNumber",
        )

        updated_dataframes[organelle] = updated_sheet

    return updated_dataframes


def context_classification_lipid(
    updated_dataframes: dict, all_segmentation: dict, id_col_values
):
    # untested for mitochondria and not adapted to mitochondria data format
    classification_df = pd.DataFrame(id_col_values)
    classification_df["CellID"] = (
        classification_df[id_col_values.name].str.split("_", n=1).str[0]
    )

    nucleus_segmentation = all_segmentation[constants.NUCLEUS]
    lipiddroplets_segmentation = all_segmentation[constants.LIPID_DROPLETS]
    cell_segmentation = all_segmentation[constants.CELL]
    mitochondria_segmentation = all_segmentation[constants.MITOCHONDRIA]

    lipiddroplets_dataset = updated_dataframes[constants.LIPID_DROPLETS]

    cell_list = list(lipiddroplets_segmentation.keys())

    perinuclear_ids = []
    edge_ids = []
    cluster_ids = []
    permitochondria_ids = []

    for cell in tqdm(cell_list):
        _, _, perinuclear_obj_id = perinuclear_region_obj(
            nucleus_segmentation[cell],
            lipiddroplets_segmentation[cell],
            #show_plot=False,
        )

        _, _, cell_edge_obj_id = cell_edge_obj(
            cell_segmentation[cell], lipiddroplets_segmentation[cell], #show_plot=False
        )

        img_list = list(
            mitochondria_segmentation
            .loc[
                mitochondria_segmentation[constants.CELLID_COL] == cell,
                "image",
            ]
            .values
        )
        mitochondria_mask = collapse_multilayer(img_list) > 0

        _,_, perimitochondrial_obj_id = mitochondria_region_obj(
            mitochondria_mask, lipiddroplets_segmentation[cell], radius = 30
        )

        cell_dataset = lipiddroplets_dataset[lipiddroplets_dataset[constants.CELLID_COL] == cell]
        lipiddroplet_points = list(
            zip(cell_dataset["AreaShape_Center_X"], cell_dataset["AreaShape_Center_Y"])
        )
        _, _, cluster_member_ids_string, noise_ids_string = cluster_number_points(
            lipiddroplet_points
        )

        cell_perinuclear_ids = [f"{cell}_0_" + x for x in perinuclear_obj_id]
        cell_edge_ids = [f"{cell}_0_" + x for x in cell_edge_obj_id]
        cell_cluster_ids = [f"{cell}_0_" + x for x in cluster_member_ids_string]
        cell_perimitochondria_ids = [f"{cell}_0_" + x for x in perimitochondrial_obj_id]

        perinuclear_ids.extend(cell_perinuclear_ids)
        edge_ids.extend(cell_edge_ids)
        cluster_ids.extend(cell_cluster_ids)
        permitochondria_ids.extend(cell_perimitochondria_ids)

    classification_df["Classification_Perinuclear"] = (
        classification_df[id_col_values.name].isin(perinuclear_ids).astype(int)
    )
    classification_df["Classification_CellEdge"] = (
        classification_df[id_col_values.name].isin(edge_ids).astype(int)
    )
    classification_df["Classification_Cluster"] = (
        classification_df[id_col_values.name].isin(cluster_ids).astype(int)
    )
    classification_df["Classification_Perimitochondrial"] = (
        classification_df[id_col_values.name].isin(permitochondria_ids).astype(int)
    )

    return classification_df


def context_classification_mitochondria(
    updated_dataframes: dict, all_segmentation: dict, id_col_values, show_plots=False, DEBUG = False
):
    classification_df = pd.DataFrame(id_col_values)
    classification_df["CellID"] = (
        classification_df[id_col_values.name].str.split("_", n=1).str[0]
    )

    nucleus_segmentation = all_segmentation[constants.NUCLEUS]

    mitochondria_segmentation = all_segmentation[constants.MITOCHONDRIA]
    cell_segmentation = all_segmentation[constants.CELL]

    mitochondria_dataset = updated_dataframes[constants.MITOCHONDRIA]

    cell_list = list(nucleus_segmentation.keys())
    
    perinuclear_ids = []
    edge_ids = []
    cluster_ids = []

    for cell in tqdm(cell_list):
        cell_dataset = mitochondria_dataset[mitochondria_dataset[constants.CELLID_COL] == cell]

        subset_images_list = list(
            mitochondria_segmentation.loc[
                mitochondria_segmentation[constants.CELLID_COL] == cell, "image"
            ].values
        )

        mitochondria_points_centroid = list(
            zip(
                cell_dataset["AreaShape_Center_X"],
                cell_dataset["AreaShape_Center_Y"],
            )
        )

        if DEBUG:
            return mitochondria_dataset

        cell_perinuclear_ids = []
        cell_edge_ids = []
        cell_cluster_ids = []
        for layer_num, layer_img in enumerate(subset_images_list):
            _, _, perinuclear_obj_id = perinuclear_region_obj(
                nucleus_segmentation[cell], layer_img, 
                #show_plot=show_plots
            )
            _, _, cell_edge_obj_id = cell_edge_obj(
                cell_segmentation[cell], layer_img, 
                #show_plot=show_plots
            )
            _, _, cluster_member_ids_string, noise_ids_string = cluster_number_points(
                mitochondria_points_centroid, eps=1000
            )

            layer_perinuclear_ids = [
                f"{cell}_{layer_num}_" + x for x in perinuclear_obj_id
            ]
            layer_cell_edge_ids = [f"{cell}_{layer_num}_" + x for x in cell_edge_obj_id]
            cell_cluster_ids = [
                f"{cell}_{layer_num}_" + x for x in cluster_member_ids_string
            ]

            cell_perinuclear_ids.extend(layer_perinuclear_ids)
            cell_edge_ids.extend(layer_cell_edge_ids)
            cell_cluster_ids.extend(cell_cluster_ids)

        perinuclear_ids.extend(cell_perinuclear_ids)
        edge_ids.extend(cell_edge_ids)
        cluster_ids.extend(cell_cluster_ids)

    classification_df["Classification_Perinuclear"] = (
        classification_df[id_col_values.name].isin(perinuclear_ids).astype(int)
    )
    classification_df["Classification_CellEdge"] = (
        classification_df[id_col_values.name].isin(edge_ids).astype(int)
    )
    classification_df["Classification_Cluster"] = (
        classification_df[id_col_values.name].isin(cluster_ids).astype(int)
    )

    return classification_df


def texture_measure_average(dataframe):
    # column-wise average
    cols = {}
    feature_basename_list = []
    for feature in dataframe.columns:
        components = feature.split("_")
        if len(components) < 4:
            continue

        if components[0] != "Texture":
            continue

        basename = components[0] + "_" + components[1] + "_" + components[2]
        if basename not in feature_basename_list:
            feature_basename_list.append(basename)

    dataset_avg = pd.DataFrame()
    for base in feature_basename_list:
        # find columns containing the base name
        cols = [c for c in dataframe.columns if base in c]

        if len(cols) < 2:
            continue

        # compute row-wise mean
        dataset_avg[f"{base}_avg"] = dataframe[cols].mean(axis=1)

    return dataset_avg


def dilate_l1_by_radius(mask_uint8, radius):
    """
    mask_uint8: 0 or 255 image (uint8) of the ROI
    radius: integer neighborhood radius (in pixels)
    returns: 0/255 uint8 mask dilated with L1 (Manhattan) radius
    """
    # Convert to 0/1
    mask = (mask_uint8 > 0).astype(np.uint8)

    # Invert: we want distances from background to foreground
    inv = 1 - mask

    # Distance transform with L1 metric
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L1, maskSize=3)

    # Pixels with distance <= radius are within L1 radius
    expanded = (dist <= radius).astype(np.uint8) * 255

    return expanded


def neighbour_analysis(single_roi, remaining_roi_list, neighbor_radius=0):
    shape_img = single_roi.astype(np.uint8) * 255

    other_imgs = [(img.astype(np.uint8) * 255) for img in remaining_roi_list]
    combined_others = combine_indiv_roi_masks(other_imgs)

    expanded = shape_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    steps = 0
    while True:
        # Dilate by 1 pixel
        expanded = cv2.dilate(expanded, kernel, iterations=1)
        # Check if it touches
        overlap = cv2.bitwise_and(expanded, combined_others)
        if np.any(overlap):
            break

        steps += 1

    if neighbor_radius != 0:
        expanded_neighborhood = shape_img.copy()
        expanded_neighborhood = dilate_l1_by_radius(
            expanded_neighborhood, neighbor_radius
        )

    neighbours_touched = 0
    for other_img in other_imgs:
        if neighbor_radius == 0:
            overlap = np.logical_and(expanded > 0, other_img > 0)
        else:
            overlap = np.logical_and(expanded_neighborhood > 0, other_img > 0)

        if np.any(overlap):
            neighbours_touched += 1

    return steps, neighbours_touched


def combine_indiv_roi_masks(roi_list):
    """
    roi_list: list
        list of individual roi images
    """

    # all_imgs = [(img.astype(np.uint8) * 255) for img in roi_list]

    combined_rois = np.zeros_like(roi_list[0], dtype=np.uint8)
    for other in roi_list:
        combined_rois = cv2.bitwise_or(combined_rois, other)

    return combined_rois
