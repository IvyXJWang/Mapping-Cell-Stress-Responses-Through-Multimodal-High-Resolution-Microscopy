# idnividual organelle analysis functions
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

import data_analysis_utils as utils
from feature_extraction_utils import combine_files_to_sheets, load_segmentation
from segmentation_utils import load_path_into_dict, grayscale_histogram
from classification import (
    filter_dataframes,
    silhouette_score_indiv,
    silhouette_score_stats,
    add_cluster_label,
)
import plotting as plot
import constants

# plotting


def plot_hover(dataset, dataset_labelled, scatter_feat_list, colors, savedir=""):
    _ = plot.plot_scatter_feat_combo(
        dataset,
        scatter_feat_list,
        hover=True,
        id_col="Metadata_CellID",
        savedir=savedir,
        labels=dataset_labelled["Subtype"],
        cluster_colors=colors,
    )

    return


def plot_scatters(
    dataset, dataset_labelled, scatter_feat_list, cmap_dict={}, savedir=""
):
    _ = plot.plot_scatter_feat_combo(
        dataset,
        scatter_feat_list,
        hover=False,
        id_col="Metadata_CellID",
        savedir=savedir,
        labels=dataset_labelled["Subtype"],
        kind="density",
    )

    if cmap_dict != {}:
        _ = plot.plot_scatter_feat_combo(
            dataset,
            scatter_feat_list,
            hover=False,
            id_col="Metadata_CellID",
            savedir=savedir,
            labels=dataset_labelled["Subtype"],
            cluster_colors=cmap_dict,
            kind="",
        )

    return

def add_scale_bar(
    ax,
    px_size,
    length_um,
    color="white",
    lw=3,
    pad=100,
    location="lower right",
    fontsize=10,
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
    ax.text(
        x0 + length_px / 2,
        text_y,
        f"{length_um:g} µm",
        ha="center",
        va=text_va,
        color=color,
        fontsize=fontsize,
    )

def plot_selected(
    segmentation_df,
    organelle,
    silhouette_score_stats_dict=None,
    list_of_obj=None,
    readfile=False,
    max_display="",
    population=0,
    xray_overlay=False,
    histogram=False,
    xray_paths=None,
    display_window=(0, 100),
    cropped_roi=True,
    cropped_xray=False,
    mask_outside = False,
    lw = 10,
    scale_bar = False,
):
    # get list of objects to display
    if silhouette_score_stats_dict is not None and readfile is False:
        list_of_obj = [
            ID
            for ID, score in silhouette_score_stats_dict["subpopulation_ranked_cells"][
                population
            ]
        ]

    elif readfile:
        selected_obj = pd.read_csv(constants.DOWNLOADDIR / "selected_points.csv")
        list_of_obj = selected_obj["ID"].tolist()

    elif list_of_obj is not None:
        pass

    else:
        print("No objects to display.")
        return

    if max_display != "":
        list_of_obj = list_of_obj[0:max_display]

    objects = {}
    for obj in list_of_obj:
        objects[obj], cellID = utils.show_object(
            segmentation_df[organelle], organelle, obj
        )

        if np.sum(objects[obj]) == 0:
            print(f"Object {obj} is empty. Skipping.")
            continue

        if cropped_roi:
            ys, xs = np.where(objects[obj])

            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            cropped = objects[obj][ymin : ymax + 1, xmin : xmax + 1]

        else:
            cropped = objects[obj]

        plt.imshow(cropped, cmap="gray")
        # plt.title(f"Subtype {population}: {obj}", fontsize=12)
        plt.title(f"{obj}", fontsize=10)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()

        if xray_overlay and xray_paths is not None:
            fig, ax = plt.subplots(figsize=(8, 8))
            xray_img = np.asarray(tiff.imread(xray_paths[cellID]))
            
            mask = objects[obj]

            if mask_outside:
                # invert mask 
                inverted_mask = np.logical_not(mask)
            else:
                inverted_mask = mask

            overlay = utils.overlay_mask_xray(
                inverted_mask,
                xray_img,
                color=(255, 0, 0),
                lw=lw,
                fill=True if mask_outside else False,
                show=False,
                display_window=display_window,
                alpha=0.5,
            )

            if cropped_xray:
                padding = 250

                h, w = overlay.shape[:2]

                # Desired crop coordinates
                y1 = ymin - padding
                y2 = ymax + 1 + padding
                x1 = xmin - padding
                x2 = xmax + 1 + padding

                # Clamp to image bounds
                y1_clamped = max(0, y1)
                y2_clamped = min(h, y2)
                x1_clamped = max(0, x1)
                x2_clamped = min(w, x2)

                cropped_overlay = overlay[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                # Compute how much padding is needed
                pad_top = y1_clamped - y1
                pad_bottom = y2 - y2_clamped
                pad_left = x1_clamped - x1
                pad_right = x2 - x2_clamped

                # Apply edge padding if needed
                if any(p > 0 for p in [pad_top, pad_bottom, pad_left, pad_right]):
                    if overlay.ndim == 3:
                        pad_width = (
                            (pad_top, pad_bottom),
                            (pad_left, pad_right),
                            (0, 0),
                        )
                    else:
                        pad_width = (
                            (pad_top, pad_bottom),
                            (pad_left, pad_right),
                        )

                    cropped_overlay = np.pad(cropped_overlay, pad_width, mode="edge")

            else:
                cropped_overlay = overlay

            plt.imshow(cropped_overlay)
            
            if scale_bar: 
                add_scale_bar(
                    ax=ax,
                    px_size=constants.PX_SIZE,
                    length_um=10,
                    color="white",
                    lw=10,
                    pad=150,
                    location="lower right",
                    fontsize=0,
                )
            
            plt.axis("off")
            plt.tight_layout(pad=0)
            #plt.title(f"{obj}")
            plt.show()

        if histogram and xray_paths is not None:
            xray_img = np.asarray(tiff.imread(xray_paths[cellID]))
            grayscale_histogram(
                xray_img,
                objects[obj],
                apply_within_mask=True,
                title=f"{obj} Histogram and CDF",
                plot_cdf=True,
            )

    utils.delete_file(constants.DOWNLOADDIR / "selected_points.csv")

    return


def all_feat_violin_plots(
    dataset_labelled, cluster_col=None, id_col=constants.CELLID_COL, cmap=None, axes=True
):
    if cluster_col is None:
        cluster_col = dataset_labelled.columns[-1]

    significant_features, molten_dunn_df, dunn_df = utils.kw_dunn(
        dataset_labelled, alpha=0.05, p_adjust="fdr_bh", cluster_col=cluster_col
    )

    if len(significant_features) == 0:
        print("No significant features found.")
        return significant_features, None

    for feature in significant_features:
        if feature == constants.CELLID_COL:
            continue

        subpopulation_cells = plot.subpopulation_violplt(
            dataset_labelled, dunn_df[feature], cluster_col, feature, cmap=cmap, axes=axes
        )

    return significant_features, subpopulation_cells


# data prep


def load_organelle_measurements(
    datadir, organelle, outlier_check=False, object_id=True, select_features=False, DEBUG=False
):
    segmentationdir = datadir / "results"
    outputdir = datadir / "summary_sheets_updated"
    outputdir.mkdir(parents=True, exist_ok=True)
    figdir = datadir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # combine excel data sheets
    filedir = datadir / "cellprofiler" / "measurements_updated"
    all_sheets = combine_files_to_sheets(filedir)

    if select_features:
        a = datadir / "feature_list_full.xlsx"
        with pd.ExcelWriter(a, engine="openpyxl") as writer:
            for sheet_name, df in all_sheets.items():

                # Convert column index to a vertical DataFrame
                col_df = pd.DataFrame(df.columns.tolist(), columns=["column_name"])

                col_df.to_excel(
                    writer,
                    sheet_name=str(sheet_name)[:31],  # Excel sheet name limit
                    index=False
                )

        return all_sheets

    if object_id:
        # add object ID to individual organelles
        all_sheets[constants.MITOCHONDRIA] = utils.update_cellID_objectnum(
            all_sheets[constants.MITOCHONDRIA],
            "ImageNumber",
            "ObjectNumber",
            constants.CELLID_COL,
        )
        all_sheets[constants.LIPID_DROPLETS] = utils.update_cellID_objectnum(
            all_sheets[constants.LIPID_DROPLETS],
            "ImageNumber",
            "ObjectNumber",
            constants.CELLID_COL,
        )

    keep_cols = pd.read_excel(datadir / "selected_features.xlsx", sheet_name=None, header=None)

    # extract selected features
    all_organelles_filtered = filter_dataframes(
        all_sheets,
        keep_cols,
        savedir="",
    )
    if DEBUG:
        return all_organelles_filtered

    all_features_list = list(all_organelles_filtered[organelle].columns)
    organelle_dataframe = all_organelles_filtered[organelle][all_features_list]

    # drop empty columns
    organelle_dataframe_nonempty, empty_cols = utils.drop_empty_cols(
        organelle_dataframe, threshold=0
    )  # 0.2 = 20% empty

    # check for outliers
    if outlier_check:
        organelle_dataframe_clean, outlier_cols = utils.remove_outliers(
            organelle_dataframe_nonempty
        )
    else:
        organelle_dataframe_clean = organelle_dataframe_nonempty

    segmentation_masks = load_segmentation(segmentationdir, organelle=[organelle])
    xray_paths = load_path_into_dict(segmentationdir, keyword="xray")

    return (
        organelle_dataframe_clean,
        segmentation_masks,
        xray_paths,
        all_organelles_filtered,
    )


def split_dataset_by_featuretype(dataframe):
    dataset_type_split = {}
    feature_list = [[feature] for feature in constants.FEATURE_TYPE_LIST]
    
    for feature in feature_list:
        if len(feature) > 1:
            feature_combined_list = [x for x in feature]
            feature_name = "_".join(feature_combined_list)
        else:
            feature_name = feature[0]

        split_dataframe = utils.extract_df_subset_sequential(
            dataframe, subset=[feature], constant_key=constants.CELLID_COL
        )

        dataset_type_split[feature_name] = split_dataframe.reset_index()

    return dataset_type_split


def organelle_shape_dataset(dataset_type_split, type="all"):
    remove_features_Shape = [
        "AreaShape_EulerNumber",
        "AreaShape_MaxFeretDiameter",
        "AreaShape_MinFeretDiameter",
        "AreaShape_MaximumRadius",
        "AreaShape_MedianRadius",
        "index",
    ]

    dataset_type_col_list = list(dataset_type_split[constants.OBJ_SHAPE].columns)

    if type == "all":
        feat_list = [x for x in dataset_type_col_list if x not in remove_features_Shape]

    elif type == "zernike":
        feat_list = [
            x
            for x in dataset_type_col_list
            if x not in remove_features_Shape
            and ("Zernike" in x or constants.CELLID_COL in x)
        ]

    elif type == "geometric":
        feat_list = [
            x
            for x in dataset_type_col_list
            if x not in remove_features_Shape and "Zernike" not in x
        ]

    dataset_shape = dataset_type_split[constants.OBJ_SHAPE][feat_list]

    return dataset_shape


def organelle_texture_dataset(dataset_type_split):
    feature_basename = []
    for feature in dataset_type_split[constants.OBJ_TEXTURE].columns:
        if feature == "index":
            continue

        components = feature.split("_")
        basename = components[0] + "_" + components[1]

        if basename not in feature_basename:
            feature_basename.append(basename)

    dataset_texture = pd.DataFrame()
    for base in feature_basename:
        if base == constants.CELLID_COL:
            continue

        # find columns containing the base name
        cols = [
            c for c in dataset_type_split[constants.OBJ_TEXTURE].columns if base in c
        ]

        # compute row-wise mean
        dataset_texture[f"{base}_avg"] = dataset_type_split[constants.OBJ_TEXTURE][
            cols
        ].mean(axis=1)

    dataset_texture[constants.CELLID_COL] = dataset_type_split[constants.OBJ_TEXTURE][
        constants.CELLID_COL
    ]

    return dataset_texture


def organelle_intensity_dataset(dataset_type_split):
    remove_features_Intensity = [
        "Intensity_MaxIntensity_xray",
        "Intensity_MinIntensity_xray",
        "Intensity_MaxIntensityEdge_xray",
        "Intensity_MinIntensityEdge_xray",
        "Intensity_MedianIntensity_xray",
        "Intensity_MADIntensity_xray",
        "Intensity_LowerQuartileIntensity_xray",
        "Intensity_UpperQuartileIntensity_xray",
        "RadialDistribution_FracAtD_xray_1of3",
        "RadialDistribution_FracAtD_xray_2of3",
        "RadialDistribution_FracAtD_xray_3of3",
        "index",
    ]

    itensity_key = constants.OBJ_INTENSITY + "_" + constants.OBJ_RADIALDISTRIBUTION
    src = dataset_type_split[itensity_key]

    dataset_type_col_list = list(dataset_type_split[itensity_key].columns)
    feat_list = [x for x in dataset_type_col_list if x not in remove_features_Intensity]
    dataset_intensity = src.loc[:, feat_list].copy()

    # add additional measurements
    intensity_range = (
        dataset_type_split[itensity_key]["Intensity_MaxIntensity_xray"]
        - dataset_type_split[itensity_key]["Intensity_MinIntensity_xray"]
    )
    edgeintensity_range = (
        dataset_type_split[itensity_key]["Intensity_MaxIntensityEdge_xray"]
        - dataset_type_split[itensity_key]["Intensity_MinIntensityEdge_xray"]
    )

    dataset_intensity["Intensity_Range"] = intensity_range
    dataset_intensity["Intensity_RangeEdge"] = edgeintensity_range

    return dataset_intensity


def stacked_cluster_box_plts_pair(
    dataset_labelled, id_col, cluster_col, label_color_mapping=None
):
    plot.stacked_bxplt_count(
        dataset_labelled, id_col, cluster_col, label_color_mapping=label_color_mapping
    )
    plot.stacked_bxplt_count(
        dataset_labelled,
        id_col,
        cluster_col,
        label_color_mapping=label_color_mapping,
        normalize=True,
    )

    return


def subtype_analysis(
    dataset, cluster_num=2, cluster_method="KMeans", title="", label_method="PCA"
):
    PCA_labelled, colors, label_names, label_color_mapping = (
        plot.PCA_clustering_silhouette(dataset, title, cluster_num, cluster_method)
    )

    dataset_scaled = utils.scale_data(dataset)

    dataset_labelled = dataset.copy()
    if label_method == "PCA":
        dataset_labelled["Subtype"] = PCA_labelled[PCA_labelled.columns[-1]]
    elif label_method == "KMeans":
        dataset_labelled = add_cluster_label(
            dataset_scaled,
            cluster_num=cluster_num,
            clustering_algorithm="KMeans",
            id_col=constants.CELLID_COL,
            lbl_colname="Subtype",
        )
    elif label_method == "HCA":
        dataset_labelled = add_cluster_label(
            dataset_scaled,
            cluster_num=cluster_num,
            clustering_algorithm="HCA",
            id_col=constants.CELLID_COL,
            lbl_colname="Subtype",
        )

    individual_scores_df = silhouette_score_indiv(
        dataset,
        dataset_labelled["Subtype"],
        scale=True,
        cellid_col=constants.CELLID_COL,
    )
    silhouette_score_stats_dict = silhouette_score_stats(individual_scores_df)

    significant_features = all_feat_violin_plots(dataset_labelled)

    # percentage of subtype in each cell
    dataset_labelled["CellID"] = dataset_labelled[constants.CELLID_COL].str.partition(
        "_"
    )[0]  # add cellID back

    stacked_cluster_box_plts_pair(
        dataset_labelled, "CellID", "Subtype", label_color_mapping=label_color_mapping
    )

    return (
        silhouette_score_stats_dict,
        dataset_labelled,
        significant_features,
        colors,
        label_color_mapping,
    )
