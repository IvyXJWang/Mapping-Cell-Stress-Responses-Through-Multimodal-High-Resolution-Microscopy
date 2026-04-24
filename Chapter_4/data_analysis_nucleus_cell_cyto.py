# mitochondria organelle analysis

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from segmentation_utils import load_path_into_dict
import constants
import data_analysis_utils as utils
import data_analysis_organelle_utils as org
from feature_extraction_utils import combine_files_to_sheets, load_segmentation
from classification import (
    HCA_top_features,
    filter_dataframes,
    silhouette_score_indiv,
    silhouette_score_stats,
    MRMR_dataframe,
    plot_feature_relevance,
    add_cluster_label,
    label_color_mapping_dict
)
from feature_extraction_utils import (
    load_segmentation,
    context_classification_mitochondria,
)
import plotting as plot
from utils import rename_df_columns_with_keyword
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib as mpl
import seaborn as sns

# %%
run = "run4"
treatment = "control"
datadir = Path(constants.PARENTDIR / f"{treatment}" / f"{run}")
keep_cols_reduced = pd.read_excel(
    datadir / "selected_features_organelle.xlsx", sheet_name=None, header=None
)

segmentationdir = datadir / "results"
xray_paths = load_path_into_dict(segmentationdir, keyword="xray")

filedir = datadir / "cellprofiler" / "measurements_updated"
all_sheets = combine_files_to_sheets(filedir)

rename_dict = {
    "RadialDistribution": "Densitometric",
    "AreaShape": "Geometric",
    "Intensity": "Densitometric",
    "Texture": "Textural",
    "Structure": "Structural",
    "cell" : "Cell",
    "cytoplasm" : "Cytoplasm",
    "lipiddroplets" : "LipidDroplets",
    "mitochondria" : "Mitochondria",
    "nucleus": "Nucleus"
    }


all_segmentation = load_segmentation(
    segmentationdir, organelle=constants.ORGANELLE_CP_LIST
)

all_sheets_filtered = filter_dataframes(all_sheets, keep_cols_reduced)

#%% cluster network
def clr_transform(df, eps=1e-9):
    x = df.to_numpy(dtype=float)
    x = np.clip(x, eps, None)
    clr = np.log(x) - np.log(x).mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=df.index, columns=df.columns)

from sklearn.preprocessing import StandardScaler

# network clustering
keep_cols_cell = {}
sheet = {}


keep_cols_cell["mitochondria"] = keep_cols_reduced["network"]
sheet["mitochondria"] = all_sheets["mitochondria"]
filtered = filter_dataframes(sheet, keep_cols_cell)["mitochondria"]

no_duplicated = (
    filtered
    .drop_duplicates(subset="Metadata_CellID")
    )

comp_id = no_duplicated["Metadata_CellID"].copy()
comp_props = no_duplicated[["Structure_BranchType0", "Structure_BranchType1", "Structure_BranchType2", "Structure_BranchType3"]]

# CLR transform on proportions
comp_props_clr = clr_transform(comp_props)

# optional: standardize after CLR
comp_props_clr_scaled = pd.DataFrame(
    StandardScaler().fit_transform(comp_props_clr),
    index=comp_props_clr.index,
    columns=comp_props_clr.columns
)

# combine back
mito_clr = pd.concat([comp_id, comp_props_clr_scaled], axis=1)

scaled = utils.scale_data(no_duplicated)
processed_df = scaled.copy()

processed_df = processed_df.set_index('Metadata_CellID')
mito_clr = mito_clr.set_index('Metadata_CellID')

processed_df.update(mito_clr)
processed_df = processed_df.reset_index()

props_renamed = rename_df_columns_with_keyword(processed_df, rename_dict)

props_renamed.to_excel(
            filedir / "network.xlsx", index=False
        )
#%%
clusternum = 2

labelled, _ = add_cluster_label(
    props_renamed,
    cluster_num=clusternum,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_,_,cmap_labelling = label_color_mapping_dict(
    labelled["Subtype"], palette="inferno"
)

_, clustermap_cmap = plot.show_clustermap(
    labelled, 
    color_row_cmap=cmap_labelling, 
    clustering_algorithm = "ward",
    color_feat="type", 
    label_col="Subtype", 
    id_col=constants.CELLID_COL, 
    scale = False, 
    pltsize=(25,25), 
    score_range=(-1,1),
    cmap = "viridis",
    all_legends = True
)
#%% identify cluster number dataframe

organelle = "cell"
clusternum = 3

props = all_sheets_filtered[organelle]
props_renamed = rename_df_columns_with_keyword(props, rename_dict)
scaled = utils.scale_data(props_renamed)

props_normalised = scaled.copy()
feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]
for feature in feature_types:
    cols = props_normalised.columns[props_normalised.columns.str.contains(feature, case=False)]
    #print(len(cols))
    props_normalised[cols] /= np.sqrt(len(cols))
    
labelled, _ = add_cluster_label(
    props_normalised,
    cluster_num=clusternum,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_,_,cmap_labelling = label_color_mapping_dict(
    labelled["Subtype"], palette="inferno"
)

_, clustermap_cmap = plot.show_clustermap(
    labelled, 
    color_row_cmap=cmap_labelling, 
    clustering_algorithm = "ward",
    color_feat="type", 
    label_col="Subtype", 
    id_col=constants.CELLID_COL, 
    scale = False, 
    pltsize=(25,25), 
    score_range=(-0.5,0.5),
    cmap = "viridis",
    all_legends = True
)

# evaluation of clustering
individual_scores_df = silhouette_score_indiv(
    scaled,
    labelled["Subtype"],
    scale=False,
    cellid_col="Metadata_CellID",
)
silhouette_score_stats_dict = silhouette_score_stats(individual_scores_df)
print(silhouette_score_stats_dict["silhouette_score_avg"])

counts = labelled['Subtype'].value_counts()
labels = labelled["Subtype"]

# check dominating features
features_sel, sel = MRMR_dataframe(props_renamed, labels, id_col=constants.CELLID_COL, max_features = None)

feature_scores = pd.DataFrame()
feature_scores["Feature"] = sel.variables_
feature_scores["Score"] = sel.relevance_

#%% identify characteristics of subtypes

feat_idx = 0

feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]
feat_type = feature_types[feat_idx]

dataframe_full = props_renamed.copy()

subset_feature_list = [
    x
    for x in dataframe_full.columns
    if (feat_type in x.split("_")[0] and "Zernike" not in x or constants.CELLID_COL in x)
]
'''
subset_feature_list = [
    x
    for x in dataframe_full.columns
    if (feat_type in x.split("_")[0] or constants.CELLID_COL in x)
]
'''
if len(subset_feature_list) == 1:
    print("Check feature spelling")

#subset_feature_list = [constants.CELLID_COL, "Geometric_FormFactor"]#, "Geometric_AspectRatio"]
#subset_feature_list = [constants.CELLID_COL, "Geometric_FormFactor"]
#subset_feature_list = [constants.CELLID_COL, "Densitometric_MeanDensitometric_xray"]
#subset_feature_list = [constants.CELLID_COL, "Textural_Contrast_xray_avg"]
#subset_feature_list = [constants.CELLID_COL, "Structural_DistanceNucleus"]

dataframe_labelled = dataframe_full[subset_feature_list].copy()

dataframe_labelled.loc[:, "Subtype"] = labels
dataframe_full.loc[:, "Subtype"] = labels

significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled,
    cluster_col="Subtype",
    cmap=clustermap_cmap,
    axes = True
    
)
summary_df = dataframe_labelled.groupby('Subtype').mean(numeric_only = True)
summary_all_df = dataframe_full.groupby('Subtype').mean(numeric_only = True)

#%% Features that differ between ALL groups  or no difference between ANY groups

all_features, none_features, summary_df = utils.find_all_none(dataframe_full, cluster_col="Subtype")
significant_features, molten_dunn_df, dunn_df, group_feature_table, group_summary = utils.kw_dunn_group_diffs(dataframe_full, cluster_col = "Subtype")

# plot insignificant features
import pandas as pd

def keyword_cross_table(feature_list, row_keywords, col_keywords):
    """
    feature_list : list of feature names (strings)
    row_keywords : list of keywords for rows
    col_keywords : list of keywords for columns

    Returns:
        DataFrame where:
            rows = row_keywords
            columns = col_keywords
            values = count of features containing BOTH keywords
    """

    # Initialize empty dataframe
    table = pd.DataFrame(
        0,
        index=row_keywords,
        columns=col_keywords,
        dtype=int
    )

    for r in row_keywords:
        for c in col_keywords:
            count = sum(
                (r.lower() in f.lower()) and (c.lower() in f.lower())
                for f in feature_list
            )
            table.loc[r, c] = count

    return table


results = keyword_cross_table(all_features, constants.ORGANELLE_LIST, constants.FEATURE_TYPE_LIST).reset_index().rename(columns={'index': 'Organelle'})

plot.create_table(results)
#%% example per type

max_representative = 10
pop = 3

org.plot_selected(
    all_segmentation,
    organelle,
    silhouette_score_stats_dict = silhouette_score_stats_dict,
    readfile=False,
    max_display=max_representative,
    population=pop,
    xray_overlay=True,
    xray_paths=xray_paths,
    cropped_roi=False,
    cropped_xray = False, 
)
