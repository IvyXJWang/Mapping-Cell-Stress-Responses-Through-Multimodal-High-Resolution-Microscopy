import data_analysis_utils as utils
import data_analysis_organelle_utils as org
from utils import rename_df_columns_with_keyword
from feature_extraction_utils import combine_files_to_sheets, load_segmentation

import pandas as pd
from pathlib import Path
from classification import (
    HCA_top_features,
    filter_dataframes,
    silhouette_score_indiv,
    silhouette_score_stats,
)

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

import plotting as plot
import classification as clus

from pathlib import Path
import constants

# from k_means_clustering_ver1 import KMeans_2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from segmentation_utils import load_path_into_dict

#%%
def clr_transform(df, eps=1e-9):
    x = df.to_numpy(dtype=float)
    x = np.clip(x, eps, None)
    clr = np.log(x) - np.log(x).mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=df.index, columns=df.columns)

# %% dataset prep
treatment = "control"
version = "run4"

# Input: directory containing all the raw xray images, cell outline, .npz probability files
parentdir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
segmentationdir = parentdir / "results"
xray_paths = load_path_into_dict(segmentationdir, keyword="xray")

# Output: directory to save images / processed spreadsheets
figdir = parentdir / "figures"
Path(figdir).mkdir(parents=True, exist_ok=True)
resultsdir = parentdir / "summary_sheets"
Path(resultsdir).mkdir(parents=True, exist_ok=True)

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

# load in manually selected features
filedir = parentdir / "cellprofiler" / "measurements_updated"
all_sheets = combine_files_to_sheets(filedir)

keep_cols_reduced = pd.read_excel(
    parentdir / "selected_features_organelle.xlsx", sheet_name=None, header=None
)

all_sheets_filtered = filter_dataframes(all_sheets, keep_cols_reduced)

all_segmentation = load_segmentation(
    segmentationdir, organelle=constants.ORGANELLE_CP_LIST
)

#%% original dataframe
HCA_file = resultsdir / "HCA.xlsx"

if HCA_file.exists():
    cell_summary = pd.read_excel(
       HCA_file)
else:
    print("Creating HCA and summary files")
    cell_summary, novar = clus.prepare_HCA_sheet(
        all_sheets_filtered,
        keep_cols_reduced,
        savedir=resultsdir,
        variable_features=False,
        filter_col=True,
    )

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

cell_summary_renamed = rename_df_columns_with_keyword(cell_summary, rename_dict)

#%% already have updated object name in dataframe from updating measures
all_sheets = all_sheets_original.copy()
all_sheets["lipiddroplets"] = utils.update_cellID_objectnum(
    all_sheets["lipiddroplets"],
    "ImageNumber",
    "ObjectNumber",
    "Metadata_CellID",
)

all_sheets["mitochondria"] = utils.update_cellID_objectnum(
    all_sheets["mitochondria"],
    "ImageNumber",
    "ObjectNumber",
    "Metadata_CellID",
)

#%% create new cluster dataframe = lipid droplets

lipid_props = all_sheets_filtered["lipiddroplets"]
lipid_props_renamed = rename_df_columns_with_keyword(lipid_props, rename_dict)
#%% create new cluster dataframe - lipid

lipid_props = all_sheets_filtered["lipiddroplets"]
lipid_props_renamed = rename_df_columns_with_keyword(lipid_props, rename_dict)
lipid_scaled = utils.scale_data(lipid_props_renamed)

lipid_props_normalised = lipid_scaled.copy()
feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]
for feature in feature_types:
    cols = lipid_props_normalised.columns[lipid_props_normalised.columns.str.contains(feature, case=False)]
    lipid_props_normalised[cols] /= np.sqrt(len(cols))

lipid_labelled, _ = add_cluster_label(
    lipid_props_normalised,
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

# convert subtype composition per cell into cell measures
lipid_type = (
    lipid_labelled
    .groupby('Metadata_CellID')['Subtype']
    .value_counts(normalize=True)   # gives proportions instead of counts
    .unstack(fill_value=0)         # columns = types (1,2,3,4)
    .add_prefix("lipiddroplets_")
    .reset_index()
)


comp_id = lipid_type[["Metadata_CellID"]].copy()
comp_props = lipid_type.drop(columns=["Metadata_CellID"])

# CLR transform on proportions
comp_props_clr = clr_transform(comp_props)

# optional: standardize after CLR
comp_props_clr_scaled = pd.DataFrame(
    StandardScaler().fit_transform(comp_props_clr),
    index=comp_props_clr.index,
    columns=comp_props_clr.columns
)

# combine back
lipid_clr = pd.concat([comp_id, comp_props_clr_scaled], axis=1)
cluster_df = lipid_clr.copy()

# create new cluster dataframe = mitochondria

mito_props = all_sheets_filtered["mitochondria"]
mito_props_renamed = rename_df_columns_with_keyword(mito_props, rename_dict)
mito_scaled = utils.scale_data(mito_props_renamed)

mito_props_normalised = mito_scaled.copy()
feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]
for feature in feature_types:
    cols = mito_props_normalised.columns[mito_props_normalised.columns.str.contains(feature, case=False)]
    mito_props_normalised[cols] /= np.sqrt(len(cols))

mito_labelled, _ = add_cluster_label(
    mito_props_normalised,
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

# convert subtype composition per cell into cell measures
mito_type = (
    mito_labelled
    .groupby('Metadata_CellID')['Subtype']
    .value_counts(normalize=True)   # gives proportions instead of counts
    .unstack(fill_value=0)          # columns = types (1,2,3,4)
    .add_prefix("mitochondria_")
    .reset_index()
)

comp_id = mito_type[["Metadata_CellID"]].copy()
comp_props = mito_type.drop(columns=["Metadata_CellID"])

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

cluster_df = pd.merge(
                cluster_df, mito_clr, on="Metadata_CellID", how="outer"
            )

#%% reduce dimensions by clustering nucleus, cell and cytoplasm and network
plot_cluster = False

cluster_df_full = cluster_df.copy()
individual_props = {"cell": 3, "cytoplasm": 3, "nucleus": 5, "network": 2}
individual_props = {"cell": 3, "cytoplasm": 3, "nucleus": 5}

for individual, clusternum in individual_props.items():
    props = all_sheets_filtered[individual]
    
    if individual == "network": 
        scaled = props
    else:
        props_renamed = rename_df_columns_with_keyword(props, rename_dict)
        scaled = utils.scale_data(props_renamed)
    
    props_normalised = scaled.copy()
    feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]
    for feature in feature_types:
        cols = props_normalised.columns[props_normalised.columns.str.contains(feature, case=False)]
        props_normalised[cols] /= np.sqrt(len(cols))
        
    labelled, _ = add_cluster_label(
        props_normalised,
        cluster_num=clusternum,
        clustering_algorithm="HCA",
        id_col=constants.CELLID_COL,
        lbl_colname="Subtype",
    )
    
    if plot_cluster:
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
            all_legends = False
        )
     
    types = (
        labelled
        .groupby('Metadata_CellID')['Subtype']
        .value_counts(normalize=True)   # gives proportions instead of counts
        .unstack(fill_value=0)          # columns = types (1,2,3,4)
        .add_prefix(f"{individual}_")
        .reset_index()
    )
    
    cluster_df_full = pd.merge(
                    cluster_df_full, types, on="Metadata_CellID", how="outer"
                )

#%% add in other features
organelle_cell_measures = ["mitochondria", "lipiddroplets", "cell"]

for organelle in organelle_cell_measures:
    
    keep_cols_cell = {}
    sheet = {}
    keep_cols_cell[f"{organelle}"] = keep_cols_reduced[f"whole_cell_{organelle}"]
    sheet[f"{organelle}"] = all_sheets[f"{organelle}"]
    filtered = filter_dataframes(sheet, keep_cols_cell)
    
    #normalize by network measures
    scaled = utils.scale_data(filtered[f"{organelle}"])
    
    if organelle == "mitochondria":
        mito_props_network_corrected = scaled.copy()
        network = mito_props_network_corrected.columns[mito_props_network_corrected.columns.str.contains("Network|Branch", case=False)]
        mito_props_network_corrected[network] /= np.sqrt(len(network)) 
        mito_props_network_corrected = (
            mito_props_network_corrected
            .drop_duplicates(subset="Metadata_CellID")
            .rename(columns=lambda c: c if c == "Metadata_CellID" else f" mitochondria_{c}")
            )
        
        cluster_df_full = pd.merge(
                        cluster_df_full, mito_props_network_corrected, on="Metadata_CellID", how="outer"
                    )
        
    elif organelle == "lipiddroplets":
        lipid_corrected = scaled.copy()
        lipid_corrected = (
            lipid_corrected
            .drop_duplicates(subset="Metadata_CellID")
            .rename(columns=lambda c: c if c == "Metadata_CellID" else f"lipiddroplets_{c}")
            )

        cluster_df_full = pd.merge(
                        cluster_df_full, lipid_corrected, on="Metadata_CellID", how="outer")
        
    elif organelle == "cell":
        cell_corrected = scaled.copy().rename(columns=lambda c: c if c == "Metadata_CellID" else f"cell_{c}")
        cluster_df_full = pd.merge(
                        cluster_df_full, cell_corrected, on="Metadata_CellID", how="outer")

cluster_df_full.to_excel(parentdir / "cluster_df_reduced.xlsx", index = False)
#%% cluster on this dataframe

cluster_df_labelled, _ = add_cluster_label(
    cluster_df_full,
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_,_,cmap_labelling = label_color_mapping_dict(
    cluster_df_labelled["Subtype"], palette="inferno"
)

_, clustermap_cmap = plot.show_clustermap(
    cluster_df_labelled, 
    color_row_cmap=cmap_labelling, 
    clustering_algorithm = "ward",
    color_feat=None, 
    label_col="Subtype", 
    id_col=constants.CELLID_COL, 
    scale = False, 
    pltsize=(25,25), 
    score_range=(-3,3),
    cmap = "viridis",
    all_legends = False
)

labels = cluster_df_labelled["Subtype"]
#%% Identify CellID belonging to each subtype

subtypes = list(np.unique(cluster_df_labelled["Subtype"]))

subpopulation_cells = {}
for i in subtypes:  # start subpopulation count at 1
    subpopulation_cells["Subtype" + str(i)] = cluster_df_labelled.loc[
        cluster_df_labelled["Subtype"] == i, constants.CELLID_COL
    ].tolist()
    
#%%  MRMR top features (unscaled dataset same as scaled dataset)
features_sel, sel = clus.MRMR_dataframe(cell_summary_renamed, cluster_df_labelled["Subtype"], id_col=constants.CELLID_COL, max_features = None)

feature_scores = pd.DataFrame()
feature_scores["Feature"] = sel.variables_
feature_scores["Score"] = sel.relevance_

#%% plot features selected by elbow method 
feature_scores.reset_index()
feature_scores_thresholded = feature_scores[
    feature_scores["Feature"].isin(features_sel)
]
feature_scores_thresholded = feature_scores_thresholded.sort_values(
    by="Score",
    ascending=False   # change to True if you want lowest first
)

colors = feature_scores_thresholded["Feature"].apply(plot.get_color)

plt.figure(figsize=(50, 50))
plt.bar(feature_scores_thresholded["Feature"], feature_scores_thresholded["Score"], color=colors, width=1)
plt.xticks([])
plt.yticks(fontsize = 100)

ax = plt.gca()

# Modify outline
for spine in ax.spines.values():
    spine.set_linewidth(8)   # thickness
    spine.set_color("black")  # color

ax.tick_params(axis="y", labelsize=125)

plt.tight_layout()
plt.show()

#%% Average score per organelle
top_only = True

if top_only:
    feature_scores_thresholded = feature_scores_thresholded
else:
    feature_scores_thresholded = feature_scores.copy()

feature_score_organelle = {}
for organelle in constants.ORGANELLE_LIST:
    feature_score_organelle[organelle] = feature_scores_thresholded.loc[
        feature_scores_thresholded["Feature"].str.contains(organelle, case=False, na=False),
        "Score"
    ]

average_contribution_organelle = {}
total_contribution_organelle = {}

for organelle, scores in feature_score_organelle.items():
    total_contribution_organelle[organelle] = scores.sum()
    average_contribution_organelle[organelle] = scores.mean()

contribution_df = pd.DataFrame({
    'Organelle': average_contribution_organelle.keys(),
    'Average MRMR Score': average_contribution_organelle.values(),
    'Total MRMR Score': total_contribution_organelle.values()
})

# round MRMR scores
contribution_df[['Average MRMR Score', 'Total MRMR Score']] = contribution_df[['Average MRMR Score', 'Total MRMR Score']].round(2)

plot.create_table(contribution_df, body_fontsize=6, header_fontsize=6, fig_size=(4,2))

#%% evaluation of clustering
individual_scores_df = silhouette_score_indiv(
    cluster_df_full,
    cluster_df_labelled["Subtype"],
    scale=False,
    cellid_col="Metadata_CellID",
)
silhouette_score_stats_dict = silhouette_score_stats(individual_scores_df)
counts = cluster_df_labelled['Subtype'].value_counts()

#%% display overall proportion
order = [1, 2, 3, 4]
proportions = cluster_df_labelled['Subtype'].value_counts(normalize=True)
proportions = proportions.reindex(order)

colors = [cmap_labelling[cat] for cat in proportions.index]

plt.figure(figsize=(3,4))
proportions.plot(kind='bar', color = colors)
#plt.ylabel('Proportion')
#plt.title('Category Proportions')
plt.xticks([])
plt.xlabel("")
plt.show()
#%% interpret differences - CLR (heatmap proportion enrichment)

cb_palette = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999"   # grey
]


organelle_prop = lipid_type.copy()
#organelle_prop = mito_type.copy()
organelle_prop = cluster_df_full.loc[:, 
    cluster_df_full.columns.str.contains('nucleus|Metadata', case=False) &
    ~cluster_df_full.columns.str.contains('structure', case=False)
]

organelle_prop = organelle_prop.drop(columns=["Metadata_CellID"])
type_list = list(organelle_prop.columns)

organelle_prop["Subtype"] = cluster_df_labelled["Subtype"]
cluster_summary = organelle_prop.groupby("Subtype")[
    type_list
].mean()
palette = sns.color_palette("colorblind", n_colors=len(cluster_summary.columns))

cluster_summary.plot(
    figsize = (6,6),
    kind="bar",
    stacked=True,
    #color=cb_palette[:len(cluster_summary.columns)],
    color = palette,
    legend = False

)
plt.xlabel("")
plt.xticks([])

#%% heatmap proportion enrichment
plt.ylabel("Mean proportion")
plt.title("Cluster composition")
plt.legend(title="Mito type")
plt.tight_layout()
plt.show()

clr_summary = cluster_df_labelled.groupby("Subtype")[[
    "mitochondria_1", "mitochondria_2", "mitochondria_3", "mitochondria_4"
]].mean()

sns.heatmap(clr_summary, annot=True, center=0, cmap="coolwarm")
plt.title("CLR enrichment by cluster")
plt.show()

#%% identify characteristics of subtypes
convert_distance = False
filtered = {}
organelle = "lipiddroplets"
keep_cols_cell[f"{organelle}"] = keep_cols_reduced[f"whole_cell_{organelle}"]
sheet[f"{organelle}"] = all_sheets[f"{organelle}"]
filtered = filter_dataframes(sheet, keep_cols_cell)

dataframe = filtered[organelle].drop_duplicates(subset="Metadata_CellID").reset_index(drop = True)


feat_type = "Structure"

dataframe_full = cluster_df_full.copy()
dataframe_full = dataframe.copy()

#dataframe_full = cell_summary_renamed.copy()
subset_feature_list = [
    x
    for x in dataframe_full.columns
    if (feat_type in x.split("_")[0] or constants.CELLID_COL in x)
]

'''
subset_feature_list = [
    x
    for x in dataframe_full.columns
    if (feat_type in "_".join(x.split("_")[0:2]) or constants.CELLID_COL in x)
]
'''
if len(subset_feature_list) == 1:
    print("Check feature spelling")

#subset_feature_list = [constants.CELLID_COL, "Geometric_FormFactor"]#, "Geometric_AspectRatio"]
#subset_feature_list = [constants.CELLID_COL, "Geometric_FormFactor"]
#subset_feature_list = [constants.CELLID_COL, "Densitometric_MeanDensitometric_xray"]
#subset_feature_list = [constants.CELLID_COL, "Textural_Contrast_xray_avg"]
#subset_feature_list = [constants.CELLID_COL, "Structural_DistanceNucleus"]

subset_feature_list = [constants.CELLID_COL, "Structure_ClusterNumber"]

dataframe_labelled = dataframe_full[subset_feature_list].copy()

# convert measurements to physical units
if convert_distance: 
    dataframe_labelled["Structure_PerinuclearObjectDensity"] = dataframe_labelled["Structure_PerinuclearObjectDensity"] / (constants.PX_SIZE ** 2)

dataframe_labelled.loc[:, "Subtype"] = cluster_df_labelled["Subtype"]

significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled,
    cluster_col="Subtype",
    cmap=clustermap_cmap,
    axes = False
    
)
summary_df = dataframe_labelled.groupby('Subtype').mean(numeric_only = True)

#%% Features that differ between ALL groups  or no difference between ANY groups
dataframe_labelled = cluster_df_full.copy() 
#dataframe_labelled = cell_summary.copy()
dataframe_labelled.loc[:, "Subtype"] = labels

all_features, none_features, summary_df = utils.find_all_none(dataframe_labelled, cluster_col="Subtype")

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


results = keyword_cross_table(none_features, constants.ORGANELLE_LIST, constants.FEATURE_TYPE_LIST).reset_index().rename(columns={'index': 'Organelle'})

plot.create_table(results)

_, _, _, group_feature_table_org, group_summary_org = utils.kw_dunn_group_diffs(
    cluster_df_labelled, cluster_col="Subtype"
)

single_group = group_feature_table_org.xs(1, level='group')
#%% count feature differences from targets
import numpy as np
# selected subtype (can be int or string)
subtype = 1

def difference_counts(df, df_labelled, subtype_label, target = "all"):
    # prepare empty result DataFrame: index = organelles, columns = feature types
    result_df = pd.DataFrame(
        0,
        index=constants.ORGANELLE_LIST,
        columns=constants.FEATURE_TYPE_LIST,
        dtype=int,
    )
    
    # helper to find the actual key present in group_summary (robust to str/int mismatch)
    def find_group_key(gs_dict, desired_key):
        if desired_key in gs_dict:
            return desired_key
        if str(desired_key) in gs_dict:
            return str(desired_key)
        # try numeric match (if keys are numeric-like strings)
        try:
            desired_int = int(desired_key)
        except Exception:
            desired_int = None
        if desired_int is not None:
            for k in gs_dict:
                try:
                    if int(k) == desired_int:
                        return k
                except Exception:
                    continue
        return None
    
    # main loop
    for organelle in constants.ORGANELLE_LIST:
        for feature_type in constants.FEATURE_TYPE_LIST:
            # subset features for this organelle & feature_type
            dataframe_full = df.copy()
            subset_feature_list = [
                x
                for x in dataframe_full.columns
                if (organelle in x and feature_type in x) or constants.CELLID_COL in x
            ]
            if not subset_feature_list:
                # nothing to analyze for this combination; leave 0
                continue
    
            dataframe_labelled = dataframe_full[subset_feature_list].copy()
            # make sure you use the correct source for Subtype labels
            dataframe_labelled.loc[:, "Subtype"] = df_labelled["Subtype"]
    
            # run the analysis and store per-organelle results (as you intended)
            _, _, _, group_feature_table_org, group_summary_org = utils.kw_dunn_group_diffs(
                dataframe_labelled, cluster_col="Subtype"
            )
            group_key = find_group_key(group_summary_org, subtype_label)
            if group_key is None:
                # no such subtype present; leave zeros
                continue
            
            group_rows = group_feature_table_org.xs(group_key, level="group")

            # group_summary_org is a dict: group_label -> {"differs_from_both": [...], ...}
            key = find_group_key(group_summary_org, subtype_label)
            if key is None:
                cnt = 0
            elif target == "all":
                # safe access in case structure differs
                cnt = len(group_summary_org.get(key, {}).get("differs_from_both", []))
            elif target in np.unique(dataframe_labelled["Subtype"]):
                t_str = str(target)

                cnt = (
                    group_rows.apply(
                        lambda row: (
                            row["category"] == "differs_from_exactly_one"
                            and any(str(p) == t_str for p in row["partners"])
                        ),
                        axis=1
                    ).sum()
                )
            elif target == "none":
                cnt = len(group_summary_org.get(key, {}).get("differs_from_none", []))
    
            result_df.at[organelle, feature_type] = cnt
    
    # optional: reset index to have Organelle column rather than index
    result_df = result_df.reset_index().rename(columns={"index": "Organelle"})
    
    return result_df

result_df = difference_counts(cluster_df_full, cluster_df_labelled, 1, target = 3)
plot.create_table(result_df, body_fontsize=8, header_fontsize=10)
#%% example cell per type

max_representative = 10
pop = 2
cluster_num = 4
organelle = "cell"
#for pop in list(range(1, cluster_num)):
    
org.plot_selected(
    all_segmentation,
    organelle,
    silhouette_score_stats_dict = silhouette_score_stats_dict,
    readfile=False,
    max_display=max_representative,
    population=pop,
    xray_overlay=True,
    xray_paths=xray_paths,
    cropped_roi=True,
    cropped_xray = False
)

#%% feature correlation - correlation matrix (different x and y columns)

corr = cluster_df_full.copy()
# define your feature groups
x_cols = ['mitochondria_1', 'mitochondria_2', 'mitochondria_3', 'mitochondria_4']
y_cols = ['lipiddroplets_1', 'lipiddroplets_2', 'lipiddroplets_3', 'lipiddroplets_4']

# create empty dataframe for results
corr_matrix = pd.DataFrame(index=y_cols, columns=x_cols)

# compute correlations
for y in y_cols:
    for x in x_cols:
        corr_matrix.loc[y, x] = corr[y].corr(corr[x])

# convert to float
corr_matrix = corr_matrix.astype(float)

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.show()

#%% feature correlation - all 

correlation_df = cell_summary_renamed.drop(columns = "Metadata_CellID")
corr = correlation_df.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.show()