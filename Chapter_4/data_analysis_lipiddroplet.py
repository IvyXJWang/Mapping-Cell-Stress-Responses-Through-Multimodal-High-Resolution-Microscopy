# same analysis as mitochondria for thesis
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
    context_classification_lipid,
)
import plotting as plot
from utils import rename_df_columns_with_keyword
import numpy as np
import seaborn as sns
import matplotlib as mpl

# %%
run = "run4"
treatment = "control"
organelle = constants.LIPID_DROPLETS
datadir = constants.PARENTDIR / f"{treatment}/{run}"
segmentationdir = datadir / "results"

filedir = datadir / "cellprofiler" / "measurements_updated"
all_sheets_original = combine_files_to_sheets(filedir)

all_sheets = all_sheets_original.copy()
all_sheets["lipiddroplets"] = utils.update_cellID_objectnum(
    all_sheets["lipiddroplets"],
    "ImageNumber",
    "ObjectNumber",
    "Metadata_CellID",
)

#organelle_dataframe_clean, segmentation_masks, xray_paths, all_sheets_unfiltered = (
   # org.load_organelle_measurements(datadir, organelle, outlier_check=False)
#)

#organelle_dataframe_renamed = rename_df_columns_with_keyword(organelle_dataframe_clean, rename_dict)

#dataset_type_split = org.split_dataset_by_featuretype(organelle_dataframe_renamed)
all_segmentation = load_segmentation(
    segmentationdir, organelle=constants.ORGANELLE_CP_LIST
)

#%% clean lipid droplet dataframe
parentdir = Path(constants.PARENTDIR / f"{treatment}" / f"{run}")
keep_cols_reduced = pd.read_excel(
    parentdir / "selected_features_organelle.xlsx", sheet_name=None, header=None
)

all_sheets_filtered = filter_dataframes(all_sheets, keep_cols_reduced)

lipid_props = all_sheets_filtered["lipiddroplets"].copy()
lipid_props["ObjectID"] = all_sheets["lipiddroplets"]["ObjectID"]

# identify subpopulations of mitochondria
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

lipid_props_renamed = rename_df_columns_with_keyword(lipid_props, rename_dict)
lipid_scaled = utils.scale_data(lipid_props_renamed)
#lipid_scaled = lipid_scaled.drop(labels="Structural_DistanceMitochondria", axis = 1)

#%% normalize all variables [SELECTED]
lipid_props_normalised = lipid_scaled.copy()
lipid_props_normalised_loop = lipid_scaled.copy()
feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]

shape_cols = lipid_props_normalised.columns[lipid_props_normalised.columns.str.contains("Geometric", case=False)]
intensity_cols = lipid_props_normalised.columns[lipid_props_normalised.columns.str.contains("Densitometric", case=False)]
texture_cols = lipid_props_normalised.columns[lipid_props_normalised.columns.str.contains("Textural", case=False)]
structure_cols = lipid_props_normalised.columns[lipid_props_normalised.columns.str.contains("Structural", case=False)]

lipid_props_normalised[shape_cols] /= np.sqrt(len(shape_cols))
lipid_props_normalised[intensity_cols] /= np.sqrt(len(intensity_cols))
lipid_props_normalised[texture_cols] /= np.sqrt(len(texture_cols))
lipid_props_normalised[structure_cols] /= np.sqrt(len(structure_cols))

for feature in feature_types:
    
    cols = lipid_props_normalised.columns[lipid_props_normalised.columns.str.contains(feature, case=False)]
    lipid_props_normalised_loop[cols] /= np.sqrt(len(cols))


#%% cluster lipid droplets 

lipid_labelled, _ = add_cluster_label(
    lipid_props_normalised,
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_,_,cmap_labelling = label_color_mapping_dict(
    lipid_labelled["Subtype"], palette="inferno"
)

_, clustermap_cmap = plot.show_clustermap(
    lipid_labelled, 
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

labels = lipid_labelled["Subtype"]

#%% display overall proportion
order = [1, 2, 3, 4]
proportions = lipid_labelled['Subtype'].value_counts(normalize=True)
proportions = proportions.reindex(order)

colors = [cmap_labelling[cat] for cat in proportions.index]

plt.figure(figsize=(3,4))
proportions.plot(kind='bar', color = colors)
#plt.ylabel('Proportion')
#plt.title('Category Proportions')
plt.xticks([])
plt.xlabel("")
plt.show()
#%% evaluation of clustering
individual_scores_df = silhouette_score_indiv(
    lipid_scaled,
    lipid_labelled["Subtype"],
    scale=False,
    cellid_col="Metadata_CellID",
)
silhouette_score_stats_dict = silhouette_score_stats(individual_scores_df)
counts = lipid_labelled['Subtype'].value_counts()
#%% check dominating features
features_sel, sel = MRMR_dataframe(lipid_scaled, labels, id_col=constants.CELLID_COL, max_features = None)

feature_scores = pd.DataFrame()
feature_scores["Feature"] = sel.variables_
feature_scores["Score"] = sel.relevance_

#%% identify characteristics of subtypes

feat_type = "Structural"

dataframe_full = lipid_props_renamed.copy()
subset_feature_list = [
    x
    for x in dataframe_full.columns
    if (feat_type in x.split("_")[0] and "Zernike" not in x or constants.CELLID_COL in x)
]

if len(subset_feature_list) == 1:
    print("Check feature spelling")

#subset_feature_list = [constants.CELLID_COL, "Structural_OrientationNucleus"]#, "Geometric_AspectRatio"]
dataframe_labelled = dataframe_full[subset_feature_list].copy()

dataframe_labelled.loc[:, "Subtype"] = labels

significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled,
    cluster_col="Subtype",
    cmap=clustermap_cmap,
    axes = True
)

summary_df = dataframe_labelled.groupby('Subtype').mean(numeric_only = True)

# effect size

significant_features, stat_df, dunn_df = utils.kw_dunn_effects(dataframe_labelled, cluster_col = "Subtype")
subset_feature_list_plot = [x for x in subset_feature_list if x != constants.CELLID_COL]

for feature in subset_feature_list_plot:
    table_df = stat_df[stat_df["feature"]==feature]
    table_df = table_df[["pair", "p_adj", "cliffs_delta"]]
    table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
    table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
    table_renamed = table_df.rename(columns={"pair": "Type pair", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })
    
    #plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(3,1.6), title = "")

    plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(4,2), title = f"{feature}")

#%% example lipid droplet per type

max_representative = 20
pop = 2
cluster_num = 1

#for pop in list(range(1, cluster_num)):
    
org.plot_selected(
    segmentation_masks,
    organelle,
    silhouette_score_stats_dict = silhouette_score_stats_dict,
    readfile=False,
    max_display=max_representative,
    population=pop,
    xray_overlay=True,
    xray_paths=xray_paths,
    cropped_roi=True,
    cropped_xray = True
)

#%% distribution of lipid droplet types within cells

# add cellID back
lipid_labelled["CellID"] = all_sheets_original["lipiddroplets"]["Metadata_CellID"]

plot.stacked_bxplt_count(
    lipid_labelled, "CellID", "Subtype", 
    label_color_mapping=cmap_labelling, 
    legend=False, 
    axis_labels = False
)

plot.stacked_bxplt_count(
    lipid_labelled,
    "CellID",
    "Subtype",
    label_color_mapping=cmap_labelling,
    normalize=True,
    legend = False,
    axis_labels = False
)

#%% distribution of lipid droplet types in subcellular regions
classification_df = context_classification_lipid(
    all_sheets_original,
    all_segmentation,
    lipid_props_renamed[constants.CELLID_COL],
)

classification_list = [x for x in classification_df.columns if "Classification" in x]

for cluster_type in classification_list:
    org.stacked_cluster_box_plts_pair(classification_df, "CellID", cluster_type)



#%% distribution of lipid droplet types in subcellular regions (cluster based on spatial localisation)
classification_df = context_classification_lipid(
    all_sheets_original,
    all_segmentation,
    lipid_props["ObjectID"],
    #DEBUG = False
)

classification_list = [x for x in classification_df.columns if "Classification" in x]

#%%
#if not in cell edge or perinuclear then cytoplasm 
classification_df['Classification_Cytoplasm'] = ((classification_df['Classification_Perinuclear'] == 0) & (classification_df['Classification_CellEdge'] == 0)).astype(int)
classification_df['Classification_CellEdge_only'] = ((classification_df['Classification_CellEdge'] == 1) & (classification_df['Classification_Perinuclear'] == 0)).astype(int)
classification_df['Classification_Perinuclear_only'] = ((classification_df['Classification_CellEdge'] == 0) & (classification_df['Classification_Perinuclear'] == 1)).astype(int)
classification_df['Classification_Squished'] = ((classification_df['Classification_CellEdge'] == 1) & (classification_df['Classification_Perinuclear'] == 1)).astype(int)


classification_list = [x for x in classification_df.columns if "Classification" in x]

for cluster_type in classification_list:
    org.stacked_cluster_box_plts_pair(classification_df, "CellID", cluster_type)


#%% subtypes in each classification 
stack_order = [1, 2, 3, 4]
colors = [cmap_labelling[val] for val in stack_order]

classification_plot = classification_df.copy()
classification_plot = classification_df.merge(lipid_labelled[['Metadata_CellID', 'Subtype']],
                                            on='Metadata_CellID',
                                            how='left'
                                            )
#classification_plot = classification_plot[classification_plot["CellID"]=="CELL001"]

cols = ["Classification_Perinuclear_only", "Classification_Perimitochondrial", "Classification_Cytoplasm", "Classification_Cluster", "Classification_CellEdge_only"]#, "Classification_Squished"]

counts = pd.concat({
    col: classification_plot[classification_plot[col] == 1]["Subtype"].value_counts()
    for col in cols
}, axis=1).fillna(0).sort_index()


#counts.T.plot(kind="bar", stacked=True)
#ax = counts.T.plot(kind="bar", stacked=True, figsize=(5, 5), color = colors)

percent = counts.div(counts.sum(axis=0), axis=1)# * 100
ax = percent.T.plot(kind="bar", stacked=True, figsize=(5, 5), color = colors, legend = False)

#ax.set_xlabel("")
#ax.set_xticks([])
#ax.set_ylabel("")
ax.set_ylim(0, 1)
#ax.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()

#%% test enrichment in subtype in location (no enrichment)
import pandas as pd

#classification_plot["is"] = classification_plot["Classification_Perinuclear_only"] == 1 # subtype 4 found not in perinuclear region
#classification_plot["is"] = classification_plot["Classification_cytoplasm"] == 1 # subtype 4 found in cytoplasm region 
#classification_plot["is"] = classification_plot["Classification_Squished"] == 1 
#classification_plot["is"] = classification_plot["Classification_CellEdge_only"] == 1
classification_plot["is"] = classification_plot["Classification_Cluster"] == 1
regions = ["Perinuclear", "Cytoplasm", "Perimitochondrial", "Cluster", "Cell Edge"]


# Contingency table: Subtype vs Edge
contingency = pd.crosstab(
    classification_plot["Subtype"],
    classification_plot["is"]
)

from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(contingency)

print("p-value:", p)
import numpy as np

residuals = (contingency - expected) / np.sqrt(expected)
print(residuals)

#%% plot enrichment
axes = True

classification_plot["Perinuclear"] = classification_plot["Classification_Perinuclear"] == 1 
classification_plot["Perimitochondrial"] = classification_plot["Classification_Perimitochondrial"] == 1
classification_plot["Cytoplasmic"] = classification_plot["Classification_Cytoplasm"] == 1 
classification_plot["Cluster"] = classification_plot["Classification_Cluster"] == 1 
classification_plot["Cell Edge"] = classification_plot["Classification_CellEdge_only"] == 1

regions = ["Perinuclear", "Cytoplasmic",  "Perimitochondrial", "Cluster", "Cell Edge"]
residuals = {}

for region in regions: 
    
    # Contingency table: Subtype vs Edge
    contingency = pd.crosstab(
        classification_plot["Subtype"],
        classification_plot[region]
    )
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    residuals[region] = (contingency - expected) / np.sqrt(expected)

for region, res in residuals.items():
    plt.figure(figsize=(4,4))
    sns.heatmap(res, annot=True, 
                cmap="coolwarm", 
                center=0,
                vmin=-2,
                vmax=2,
                annot_kws={"size": 16, "color":"k"},
                cbar = False
                )
    
    #cbar = ax.collections.colorbar
    #cbar.ax.tick_params(labelsize=10)
    #cbar.set_label("Residuals", fontsize=12)
    
    if axes:
        plt.title(f"Standardized Residuals: {region}")
        plt.xlabel("Region Presence")
        plt.ylabel("Subtype")
    
    else:
        print(region)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        
    plt.show()

# scale bar only
fig, ax = plt.subplots(figsize=(0.5, 8))

norm = mpl.colors.Normalize(vmin=-2, vmax=2)
cmap = mpl.cm.coolwarm

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax
)

#cbar.set_label("Standardized Residuals", fontsize=12)
cbar.ax.tick_params(labelsize=16)

plt.show()

#%% adjusted residuals
axes = False

classification_plot["Perinuclear"] = classification_plot["Classification_Perinuclear"] == 1 
classification_plot["Perimitochondrial"] = classification_plot["Classification_Perimitochondrial"] == 1
classification_plot["Cytoplasmic"] = classification_plot["Classification_Cytoplasm"] == 1 
classification_plot["Cluster"] = classification_plot["Classification_Cluster"] == 1 
classification_plot["Cell Edge"] = classification_plot["Classification_CellEdge_only"] == 1

regions = ["Perinuclear", "Cytoplasmic",  "Perimitochondrial", "Cluster", "Cell Edge"]
adjusted_residuals = {}

for region in regions:
    contingency = pd.crosstab(
        classification_plot["Subtype"],
        classification_plot[region]
    )

    chi2, p, dof, expected = chi2_contingency(contingency)

    expected = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns
    )

    # proportions
    row_sums = contingency.sum(axis=1).values.reshape(-1, 1)
    col_sums = contingency.sum(axis=0).values.reshape(1, -1)
    total = contingency.values.sum()

    row_prop = row_sums / total
    col_prop = col_sums / total

    adj_res = (contingency - expected) / np.sqrt(
        expected * (1 - row_prop) * (1 - col_prop)
    )

    adjusted_residuals[region] = adj_res

for region, res in adjusted_residuals.items():
    plt.figure(figsize=(4,4))
    sns.heatmap(res, annot=True, 
                cmap="coolwarm", 
                center=0,
                vmin=-2,
                vmax=2,
                annot_kws={"size": 16, "color":"k"},
                cbar = False
                )
    
    #cbar = ax.collections.colorbar
    #cbar.ax.tick_params(labelsize=10)
    #cbar.set_label("Residuals", fontsize=12)
    
    if axes:
        plt.title(f"Standardized Residuals: {region}")
        plt.xlabel("Region Presence")
        plt.ylabel("Subtype")
    
    else:
        print(region)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        
    plt.show()

# scale bar only
fig, ax = plt.subplots(figsize=(0.5, 8))

norm = mpl.colors.Normalize(vmin=-2, vmax=2)
cmap = mpl.cm.coolwarm

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax
)

#cbar.set_label("Standardized Residuals", fontsize=12)
cbar.ax.tick_params(labelsize=16)

plt.show()
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(8, 0.5))

norm = mpl.colors.Normalize(vmin=-2, vmax=2)
cmap = mpl.cm.coolwarm

cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation="horizontal"
)

# Label + font control
#cbar.set_label("Standardized Residuals", fontsize=12)
cbar.ax.tick_params(labelsize=16)

# Optional: cleaner ticks
cbar.set_ticks([-2, -1, 0, 1, 2])

plt.show()
#%%
summary_df = pd.DataFrame({
    "Type": [1,2,3,4],
    "Perinuclear": [1,0,-1,0],
    "Cell Edge": [-1,-1,1,0],
    "Cluster": [0,1,-1,0],
    "Cytoplasm": [-1,0,1,0],
    })

plot.create_table(summary_df, body_fontsize=6, header_fontsize=6, fig_size=(4,2))