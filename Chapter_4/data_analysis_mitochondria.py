# mitochondria organelle analysis

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
    label_color_mapping_dict,
    random_forest_train_model
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
organelle = constants.MITOCHONDRIA
datadir = constants.PARENTDIR / f"{treatment}/{run}"
segmentationdir = datadir / "results"

filedir = datadir / "cellprofiler" / "measurements_updated"
all_sheets_original = combine_files_to_sheets(filedir)

all_sheets = all_sheets_original.copy()
all_sheets["mitochondria"] = utils.update_cellID_objectnum(
    all_sheets["mitochondria"],
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


#%% clean mitochondria dataframe
parentdir = Path(constants.PARENTDIR / f"{treatment}" / f"{run}")
keep_cols_reduced = pd.read_excel(
    parentdir / "selected_features_organelle.xlsx", sheet_name=None, header=None
)

all_sheets_filtered = filter_dataframes(all_sheets, keep_cols_reduced)


mito_props = all_sheets_filtered["mitochondria"]

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

mito_props_renamed = rename_df_columns_with_keyword(mito_props, rename_dict)
mito_scaled = utils.scale_data(mito_props_renamed)



#%% PCA dimension reduction - loses all texture measurements
import itertools
import classification as clus

feature_types_name = ["Geometric_Zernike", "Densitometric_Zernike"]

feature_type_split = {
        keyword: mito_scaled[[constants.CELLID_COL] + 
                    [col for col in mito_scaled.columns if keyword in col]]
        for keyword in feature_types_name
    }

PCA_feature_type = pd.DataFrame(
    {constants.CELLID_COL : mito_scaled[constants.CELLID_COL]
     }
    )

contributions = {}
explained_variance = {}
for feature_name, df in feature_type_split.items():
    
    if len(df.columns) > 1:
        
        df_scaled = utils.scale_data(df)
        #n_cluster = clus.PCA_n_components(df, percent_threshold=0.5)
        
        (fig,
        principal_df,
        pca,
        loadings_matrix,
        combined_df,
        colors,
        label_names,
        label_color_mapping) = clus.PCA_components(
            df_scaled, 
            n_PCA_components = 10, 
            #n_PCA_components = n_cluster,
            cluster_num = 1, 
            id_col = constants.CELLID_COL, 
            plot = True, 
            cluster_method = "HCA")
        
        principal_df = principal_df.add_prefix(f"{feature_name}_")
        contributions[feature_name] = loadings_matrix
        explained_variance[feature_name] = pca.explained_variance_ratio_
        
        PCA_feature_type = pd.concat([PCA_feature_type, principal_df], axis = 1)

#PCA_feature_type_scaled = utils.scale_data(PCA_feature_type) # do not scale if scaled before PCA

PCA_feature_type_labelled, _ = clus.add_cluster_label(
    PCA_feature_type,
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_, _, cmap_labelling = clus.label_color_mapping_dict(
    PCA_feature_type_labelled["Subtype"], palette="inferno"
)
#cmap_labelling = {str(k): v for k, v in cmap_labelling.items()}
combined_df[constants.CELLID_COL] = mito_props_renamed[constants.CELLID_COL]

_, clustermap_cmap = plot.show_clustermap(
    PCA_feature_type_labelled, color_row_cmap=None, color_feat="type", 
    label_col="Subtype", id_col=constants.CELLID_COL, 
    scale = False, pltsize=(30,30)
)

labels = PCA_feature_type_labelled["Subtype"]

#%% Z-score averaging 
#%% change organelle weighting (normalize all features to feature number per feature type)

all_cols = mito_props_renamed.columns.drop(constants.CELLID_COL)
base_names = all_cols.str.split('_').str[0]

cell_summary_scaled_renamed = rename_df_columns_with_keyword(mito_scaled, rename_dict)
cell_summary_scaled_weighted = cell_summary_scaled_renamed.copy()

for base in base_names:
    # Find all columns that belong to this base feature
    matching_cols = cell_summary_scaled_renamed.columns[cell_summary_scaled_renamed.columns.str.startswith(base + '_')]
    
    # Number of stats for this base feature
    n_stats = len(matching_cols)
    #print(n_stats)
    # Apply weight
    cell_summary_scaled_weighted[matching_cols] *= (1 / n_stats)

#a = cell_summary_scaled_renamed["LipidDroplets_Structural_NearestNeighbourDistance_mean"].std()
#aa = cell_summary_scaled_weighted["LipidDroplets_Structural_NearestNeighbourDistance_mean"].std()

cell_summary_weighted_labelled, _ = clus.add_cluster_label(
    cell_summary_scaled_weighted,
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_, clustermap_cmap = plot.show_clustermap(
    cell_summary_weighted_labelled, 
    color_row_cmap=cmap_labelling, 
    color_feat="organelle", 
    label_col="Subtype", 
    id_col=constants.CELLID_COL, 
    scale = False, 
    pltsize=(60,30), 
    score_range=None
)
labels = cell_summary_weighted_labelled["Subtype"]

#%% normalize Zernike
import numpy as np
mito_props_zernike_corrected = mito_scaled.copy()

# 2. Select Zernike columns (adjust pattern if needed)
shape_cols = mito_props_zernike_corrected.columns[mito_props_zernike_corrected.columns.str.contains("Geometric_Zernike", case=False)]
intensity_cols = mito_props_zernike_corrected.columns[mito_props_zernike_corrected.columns.str.contains("Densitometric_Zernike", case=False)]

# 3. Scale Zernike block by sqrt(d)
mito_props_zernike_corrected[shape_cols] /= np.sqrt(len(shape_cols))
mito_props_zernike_corrected[intensity_cols] /= np.sqrt(len(intensity_cols))

#%% normalize all variables [SELECTED]
mito_props_zernike_corrected = mito_scaled.copy()

shape_cols = mito_props_zernike_corrected.columns[mito_props_zernike_corrected.columns.str.contains("Geometric", case=False)]
intensity_cols = mito_props_zernike_corrected.columns[mito_props_zernike_corrected.columns.str.contains("Densitometric", case=False)]
texture_cols = mito_props_zernike_corrected.columns[mito_props_zernike_corrected.columns.str.contains("Textural", case=False)]
structure_cols = mito_props_zernike_corrected.columns[mito_props_zernike_corrected.columns.str.contains("Structural", case=False)]

mito_props_zernike_corrected[shape_cols] /= np.sqrt(len(shape_cols))
mito_props_zernike_corrected[intensity_cols] /= np.sqrt(len(intensity_cols))
mito_props_zernike_corrected[texture_cols] /= np.sqrt(len(texture_cols))
mito_props_zernike_corrected[structure_cols] /= np.sqrt(len(structure_cols))

#%% cluster mitochondria 

mito_labelled, _ = add_cluster_label(
    mito_props_zernike_corrected,
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_,_,cmap_labelling = label_color_mapping_dict(
    mito_labelled["Subtype"], palette="inferno"
)

_, clustermap_cmap = plot.show_clustermap(
    mito_labelled, 
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

labels = mito_labelled["Subtype"]

#%% random forest classification from label
mito_dataset = mito_props_zernike_corrected.copy()
mito_dataset["Subtype"] = labels

rf_model, data_scaler, scores, scores_mean, scores_std, feature_importance = (
    random_forest_train_model(
        mito_dataset, "Subtype", savedir=datadir, scale = False
    )
)

#%% display overall proportion
order = [1, 2, 3, 4]
proportions = mito_labelled['Subtype'].value_counts(normalize=True)
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
    mito_scaled,
    mito_labelled["Subtype"],
    scale=False,
    cellid_col="Metadata_CellID",
)
silhouette_score_stats_dict = silhouette_score_stats(individual_scores_df)
counts = mito_labelled['Subtype'].value_counts()
#%% check dominating features
features_sel, sel = MRMR_dataframe(mito_props_renamed, labels, id_col=constants.CELLID_COL, max_features = None)

feature_scores = pd.DataFrame()
feature_scores["Feature"] = sel.variables_
feature_scores["Score"] = sel.relevance_

#%% plot all MRMR scores
feature_scores = feature_scores.sort_values(
    by="Score",
    ascending=False   # change to True if you want lowest first
)


colors = feature_scores["Feature"].apply(
    lambda f: plot.get_color(f, color_by="feature_type")
)
plt.figure(figsize=(10, 5))
plt.bar(feature_scores["Feature"], feature_scores["Score"], color=colors),
plt.xticks([])
plt.tight_layout()
plt.show()

#%% plot sorted MRMR to determine threshold (line plot)
MRMR_score_threshold = 440
import matplotlib.ticker as ticker

ax = pd.Series(sel.relevance_, index=sel.variables_).sort_values(
    ascending=False).plot.line(figsize=(6, 6), color = "black")

ax.yaxis.set_major_locator(ticker.MultipleLocator(100))   # adjust spacing
ax.yaxis.set_minor_locator(ticker.MultipleLocator(50))  # optional finer ticks
ax.tick_params(axis='y', which='major', length=6)
ax.tick_params(axis='y', which='minor', length=3)

ax.minorticks_on()  # ensure minor ticks are enabled

ax.grid(which='major', axis='y', linewidth=0.8, linestyle = '--')
ax.grid(which='minor', axis='y', linewidth=0.4, linestyle=':')

for spine in ax.spines.values():
    spine.set_linewidth(0.5)      # thickness
    spine.set_color("black")    # color

#ax.spines["top"].set_visible(False)
#ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=15)
plt.axhline(y=MRMR_score_threshold, color='red', linestyle='--', linewidth=2)

plt.xticks([])
plt.show()

print(f"Features selected: {len(feature_scores[feature_scores["Score"] > MRMR_score_threshold])}"
)

#%% plot features selected by elbow method 
feature_scores.reset_index()
feature_scores_thresholded = feature_scores[feature_scores["Score"] > MRMR_score_threshold]
feature_scores_thresholded = feature_scores_thresholded.sort_values(
    by="Score",
    ascending=False   # change to True if you want lowest first
)

colors = feature_scores["Feature"].apply(
    lambda f: plot.get_color(f, color_by="feature_type")
)
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

#%% Average score per feature type 

top_only = True

if top_only:
    feature_scores_thresholded = feature_scores[feature_scores["Score"] > MRMR_score_threshold]
else:
    feature_scores_thresholded = feature_scores.copy()

feature_score_type = {}
for feat_type in constants.FEATURE_TYPE_LIST:
    feature_score_type[feat_type] = feature_scores_thresholded.loc[
        feature_scores_thresholded["Feature"].str.contains(feat_type, case=False, na=False),
        "Score"
    ]

average_contribution_type = {}
total_contribution_type = {}

for feat_type, scores in feature_score_type.items():
    total_contribution_type[feat_type] = scores.sum()
    average_contribution_type[feat_type] = scores.mean()

contribution_df = pd.DataFrame({
    'Type': average_contribution_type.keys(),
    'Average MRMR Score': average_contribution_type.values(),
    'Total MRMR Score': total_contribution_type.values()
})

# round MRMR scores
contribution_df[['Average MRMR Score', 'Total MRMR Score']] = contribution_df[['Average MRMR Score', 'Total MRMR Score']].round(2)

plot.create_table(contribution_df, body_fontsize=6, header_fontsize=6, fig_size=(4,2))
#%% identify characteristics of subtypes

feat_type = "Structural"

dataframe_full = mito_props_renamed.copy()
subset_feature_list = [
    x
    for x in dataframe_full.columns
    if (feat_type in x.split("_")[0] and "Zernike" not in x or constants.CELLID_COL in x)
]

if len(subset_feature_list) == 1:
    print("Check feature spelling")

#subset_feature_list = [constants.CELLID_COL, "Geometric_FormFactor"]#, "Geometric_AspectRatio"]
#subset_feature_list = [constants.CELLID_COL, "Geometric_FormFactor"]
#subset_feature_list = [constants.CELLID_COL, "Densitometric_MeanDensitometric_xray"]
#subset_feature_list = [constants.CELLID_COL, "Textural_Contrast_xray_avg"]
#subset_feature_list = [constants.CELLID_COL, "Structural_DistanceNucleus"]

dataframe_labelled = dataframe_full[subset_feature_list].copy()

dataframe_labelled.loc[:, "Subtype"] = labels

significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled,
    cluster_col="Subtype",
    cmap=clustermap_cmap,
    axes = True
    
)
summary_df = dataframe_labelled.groupby('Subtype').mean(numeric_only = True)

#%%
summary_df = dataframe_labelled.groupby('Subtype')['Geometric_MeanRadius'].mean()
print(summary_df)
#%% effect size

significant_features, stat_df, dunn_df = utils.kw_dunn_effects(dataframe_labelled, cluster_col = "Subtype")
subset_feature_list_plot = [x for x in subset_feature_list if x != constants.CELLID_COL]

#subset_feature_list_plot = ["Geometric_FormFactor"]
#subset_feature_list_plot = ["Structural_DistanceNucleus"]
#subset_feature_list_plot = [ "Densitometric_MeanDensitometric_xray"]
#subset_feature_list_plot = ["Textural_Contrast_xray_avg"]

for feature in subset_feature_list_plot:
    table_df = stat_df[stat_df["feature"]==feature]
    table_df = table_df[["pair", "p_adj", "cliffs_delta"]]
    table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
    table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
    table_renamed = table_df.rename(columns={"pair": "Type pair", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })
    
    #plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(3,1.6), title = "")

    plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(4,2), title = f"{feature}")

#%% example mitochondria per type

max_representative = 20
pop = 4
cluster_num = 4

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

#%% distribution of mitochondria types within cells

# add cellID back
mito_labelled["CellID"] = all_sheets_original["mitochondria"]["Metadata_CellID"]

plot.stacked_bxplt_count(
    mito_labelled, "CellID", "Subtype", 
    label_color_mapping=cmap_labelling, 
    legend=False, 
    axis_labels = False
)

plot.stacked_bxplt_count(
    mito_labelled,
    "CellID",
    "Subtype",
    label_color_mapping=cmap_labelling,
    normalize=True,
    legend = False,
    axis_labels = False
)

#%% distribution of mitochondria feature per population 
import matplotlib.pyplot as plt
mito_props_renamed_cell = mito_props_renamed.copy()
mito_props_renamed_cell["CellID"] = all_sheets_original["mitochondria"]["Metadata_CellID"]


mito_props_renamed_cell1 = mito_props_renamed_cell[mito_props_renamed_cell["CellID"]=="CELL005"]
mito_props_renamed_cell2 = mito_props_renamed_cell[mito_props_renamed_cell["CellID"]=="CELL012"]
col1 = pd.to_numeric(mito_props_renamed_cell1["Geometric_Area"], errors="coerce").dropna()
col2 = pd.to_numeric(mito_props_renamed_cell2["Geometric_Area"], errors="coerce").dropna()

df = pd.DataFrame({
    "Geometric_Area": np.concatenate([col1, col2]),
    "group": ["A"] * len(col1) + ["B"] * len(col2)
})

# Example: df is your DataFrame and 'column_name' is your column
sns.histplot(data=df, x='Geometric_Area', bins = 20, hue = "group", kde = False, stat = "probability", color = "blue", alpha = 0.5, legend = False)

plt.ylabel("")
plt.xlabel("")
plt.show()


#%% collapse mitochondria type measurement per cell 

# example column names: 'cellid' and 'type'
per_cell_proportion_df = (
    mito_labelled.groupby('CellID')['Subtype']
      .value_counts(normalize=True)
      .unstack(fill_value=0)
      .reset_index()
)
#%% distribution of mitochondria types in subcellular regions
classification_df = context_classification_mitochondria(
    all_sheets_original,
    all_segmentation,
    all_sheets_original["mitochondria"][constants.CELLID_COL],
)

classification_list = [x for x in classification_df.columns if "Classification" in x]

for cluster_type in classification_list:
    org.stacked_cluster_box_plts_pair(classification_df, "CellID", cluster_type)



# %% classification of mitochondria (cluster based on spatial localisation)
classification_df = context_classification_mitochondria(
    all_sheets_original,
    all_segmentation,
    all_sheets[constants.MITOCHONDRIA]["ObjectID"],
    DEBUG = False
)

classification_list = [x for x in classification_df.columns if "Classification" in x]

#%%
#if not in cell edge or perinuclear then cytoplasm 
classification_df['Classification_Cytoplasm'] = ((classification_df['Classification_Perinuclear'] == 0) & (classification_df['Classification_CellEdge'] == 0)).astype(int)
classification_df['Classification_CellEdge_only'] = ((classification_df['Classification_CellEdge'] == 1) & (classification_df['Classification_Perinuclear'] == 0)).astype(int)
classification_df['Classification_Perinuclear_only'] = ((classification_df['Classification_CellEdge'] == 0) & (classification_df['Classification_Perinuclear'] == 1)).astype(int)
classification_df['Classification_Squished'] = ((classification_df['Classification_CellEdge'] == 1) & (classification_df['Classification_Perinuclear'] == 1)).astype(int)


classification_list = [x for x in classification_df.columns if "Classification" in x]

#%% plot 
for cluster_type in classification_list:
    org.stacked_cluster_box_plts_pair(classification_df, "CellID", cluster_type)


#%% subtypes in each classification 
stack_order = [1, 2, 3, 4]
colors = [cmap_labelling[val] for val in stack_order]

classification_plot = classification_df.copy()
classification_plot = classification_df.merge(mito_labelled[['Metadata_CellID', 'Subtype']],
                                            on='Metadata_CellID',
                                            how='left'
                                            )
#classification_plot = classification_plot[classification_plot["CellID"]=="CELL001"]

cols = ["Classification_Perinuclear_only", "Classification_Cytoplasm", "Classification_Cluster", "Classification_CellEdge_only"]#, "Classification_Squished"]

counts = pd.concat({
    col: classification_plot[classification_plot[col] == 1]["Subtype"].value_counts()
    for col in cols
}, axis=1).fillna(0).sort_index()


#counts.T.plot(kind="bar", stacked=True)
#ax = counts.T.plot(kind="bar", stacked=True, figsize=(5, 5), color = colors)

percent = counts.div(counts.sum(axis=0), axis=1)# * 100
ax = percent.T.plot(kind="bar", stacked=True, figsize=(5, 5), color = colors, legend = False)

ax.set_xlabel("")
ax.set_xticks([])
ax.set_ylabel("")
ax.set_ylim(0, 1)
#ax.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()

#%% per cell

import pandas as pd
import matplotlib.pyplot as plt

stack_order = [1, 2, 3, 4]
colors = [cmap_labelling[val] for val in stack_order]

classification_plot = classification_df.merge(
    mito_labelled[['Metadata_CellID', 'Subtype']],
    on='Metadata_CellID',
    how='left'
)

cols = [
    "Classification_Perinuclear_only",
    "Classification_CellEdge_only",
    "Classification_Cluster",
    "Classification_cytoplasm"
]

for col in cols:
    # keep only rows where this classification is 1
    df_sub = classification_plot[classification_plot[col] == 1]

    # count Subtype per CellID
    counts = pd.crosstab(df_sub["CellID"], df_sub["Subtype"])

    # ensure subtype order
    counts = counts.reindex(columns=stack_order, fill_value=0)

    # convert to percentages per cell
    percent = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    ax = percent.plot(
        kind="bar",
        stacked=True,
        figsize=(6, 4),
        color=colors,
        legend=False
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(0, 1)
    ax.set_title(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

#%% test enrichment in subtype in location (no enrichment)
axes = True

classification_plot["Perinuclear"] = classification_plot["Classification_Perinuclear"] == 1 
classification_plot["Cytoplasmic"] = classification_plot["Classification_Cytoplasm"] == 1 
classification_plot["Cluster"] = classification_plot["Classification_Cluster"] == 1 
classification_plot["Cell Edge"] = classification_plot["Classification_CellEdge_only"] == 1

regions = ["Perinuclear", "Cytoplasmic",  "Cluster", "Cell Edge"]
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
classification_plot["Cytoplasmic"] = classification_plot["Classification_Cytoplasm"] == 1 
classification_plot["Cluster"] = classification_plot["Classification_Cluster"] == 1 
classification_plot["Cell Edge"] = classification_plot["Classification_CellEdge_only"] == 1

regions = ["Perinuclear", "Cytoplasmic",  "Cluster", "Cell Edge"]
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
import seaborn as sns
import numpy as np

# Choose a fixed scale so all 4 heatmaps are comparable
vmin, vmax = -2, 2

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.flatten()

# Store the first heatmap mappable so we can create one shared colorbar
mappable = None

for ax, (region, res) in zip(axes, residuals.items()):
    hm = sns.heatmap(
        res,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14},
        cbar=False
    )
    
    if mappable is None:
        mappable = hm.collections[0]
    
    ax.set_xticklabels(fontsize = 14)
    ax.set_title(region, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=14)

# Add one shared colorbar on the right edge
cbar = fig.colorbar(mappable, ax=axes, location="right", shrink=0.9, pad=0.02)
#cbar.set_label("Standardized Residuals", fontsize=12)
cbar.ax.tick_params(labelsize=10)

plt.show()

#%% mosaic plot [meh]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from scipy.stats import chi2_contingency

for region in regions:
    # Force consistent labels
    subtype = classification_plot["Subtype"].astype(str)
    region_vals = classification_plot[region].astype(str)

    contingency = pd.crosstab(subtype, region_vals)

    chi2, p, dof, expected = chi2_contingency(contingency)
    expected = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)

    res = (contingency - expected) / np.sqrt(expected)

    def props(key):
        row, col = key
        z = res.loc[row, col]
        if z > 2:
            return {"color": "red", "edgecolor": "black"}
        elif z < -2:
            return {"color": "blue", "edgecolor": "black"}
        else:
            return {"color": "lightgrey", "edgecolor": "black"}

    fig, ax = plt.subplots(figsize=(8, 5))
    mosaic(contingency.stack().to_dict(), properties=props, ax=ax)
    ax.set_title(f"Mosaic Plot: {region}")
    plt.show()
#%% compare properties between locations
feat_type = "Geometric"

dataframe_full = mito_props_renamed.copy()
subset_feature_list = [
    x
    for x in dataframe_full.columns
    if (feat_type in x.split("_")[0] and "Zernike" not in x or constants.CELLID_COL in x)
]

if len(subset_feature_list) == 1:
    print("Check feature spelling")

#subset_feature_list = [constants.CELLID_COL, "Structural_OrientationNucleus"]#, "Geometric_AspectRatio"]
dataframe_labelled = dataframe_full[subset_feature_list].copy()

dataframe_labelled.loc[:, "Subtype"] = classification_plot["Classification_Perinuclear_only"]

significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled,
    cluster_col="Subtype",
    cmap=clustermap_cmap,
)


significant_features, stat_df, dunn_df = utils.kw_dunn_effects(dataframe_labelled, cluster_col = "Subtype")
subset_feature_list_plot = [x for x in subset_feature_list if x != constants.CELLID_COL and x != "CellID"]

table_df = stat_df.copy()
table_df = table_df[["feature","pair", "p_adj", "cliffs_delta"]]
table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
table_renamed = table_df.rename(columns={"pair": "Type pair", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })
 
plot.create_table(table_df, body_fontsize=6, header_fontsize=6, fig_size=(6,2), title = f"{feature}")

#%%
for feature in subset_feature_list_plot:
    table_df = stat_df[stat_df["feature"]==feature]
    table_df = table_df[["pair", "p_adj", "cliffs_delta"]]
    table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
    table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
    table_renamed = table_df.rename(columns={"pair": "Type pair", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })
    
    #plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(3,1.6), title = "")

    plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(4,2), title = f"{feature}")


# %% properties of lipid droplets based on classification

dataset_subset = {}
dataset_subset["Shape"] = org.organelle_shape_dataset(dataset_type_split, type="all")
dataset_subset["Texture"] = org.organelle_texture_dataset(dataset_type_split)
dataset_subset["Intensity"] = org.organelle_intensity_dataset(dataset_type_split)

# %% plotting

feature_type = constants.OBJ_SHAPE
classification_group = 1

dataset_subset[feature_type], dropped_cols_clean = utils.drop_no_variation_cols(
    dataset_subset[feature_type], threshold=0.0001
)

dataset_labelled = dataset_subset[feature_type].merge(
    classification_df[
        [classification_list[classification_group], constants.CELLID_COL]
    ],
    on=constants.CELLID_COL,
    how="left",
)

significant_features = org.all_feat_violin_plots(
    dataset_labelled,
    id_col=constants.CELLID_COL,
    cluster_col=classification_list[classification_group],
)

# %% shape analysis

title = "Mitochondria"
cluster_method = "KMeans"
cluster_num = 4

dataset_shape = org.organelle_shape_dataset(dataset_type_split, type="all")

(
    silhouette_score_stats_dict,
    dataset_shape_labelled,
    significant_features,
    colors,
    label_color_mapping,
) = org.subtype_analysis(
    dataset_shape,
    cluster_num=cluster_num,
    cluster_method=cluster_method,
    title=title,
    label_method="HCA",
)

features_sel, sel = MRMR_dataframe(
    dataset_shape.drop(columns=constants.CELLID_COL), dataset_shape_labelled["Subtype"]
)
features_sorted = pd.Series(sel.relevance_, index=sel.variables_).sort_values(
    ascending=False
)

contributing_features_plt = plot_feature_relevance(
    sel,
    keyword_cmap={
        "Zernike": ("steelblue", "Zernike feature"),
        "AreaShape": ("lightskyblue", "Geometric feature"),
    },
)

# %%
max_representative = 4
pop = 0
cluster_num = 4

for pop in list(range(1, cluster_num)):
    org.plot_selected(
        silhouette_score_stats_dict,
        segmentation_masks,
        organelle,
        readfile=False,
        max_display=max_representative,
        population=pop,
        xray_overlay=False,
        xray_paths=xray_paths,
        cropped_roi=True,
    )

# %% plotting correlation matrices

plot.correlation_matrix(
    dataset_shape.drop(columns=constants.CELLID_COL),
    plot=True,
    title="Geometric Shape Features Correlation Matrix",
)
plot.correlation_matrix_hover(
    dataset_shape.drop(columns=constants.CELLID_COL),
    plot=True,
    savedir=constants.DOWNLOADDIR / "correlation_matrix.html",
)

# %%

features_plot = ["AreaShape_FormFactor", "AreaShape_AspectRatio"]
plot.plot_scatter_feat_combo(
    dataset_shape,
    features_plot,
    hover=False,
    id_col=constants.CELLID_COL,
    kind="density",
)

# plot lipid droplet shape (density + labelled)
org.plot_scatters(dataset_shape, dataset_shape_labelled, features_plot, colors)

# %% IO
run = "run3"
treatment_list = ["control", "inflammation"]
pick_features = False  # if keep_cols.xlsx file already exists with selected features

datadir = {}
outputdir = {}
filedir = {}
figdir = {}
all_sheets = {}
all_segmentation = {}
roidir = {}

for treatment in treatment_list:
    datadir[treatment] = Path(
        rf"C:/Users/IvyWork/Desktop/projects/dataset/input_{treatment}/{run}"
    )
    roidir[treatment] = datadir[treatment] / "results"
    outputdir[treatment] = datadir[treatment] / "summary_sheets_updated"
    outputdir[treatment].mkdir(parents=True, exist_ok=True)
    figdir[treatment] = datadir[treatment] / "figures"
    figdir[treatment].mkdir(parents=True, exist_ok=True)

    # combine excel data sheets
    filedir[treatment] = datadir[treatment] / "measurements" / "measurements_updated"
    all_sheets[treatment] = combine_files_to_sheets(filedir[treatment])
    # add object ID to individual organelles
    all_sheets[treatment]["mitochondria"] = utils.update_cellID_objectnum(
        all_sheets[treatment]["mitochondria"],
        "ImageNumber",
        "ObjectNumber",
        "Metadata_CellID",
    )
    all_sheets[treatment]["lipiddroplets"] = utils.update_cellID_objectnum(
        all_sheets["control"]["lipiddroplets"],
        "ImageNumber",
        "ObjectNumber",
        "Metadata_CellID",
    )

if pick_features:
    # create list of features per organelle to select features from control dataframe
    treatment = "control"
    feature_list = {}
    for organelle, df in all_sheets[treatment].items():
        feature_list[organelle] = df.columns.tolist()

    with pd.ExcelWriter(outputdir[treatment] / "keep_cols_full.xlsx") as writer:
        for key, values in feature_list.items():
            df = pd.DataFrame({key: values})
            df.to_excel(writer, sheet_name=key, index=False)

    print("list of features saved - select features to keep in new keep_cols.xlsx file")
    sys.exit("stopping script so features can be selected :)")

else:
    keep_cols = pd.read_excel(outputdir["control"] / "keep_cols.xlsx", sheet_name=None)

downloaddir = Path(r"C:/Users/IvyWork/Downloads")

# extract selected features
all_organelles_ctrl = filter_dataframes(
    all_sheets["control"],
    keep_cols,
    savedir="",
)

all_organelles_inf = filter_dataframes(
    all_sheets["inflammation"],
    keep_cols,
    savedir="",
)

# check for outliers
# dataset_no_outliers, outlier_cols = utils.remove_outliers(all_organelles_ctrl)

# compare dataframes - identify common features
# HCA_df_ctrl_filtered, HCA_df_inf_filtered, _, dropped_cols = utils.compare_dataframes(
# HCA_df_ctrl, HCA_df_inf, return_variables="all"
# )

mito_segmentation = load_segmentation(roidir["control"], organelle=["mitochondria"])

# %% mitochondria analysis
outlier_check = False
mito_features = list(all_organelles_ctrl[mito].columns)
mito_features_nonZernike = [x for x in mito_features if "Zernike" not in x]
feature_subset = all_organelles_ctrl[mito][mito_features_nonZernike]

# feature_subset = all_organelles_ctrl[mito]

mito_clean, dropped_cols_empty = utils.drop_empty_cols(
    feature_subset, threshold=0
)  # 0.2 = 20% empty

# check for outliers
if outlier_check:
    mito_clean, outlier_cols = utils.remove_outliers(mito_clean)

feature_types = [
    ["AreaShape"],
    ["Intensity", "RadialDistribution"],
    ["Texture"],
    ["Structure"],
]

mito_type = {}
for feature in feature_types:
    if len(feature) > 1:
        feature_combined_list = [x for x in feature]
        feature_name = "_".join(feature_combined_list)
    else:
        feature_name = feature[0]

    mito_type[feature_name] = utils.extract_df_subset_sequential(
        mito_clean, subset=[feature], constant_key="Metadata_CellID"
    )

mito_type_featurelist = {}

for organelle, df in mito_type.items():
    mito_type_featurelist[organelle] = list(df.columns)

# %% Mitochondria shape
title = "Mitochondria Shape"
cluster_method = "KMeans"
cluster_num = 4
cluster_method = "KMeans"

remove_features_Shape = [
    "AreaShape_EulerNumber",
    "AreaShape_MaxFeretDiameter",
    "AreaShape_MinFeretDiameter",
    "AreaShape_MaximumRadius",
    "AreaShape_MedianRadius",
    # "AreaShape_Compactness",
    # "AreaShape_EquivalentDiameter",
]

feat_list = [
    x for x in mito_type_featurelist["AreaShape"] if x not in remove_features_Shape
]

dataset_shape = mito_type["AreaShape"][feat_list]
PCA_labelled, colors, label_names, label_color_mapping = plot.PCA_clustering_silhouette(
    dataset_shape, title, cluster_num, cluster_method
)
dataset_shape_labelled = dataset_shape.copy()
dataset_shape_labelled["Subtype"] = PCA_labelled[PCA_labelled.columns[-1]]
cluster_col = dataset_shape_labelled.columns[-1]

individual_scores_df = silhouette_score_indiv(
    dataset_shape,
    dataset_shape_labelled["Subtype"],
    scale=True,
    cellid_col="Metadata_CellID",
)
silhouette_score_stats_dict = silhouette_score_stats(individual_scores_df)

# %% plot mito features
scatter_feat_list = ["AreaShape_Tortuosity", "AreaShape_AspectRatio"]
_ = plot.plot_scatter_feat_combo(
    dataset_shape,
    scatter_feat_list,
    hover=True,
    id_col="Metadata_CellID",
    savedir=figdir["control"],
    labels=dataset_shape_labelled["Subtype"],
    cluster_colors=colors,
)

# %% plot selected mito
readfile = False
max_display = 6
population = 3
list_of_mito = [
    ID
    for ID, score in silhouette_score_stats_dict["subpopulation_ranked_cells"][
        population
    ]
]

if readfile:
    selected_mito = pd.read_csv(downloaddir / "selected_points.csv")
    list_of_mito = selected_mito["ID"].tolist()

if max_display != all:
    list_of_mito = list_of_mito[0:max_display]

mito_shape_obj = {}
for mito_obj in list_of_mito:
    mito_shape_obj[mito_obj], _ = utils.show_object(
        mito_segmentation["mitochondria"], "mitochondria", mito_obj
    )
    plt.imshow(mito_shape_obj[mito_obj])
    plt.title(f"{mito_obj}")
    plt.show()

# %%
# clear selected_points.csv before creating new plot
# Source - https://stackoverflow.com/a
# Posted by Matt, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-26, License - CC BY-SA 3.0

utils.delete_file(downloaddir / "selected_points.csv")

# %% distribution of mitochondria types in each cell

# add cellID back
dataset_shape_labelled["CellID"] = all_sheets["control"]["mitochondria"]["CellID"]
plot.stacked_bxplt_count(
    dataset_shape_labelled, "CellID", "Subtype", label_color_mapping=label_color_mapping
)
plot.stacked_bxplt_count(
    dataset_shape_labelled,
    "CellID",
    "Subtype",
    label_color_mapping=label_color_mapping,
    normalize=True,
)


# %%
significant_features, molten_dunn_df, dunn_df = utils.kw_dunn(
    dataset_shape_labelled, alpha=0.05, p_adjust="fdr_bh"
)

for feature in feat_list:
    if feature == "Metadata_CellID":
        continue

    plot.subpopulation_violplt(
        dataset_shape_labelled, dunn_df[feature], cluster_col, feature
    )

# %% Mitochondria texture
dataset_texture = mito_type["Texture"]
cluster_num = 2
cluster_method = "KMeans"
title = "Mitochondria Texture"

feature_basename = []
for feature in dataset_texture.columns:
    components = feature.split("_")
    basename = components[0] + "_" + components[1]

    if feature_basename not in feature_basename:
        feature_basename.append(basename)

dataset_texture_updated = pd.DataFrame()
for base in feature_basename:
    if base == "Metadata_CellID":
        continue

    # find columns containing the base name
    cols = [c for c in dataset_texture.columns if base in c]

    # compute row-wise mean
    dataset_texture_updated[f"{base}_avg"] = dataset_texture[cols].mean(axis=1)

PC_labelled = plot.PCA_clustering_silhouette(
    dataset_texture_updated, title, cluster_num, cluster_method
)

feat_list = dataset_texture_updated.columns
# _ = plot_scatter_feat_combo(dataset_texture_updated, feat_list)

dataset_texture_labelled = dataset_texture_updated.copy()
dataset_texture_labelled["Subtype"] = PC_labelled[PC_labelled.columns[-1]]

significant_features, molten_dunn_df, dunn_df = utils.kw_dunn(
    dataset_texture_labelled, alpha=0.05, p_adjust="fdr_bh"
)

cluster_col = dataset_texture_labelled.columns[-1]
for f in significant_features:
    if f == "Metadata_CellID":
        continue

    plot.subpopulation_violplt(dataset_texture_labelled, dunn_df[f], cluster_col, f)
# %% Mitochondria intensity
cluster_num = 100
cluster_method = "KMeans"
title = "Mitochondria Intensity"

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
]

feat_list = [
    x
    for x in mito_type_featurelist["Intensity_RadialDistribution"]
    if x not in remove_features_Intensity
]

dataset_Intensity = mito_type["Intensity_RadialDistribution"]
intensity_range = (
    dataset_Intensity["Intensity_MaxIntensity_xray"]
    - dataset_Intensity["Intensity_MinIntensity_xray"]
)
edgeintensity_range = (
    dataset_Intensity["Intensity_MaxIntensityEdge_xray"]
    - dataset_Intensity["Intensity_MinIntensityEdge_xray"]
)

dataset_Intensity["Intensity_Range"] = intensity_range
feat_list.append("Intensity_Range")

dataset_Intensity["Intensity_RangeEdge"] = edgeintensity_range
feat_list.append("Intensity_RangeEdge")

dataset_Intensity_updated = mito_type["Intensity_RadialDistribution"][feat_list]

PCA_labelled = plot.PCA_clustering_silhouette(
    dataset_Intensity_updated, title, cluster_num, cluster_method
)

feat_list = dataset_Intensity_updated.drop(columns="Metadata_CellID").columns
# _ = plot_scatter_feat_combo(dataset_Intensity_updated, feat_list)

dataset_intensity_labelled = dataset_Intensity_updated.copy()
dataset_intensity_labelled["Subtype"] = PCA_labelled[PCA_labelled.columns[-1]]


significant_features, molten_dunn_df, dunn_df = utils.kw_dunn(
    dataset_intensity_labelled, alpha=0.05, p_adjust="fdr_bh"
)

cluster_col = dataset_intensity_labelled.columns[-1]
for f in significant_features:
    if f == "Metadata_CellID":
        continue

    plot.subpopulation_violplt(dataset_intensity_labelled, dunn_df[f], cluster_col, f)
# %%
cluster_num = 3
cluster_method = "KMeans"
title = "Mitochondria Structure"

dataset_structure = mito_type["Structure"][
    [
        "Structure_OrientationNucleus",
        "Structure_NearestNeighbourDistance",
        "Structure_NeighbourNumber",
        "Metadata_CellID",
    ]
]

dataset_structure = mito_type["Structure"]

PCA_labelled = plot.PCA_clustering_silhouette(
    dataset_structure, title, cluster_num, cluster_method, n_PCA_components=3
)

feat_list = dataset_structure.drop(columns="Metadata_CellID").columns
# _ = plot_scatter_feat_combo(dataset_structure, feat_list)

dataset_structure_labelled = dataset_structure.copy()
dataset_structure_labelled["Subtype"] = PCA_labelled[PCA_labelled.columns[-1]]

significant_features, molten_dunn_df, dunn_df = utils.kw_dunn(
    dataset_structure_labelled, alpha=0.05, p_adjust="fdr_bh"
)

cluster_col = dataset_structure_labelled.columns[-1]
for f in significant_features:
    if f == "Metadata_CellID":
        continue

    plot.subpopulation_violplt(dataset_structure_labelled, dunn_df[f], cluster_col, f)

# %%
with pd.ExcelWriter(
    outputdir["control"] / "mitoShape_selectfeat.xlsx", engine="openpyxl"
) as writer:
    dataset_plot.to_excel(writer, index=False)

from WGCNA import convert_xlsx_to_csv
import PyWGCNA

scaled_excel = convert_xlsx_to_csv(outputdir["control"] / "mitoShape_selectfeat.xlsx")

pyWGCNA_obj = PyWGCNA.WGCNA(
    name="U2OS",
    species="human",
    geneExpPath=str(outputdir["control"] / "mitoShape_selectfeat.csv"),
    outputPath="Desktop/projects/dataset/input_control/run3/",
    save=False,
)

pyWGCNA_obj.findModules(cutHeight=0.8)
pyWGCNA_obj.CalculateSignedKME()
modules = pyWGCNA_obj.datExpr.var.moduleColors.unique().tolist()
pyWGCNA_obj.CoexpressionModulePlot(
    modules=modules, numGenes=5, numConnections=5, minTOM=0, file_name="all_features"
)

# %% differences between clusters
p_good = outputdir["control"] / "HCA.csv"
p_bad = outputdir["control"] / "mitoShape_selectfeat.csv"

for p, label in [(p_good, "GOOD (HCA.csv)"), (p_bad, "BAD (mitoShape_selectfeat.csv)")]:
    print("\n---", label, p, "---")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print("read_csv error:", e)
        continue
    print("shape:", df.shape)
    print("columns:", list(df.columns)[:30])
    print("dtypes:\n", df.dtypes.value_counts().to_dict())
    print("first rows:")
    display(df.head(5))
    # check for messy index/unnamed columns
    print("Unnamed columns:", [c for c in df.columns if c.startswith("Unnamed")])
    # check for 'Value' column
    print("'Value' in columns?", "Value" in df.columns)
    # check if looks long/tidy (has Gene/Sample/Value)
    maybe_long = set(
        ["gene", "Gene", "sample", "Sample", "Value", "value"]
    ).intersection(df.columns)
    print("Possible tidy columns present:", maybe_long)
# %% HCA clustering - per organelle features within cell (summary)

organelles = [["mitochondria"], ["nucleus"], ["lipiddroplet"], ["cell", "cytoplasm"]]
organelle_datasets = {}

for organelle in organelles:
    if len(organelle) > 1:
        organelle_combined_list = [x for x in organelle]
        organelle_name = "_".join(organelle_combined_list)
    else:
        organelle_name = organelle[0]
    organelle_datasets[organelle_name] = utils.extract_df_subset_sequential(
        HCA_df_ctrl_filtered, subset=[organelle], constant_key="Metadata_CellID"
    )

best_k = [2, 3, 2, 2]

selected_features_type = []
selected_features = {}

for keyword_filter_list, best_k in zip(
    organelles, best_k
):  # list of keyword lists for sequential dataframe column filtering
    keyword_filter_list = [keyword_filter_list]
    if len(keyword_filter_list) > 1:
        keyword_filter_flat_list = [x for xs in keyword_filter_list for x in xs]
        keyword_list_name = "_".join(keyword_filter_flat_list)
    else:
        keyword_list_name = keyword_filter_list[0][0]

    dataset_subset = utils.extract_df_subset_sequential(
        HCA_df_ctrl_filtered, subset=keyword_filter_list, constant_key="Metadata_CellID"
    )

    if len(dataset_subset.columns) < 1:  # skip if feature set doesn't exist
        continue
    elif len(dataset_subset.columns) < 20:  # if less than 20 features keep them all
        selected_features[keyword_list_name] = list(dataset_subset.columns)
    else:
        (
            _,
            _,
            selected_features[keyword_list_name],
            _,
        ) = HCA_top_features(
            HCA_df_ctrl_filtered,
            keyword_filter_list,
            best_k=best_k,
            name="control cell",
            title=f"Clustered Heatmap of {keyword_list_name} Features",
            color_row=False,
        )

    selected_features_type.extend(selected_features)
