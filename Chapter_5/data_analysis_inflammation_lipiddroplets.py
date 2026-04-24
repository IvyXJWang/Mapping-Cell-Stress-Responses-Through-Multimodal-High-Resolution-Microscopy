# lipiddroplets organelle analysis

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

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
    random_forest_train_model,
    random_forest_classification,
    model_eval
)
from feature_extraction_utils import (
    load_segmentation,
    context_classification_lipid,
)
import plotting as plot
from utils import rename_df_columns_with_keyword
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib as mpl
import seaborn as sns
from segmentation_utils import load_path_into_dict
import joblib
import sklearn.preprocessing as skprep
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from data_analysis_figs import plot_overlays_masks_cell
#%% fit same scaler from training dataset to prediction dataset

def scale_data_fit(dataframe, id_col="ObjectID"):
    df = dataframe.copy()

    id_series = df[id_col] if id_col in df.columns else None
    feature_df = df.drop(columns=[id_col]) if id_col in df.columns else df

    numeric = feature_df.select_dtypes(include=np.number)
    non_numeric = feature_df.select_dtypes(exclude=["number"])

    scaler = skprep.StandardScaler()
    scaled = scaler.fit_transform(numeric)
    scaled = np.nan_to_num(scaled, nan=0.0)

    scaled_df = pd.DataFrame(scaled, columns=numeric.columns, index=df.index)
    final_df = pd.concat([scaled_df, non_numeric], axis=1)

    if id_series is not None:
        final_df[id_col] = id_series.values

    return final_df, scaler


def scale_data_transform(dataframe, scaler, id_col="ObjectID"):
    df = dataframe.copy()

    id_series = df[id_col] if id_col in df.columns else None
    feature_df = df.drop(columns=[id_col]) if id_col in df.columns else df

    numeric = feature_df.select_dtypes(include=np.number)
    non_numeric = feature_df.select_dtypes(exclude=["number"])

    scaled = scaler.transform(numeric)
    scaled = np.nan_to_num(scaled, nan=0.0)

    scaled_df = pd.DataFrame(scaled, columns=numeric.columns, index=df.index)
    final_df = pd.concat([scaled_df, non_numeric], axis=1)

    if id_series is not None:
        final_df[id_col] = id_series.values

    return final_df

def apply_group_normalisation(df, feature_groups):
    df = df.copy()
    for feature, cols in feature_groups.items():
        df[cols] /= np.sqrt(len(cols))
    
    return df

def clr_transform(df, eps=1e-9):
    x = df.to_numpy(dtype=float)
    x = np.clip(x, eps, None)
    clr = np.log(x) - np.log(x).mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=df.index, columns=df.columns)

def cliffs_delta(x, y):
    """
    Cliff's delta for two independent samples.
    Ties contribute 0.
    Range: [-1, 1]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    diff = np.subtract.outer(x, y)
    n_greater = np.sum(diff > 0)
    n_less = np.sum(diff < 0)

    return (n_greater - n_less) / (len(x) * len(y))

def pairwise_mannwhitney_df(dataframe, feature, subtype_col, condition_col,
                        control_label="control", inflam_label="inflammation",
                        alpha=0.05, p_adjust="fdr_bh"):

    subtypes = dataframe[subtype_col].dropna().unique().tolist()
    rows = []

    for subtype in subtypes:
        sub_df = dataframe[dataframe[subtype_col] == subtype]

        x = sub_df.loc[sub_df[condition_col] == control_label, feature].dropna()
        y = sub_df.loc[sub_df[condition_col] == inflam_label, feature].dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        u_stat, p_raw = mannwhitneyu(x, y, alternative="two-sided")

        rows.append({
            "feature": feature,
            "subtype": subtype,
            "comparison": f"{control_label} vs {inflam_label}",
            "n_control": len(x),
            "n_inflam": len(y),
            "u_stat": u_stat,
            "p_raw": p_raw,
            "cliffs_delta": cliffs_delta(x, y),
        })

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        reject, p_adj, _, _ = multipletests(results_df["p_raw"], alpha=alpha, method=p_adjust)
        results_df["p_adj"] = p_adj
        results_df["reject"] = reject

    return results_df
#%% load in both datasets for clustering

version = "run4"
#organelle = constants.LIPID_DROPLETS

# identify subpopulations of lipiddroplets
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

treatment = "control"
run = "run4"
parentdir = Path(constants.PARENTDIR / f"{treatment}" / f"{run}")
keep_cols_reduced = pd.read_excel(
    parentdir / "selected_features_organelle.xlsx", sheet_name=None, header=None
)
#rf_model = joblib.load(Path(constants.PARENTDIR / "control" / "run4" / "random_forest_model.joblib"))
#data_scaler = joblib.load(Path(constants.PARENTDIR / "control" / "run4" / "random_forest_data_scalar.joblib"))

clusternum = 4

subtype_cells = {}
all_sheets = {}
all_sheets_original = {}
lipid_props = {}
lipid_props_renamed = {}
lipid_scaled = {}
lipid_labelled = {}
all_sheets_filtered = {}
rf_eval = {}

for treatment in ["control", "inflammation"]:

    datadir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
    filedir = datadir / "cellprofiler" / "measurements_updated"
    all_sheets_original[treatment] = combine_files_to_sheets(filedir)

    all_sheets[treatment] = all_sheets_original[treatment].copy()
    
    all_sheets[treatment]["lipiddroplets"] = utils.update_cellID_objectnum(
        all_sheets[treatment]["lipiddroplets"],
        "ImageNumber",
        "ObjectNumber",
        "Metadata_CellID",
    )

    all_sheets_filtered[treatment] = filter_dataframes(all_sheets[treatment], keep_cols_reduced)

    lipid_props[treatment] = all_sheets_filtered[treatment]["lipiddroplets"].copy()
    lipid_props[treatment].loc[:, "ObjectID"] = all_sheets[treatment]["lipiddroplets"]["ObjectID"]
    lipid_props[treatment] = lipid_props[treatment].drop("Metadata_CellID", axis=1)

    lipid_props_renamed[treatment] = rename_df_columns_with_keyword(lipid_props[treatment], rename_dict)

    feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]

    if treatment == "control": # new label
        lipid_scaled[treatment], scaler = scale_data_fit(lipid_props_renamed[treatment], id_col = "ObjectID")
        feature_groups = {}
        for feature in feature_types:
            cols = lipid_scaled[treatment].columns[
                lipid_scaled[treatment].columns.str.contains(feature, case=False)
            ]
            
            feature_groups[feature] = cols
        
        lipid_props_normalised = apply_group_normalisation(lipid_scaled[treatment], feature_groups)
        
        lipid_labelled[treatment], _ = add_cluster_label(
            lipid_props_normalised,
            cluster_num=clusternum,
            clustering_algorithm="HCA",
            id_col="ObjectID",
            lbl_colname="Subtype",
        )

        rf_model, data_scaler, scores, scores_mean, scores_std, feature_importance = (
            random_forest_train_model(
                lipid_labelled[treatment], "Subtype", savedir=datadir, scale = False, id_col = "ObjectID"
            )
        )
        
        X_scaled = lipid_props_normalised.drop(columns="ObjectID")
        Y = lipid_labelled[treatment]["Subtype"]
        
        rf_eval[treatment] = model_eval(X_scaled,Y, rf_model = rf_model)
    
    elif treatment == "inflammation":
        lipid_scaled[treatment] = scale_data_transform(lipid_props_renamed[treatment], scaler, id_col = "ObjectID")
        lipid_props_normalised = apply_group_normalisation(lipid_scaled[treatment], feature_groups)

        lipid_labelled[treatment], subpopulation_lipid_rf, sorted_obj, feature_importance = random_forest_classification(
            rf_model, data_scaler, lipid_props_normalised, id_col="ObjectID", scale=False
        )
        lipid_labelled[treatment] = lipid_labelled[treatment].rename(
            columns={'Random_forest_population': 'Subtype'}
        )  
        
        X_scaled = lipid_props_normalised.drop(columns="ObjectID")
        Y = lipid_labelled[treatment]["Subtype"]
        
        rf_eval[treatment] = model_eval(X_scaled,Y, rf_model = rf_model)
    
    subpopulation_cells = {}
    for i in list(range(1, clusternum + 1)):  # start subpopulation count at 1
        subpopulation_cells[i] = lipid_labelled[treatment].loc[
            lipid_labelled[treatment]["Subtype"] == i, "ObjectID"
        ].tolist()
    
    subtype_cells[treatment] = subpopulation_cells
    
_, _, cmap_labelling = label_color_mapping_dict(
    lipid_labelled["control"]["Subtype"], palette="inferno"
)

#%% feature importance of random forest

a = all_sheets["inflammation"]["lipiddroplets"]
aa = all_sheets["control"]["lipiddroplets"]

missing_in_df2 = aa.columns.difference(a.columns)

#%% identify objects in each subtype 
treatment = "control"
subpopulation_cells = {}
for i in list(range(1, 3 + 1)):  # start subpopulation count at 1
    subpopulation_cells[i] = lipid_labelled[treatment].loc[
        lipid_labelled[treatment]["Subtype"] == i, constants.CELLID_COL
    ].tolist()
    
#%% load in segmentation for visualisation
run = "run4"
treatment = "inflammation"
organelle = constants.LIPID_DROPLETS
datadir = constants.PARENTDIR / f"{treatment}/{run}"
segmentationdir = datadir / "results"
xray_paths = load_path_into_dict(segmentationdir, keyword="xray")

all_segmentation = load_segmentation(
    segmentationdir, organelle=constants.ORGANELLE_CP_LIST
)

#%% comparison with control (population-wide feature) - no significant difference in area or circularity
all_features = lipid_props_renamed["control"].columns
feature = "Geometric_Area"

dataframe_labelled = pd.concat(
    [df[[feature, "ObjectID"]].assign(Treatment=i) 
     for i, (key, df) in enumerate(lipid_props_renamed.items(), start=1)],
    ignore_index=True
)

dataframe_labelled = dataframe_labelled.rename(
    columns={'ObjectID': 'Metadata_CellID'}
)        
    

significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled,
    cluster_col="Treatment",
    cmap=cmap_labelling,
    axes = True,
)


significant_features, stat_df, dunn_df = utils.kw_dunn_effects(dataframe_labelled, cluster_col = "Treatment")

table_df = stat_df[stat_df["feature"]==feature]
table_df = table_df[["pair", "p_adj", "cliffs_delta"]]
table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
table_renamed = table_df.rename(columns={"pair": "Type pair", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })

#plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(3,1.6), title = "")

plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(4,2), title = f"{feature}")

#%% compare average lipiddroplets size in cells - no significant difference in area or circularity 
feature = "Geometric_Area"

df=lipid_props_renamed.copy()
for treatment in ["control", "inflammation"]:
    df[treatment]["Metadata_CellID"] = all_sheets[treatment]["lipiddroplets"]["Metadata_CellID"]

dataframe_labelled = {
    key: df.groupby("Metadata_CellID", as_index=False)[feature].mean()
    for key, df in lipid_props_renamed.items()
}

dataframe_labelled_2 = pd.concat(
    [df[[feature, "Metadata_CellID"]].assign(Treatment=i) 
     for i, (key, df) in enumerate(dataframe_labelled.items(), start=1)],
    ignore_index=True
)


significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled_2,
    cluster_col="Treatment",
    cmap=cmap_labelling,
    axes = True,
)

significant_features, stat_df, dunn_df = utils.kw_dunn_effects(dataframe_labelled_2, cluster_col = "Treatment")

table_df = stat_df[stat_df["feature"]==feature]
table_df = table_df[["pair", "p_adj", "cliffs_delta"]]
table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
table_renamed = table_df.rename(columns={"pair": "Type pair", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })

#plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(3,1.6), title = "")

plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(4,2), title = f"{feature}")

#%% comparison with control (subtype feature) - individually

all_features = lipid_props_renamed["control"].columns
feature = "Geometric_Area"
subtype = 4

subtype_df = lipid_props_renamed.copy()
subtype_df["control"]["Subtype"] = lipid_labelled["control"]["Subtype"]
subtype_df["inflammation"]["Subtype"] = lipid_labelled["inflammation"]["Subtype"]

subtype_df["control"] = subtype_df["control"][subtype_df["control"]["Subtype"]==subtype]
subtype_df["inflammation"] = subtype_df["inflammation"][subtype_df["inflammation"]["Subtype"]==subtype]

dataframe_labelled = pd.concat(
    [df[[feature, "ObjectID"]].assign(Treatment=i) 
     for i, (key, df) in enumerate(subtype_df.items(), start=1)],
    ignore_index=True
)

dataframe_labelled = dataframe_labelled.rename(
    columns={'ObjectID': 'Metadata_CellID'}
)        
    
_,_,cmap_treatment = label_color_mapping_dict(
    dataframe_labelled["Treatment"], palette="Spectral"
)

significant_features_1, subpopulation_cells_viol = org.all_feat_violin_plots(
    dataframe_labelled,
    cluster_col="Treatment",
    cmap=cmap_treatment,
    axes = True,
)

summary_df = dataframe_labelled.groupby('Treatment').mean(numeric_only = True)

significant_features, stat_df, dunn_df = utils.kw_dunn_effects(dataframe_labelled, cluster_col = "Treatment")

if len(stat_df) != 0:
    table_df = stat_df[stat_df["feature"]==feature]
    table_df = table_df[["pair", "p_adj", "cliffs_delta"]]
    table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
    table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
    table_renamed = table_df.rename(columns={"pair": "Type pair", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })

    #plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(3,1.6), title = "")

    plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(4,2), title = f"{feature}")

#%% put all on same plot
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator

all_features = list(lipid_props_renamed["inflammation"].columns)
subtype_df = lipid_props_renamed.copy()
subtype_df["control"]["Subtype"] = lipid_labelled["control"]["Subtype"]
subtype_df["inflammation"]["Subtype"] = lipid_labelled["inflammation"]["Subtype"]

df_combined = pd.concat(subtype_df, names=["Treatment"]).reset_index(level=0)
df_combined["Geometric_Area"] = df_combined["Geometric_Area"] * (0.01 ** 2)

cmap = {
        "control": (0.9934640522875817, 0.7477124183006535, 0.4352941176470587),
        "inflammation": (0.7477124183006538, 0.8980392156862746, 0.6274509803921569)
        }

# params 
comparison_col = "Subtype" # x axis comparison col
condition_col = "Treatment" # comparison within comparison col
measure = "Geometric_Area"

# prepare data 
#df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()
#df_clean[measure] = pd.to_numeric(df_clean[measure], errors="coerce")
#df_clean = df_clean.dropna(subset=[measure]).copy()

subtypes = sorted(pd.unique(df_combined[comparison_col]).tolist())
hue_order = ["control", "inflammation"]

# helper for p-values
def mw_p(a_vals, b_vals):
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)

    a_vals = a_vals[~np.isnan(a_vals)]
    b_vals = b_vals[~np.isnan(b_vals)]

    if len(a_vals) < 1 or len(b_vals) < 1:
        return np.nan

    _, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided", method="auto")
    return p

# Build pair lists and p-value lists for each family
within_pairs = []          # ((Line, TP0), (Line, TP4))
within_pvals = []

# Within-line (0 vs 4)
for L in subtypes:
    a = df_combined[(df_combined[comparison_col] == L) & (df_combined[condition_col].astype(str) == "control")][measure].astype(float).values
    b = df_combined[(df_combined[comparison_col] == L) & (df_combined[condition_col].astype(str) == "inflammation")][measure].astype(float).values
    p = mw_p(a, b)
    if not np.isnan(p):
        within_pairs.append(((L, "control"), (L, "inflammation")))
        within_pvals.append(p)

# Apply FDR correction per family
within_corr = multipletests(within_pvals, method="fdr_bh")[1] if within_pvals else []

# Create single grouped violin plot: x=Line, hue=Timepoint (0 and 4)
plt.figure(figsize=(max(8, len(subtypes) * 1.2), 6))
ax = sns.violinplot(data=df_combined, x=comparison_col, y=measure, hue=condition_col,
                    order=subtypes, hue_order=hue_order, split=False, cut=0, palette=cmap)
sns.stripplot(data=df_combined, x=comparison_col, y=measure, hue=condition_col, order=subtypes,
              hue_order=hue_order, dodge=True, size=3, jitter=True, palette=["k","k"], alpha=0.6)

# The overlay above added a second legend; remove duplicate legend entries and keep single
handles, labels = ax.get_legend_handles_labels()
# keep only the first two (hue) and set the legend properly
if len(handles) >= 2:
    new_labels = ["Control", "Inflammation"]  # your custom legend names
   
    ax.legend(
       handles[:2],
       new_labels,
       title=condition_col,
       fontsize=12,        # legend text size
       title_fontsize=14   # legend title size
   )

if within_pairs:
    annot_within = Annotator(ax, within_pairs, data=df_combined, x=comparison_col, y=measure, hue=condition_col,
                             order=subtypes, hue_order=hue_order)
    annot_within.configure(text_format="star", loc="inside")
    annot_within.set_pvalues_and_annotate(list(within_pvals))

ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis='both', labelsize=14)
#ax.set_title(f"{measure}: Time {time0_label} vs {time4_label} per Line; within-line and between-line comparisons")
plt.tight_layout()
plt.show()

#%% effect sizes
show_title = False
table_df = pairwise_mannwhitney_df(df_combined, feature = measure, subtype_col = comparison_col, condition_col = condition_col)

table_df = table_df[["subtype", "p_adj", "cliffs_delta"]]
table_df['p_adj'] = table_df['p_adj'].apply(lambda x: f"{x:.2e}")
table_df['cliffs_delta'] = table_df['cliffs_delta'].round(2)
table_renamed = table_df.rename(columns={"subtype": "Type", "p_adj": r"$p$-value", "cliffs_delta": "Effect Size" })

if show_title:
    plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(4,2), title = f"{feature}")
else:
    plot.create_table(table_renamed, body_fontsize=6, header_fontsize=6, fig_size=(3,1.6), title = "")

#%% feature heatmap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

plot_vertical = False
axes_show = False

comparison_col = "Subtype"
condition_col = "Treatment"
group_a = "inflammation"
group_b = "control"

# all numeric features you want on the x-axis
features = [
    c for c in df_combined.columns
    if c not in [comparison_col, condition_col, "ObjectID", "Metadata_CellID"]
]

features = [feature for feature in features if "Zernike" not in feature]

subtypes = sorted(df_combined[comparison_col].dropna().unique())

def cliffs_delta_from_u(a, b):
    """
    Signed Cliff's delta using Mann-Whitney U.
    Positive => a tends to be larger than b.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    u, _ = mannwhitneyu(a, b, alternative="two-sided", method="auto")
    delta = (2 * u) / (len(a) * len(b)) - 1
    return delta

# Build effect size matrix: rows=subtypes, cols=features
effect_mat = pd.DataFrame(index=subtypes, columns=features, dtype=float)

for subtype in subtypes:
    for feat in features:
        a = df_combined[
            (df_combined[comparison_col] == subtype) &
            (df_combined[condition_col].astype(str) == group_a)
        ][feat].astype(float).values

        b = df_combined[
            (df_combined[comparison_col] == subtype) &
            (df_combined[condition_col].astype(str) == group_b)
        ][feat].astype(float).values

        effect_mat.loc[subtype, feat] = cliffs_delta_from_u(a, b)

# Optional: sort features by overall magnitude or keep original order
# effect_mat = effect_mat[effect_mat.abs().mean().sort_values(ascending=False).index]

def get_feature_color(feature):
    for key, color in constants.FEATURE_TYPE_CMAP.items():
        if key.lower() in feature.lower():
            return color
    return "#cccccc"  # default if no keyword match

if plot_vertical:
    effect_mat_plot = effect_mat.T
    row_colors = [get_feature_color(f) for f in effect_mat.columns]
    col_colors = [c for c in cmap_labelling.values()]
    figsize = (5,8)
    cbar_pos=(-0.2, 0.65, 0.015, 0.3)

else:
    effect_mat_plot = effect_mat
    col_colors = [get_feature_color(f) for f in effect_mat.columns]
    row_colors = [c for c in cmap_labelling.values()]
    figsize = (8,5)
    cbar_pos=(1, 0.65, 0.015, 0.3)

g = sns.clustermap(
    effect_mat_plot, # .T to plot vertically
    cmap="vlag",
    center=0,
    vmin=-0.3,
    vmax=0.3,
    col_colors=col_colors,
    row_cluster=False,
    row_colors=row_colors,
    col_cluster=False,
    linewidths=0.5,
    linecolor="white",
    #cbar_kws={"label": "Cliff's delta (control vs inflammation)"},
    #figsize=(max(10, len(features) * 0.4), max(4, len(subtypes) * 0.4))
    figsize=figsize,
    cbar_pos=cbar_pos,
    dendrogram_ratio = (0.02, 0.02)
)

# Remove x-axis labels completely
def clean_feature_name(name):
    name = name.split("_",1)[-1]

    # replace keywords
    name = name.replace("Densitometric", "Intensity")

    return name

new_labels = [clean_feature_name(feature) for feature in features]

if plot_vertical:
    # effect_mat_plot = effect_mat.T
    x_labels = list(g.data2d.columns)   # subtypes -> 4 labels
    y_labels = [clean_feature_name(f) for f in g.data2d.index]  # features -> many labels
    g.ax_heatmap.set_yticklabels(y_labels, rotation=90)
    if not axes_show:
        g.ax_heatmap.set_xticklabels([])
        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.tick_params(axis='x', bottom=False, top=False)

else:
    # effect_mat_plot = effect_mat
    x_labels = [clean_feature_name(f) for f in g.data2d.columns]  # features -> many labels
    y_labels = list(g.data2d.index)   # subtypes -> 4 labels
    g.ax_heatmap.set_xticklabels(x_labels, rotation=90) 
    if not axes_show:
        g.ax_heatmap.set_yticklabels([])
        g.ax_heatmap.set_ylabel("")
        g.ax_heatmap.tick_params(axis='y', left=False, right=False)


# Keep y labels
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

plt.show()

#%% display overall proportion
treatment = "inflammation"
order = [1, 2, 3, 4]
proportions = lipid_labelled[treatment]['Subtype'].value_counts(normalize=True)
proportions = proportions.reindex(order)

colors = [cmap_labelling[cat] for cat in proportions.index]

plt.figure(figsize=(3,4))
proportions.plot(kind='bar', color = colors)
#plt.ylabel('Proportion')
#plt.title('Category Proportions')
plt.xticks([])
plt.xlabel("")
plt.show()

#%% display overall proportion stacked bar plot
show_axes = False

proportions = {}

for condition, subtypes in subtype_cells.items():
    total = sum(len(values) for values in subtypes.values())
    proportions[condition] = {
        subtype: len(values) / total
        for subtype, values in subtypes.items()
    }

df = pd.DataFrame(proportions).T  # transpose for plotting

colors = [cmap_labelling[subtype] for subtype in df.columns]

fig, ax = plt.subplots(figsize=(4, 4))   # wider or narrower as you prefer

# plot on the provided axes (important!)
df.plot(
    kind="bar",
    stacked=True,
    color=colors,
    width=0.4,    # bar thickness; increase to make bars closer
    ax=ax,
)

# make x-ticks horizontal and nicely aligned
ax.set_xticklabels(df.index, rotation=0, ha="center")

if show_axes: 
    # ylabel and title
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Subtypes per Condition")
    
    
    # Put legend outside on the right and avoid clipping by reserving space
    leg = ax.legend(
        title="Subtype",
        bbox_to_anchor=(1.02, 0.5),   # x just outside axes, y centered
        loc="center left",
        borderaxespad=0
    )

else: 
    ax.legend().remove()
    ax.set_xticks([])

# Reserve space on the right for the legend — tweak the 0.78 value if needed
plt.subplots_adjust(right=0.78)

# Draw once
plt.show()

#%% test difference in proportion

conditions = list(subtype_cells.keys())
types = sorted({t for cond in subtype_cells for t in subtype_cells[cond].keys()})

table = np.array([
    [len(subtype_cells[cond].get(t, [])) for t in types]
    for cond in conditions
], dtype=int)

chi2, p, dof, expected = chi2_contingency(table)

print("table:")
print(table)
print("chi2:", chi2)
print("p-value:", p)
print("dof:", dof)
print("expected:")
print(expected)

residuals = (table - expected) / np.sqrt(expected)

# Cramer's V
n = table.sum()
chi2 = chi2  # from your test
k = min(table.shape)  # smaller dimension

cramers_v = np.sqrt(chi2 / (n * (k - 1)))
print(cramers_v)

#%% z-test
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests

cond1 = subtype_cells["control"]
cond2 = subtype_cells["inflammation"]

# totals per condition
n1 = sum(len(v) for v in cond1.values())
n2 = sum(len(v) for v in cond2.values())

results = []

for g in cond1:
    x1 = len(cond1[g])
    x2 = len(cond2[g])

    stat, pval = proportions_ztest([x2, x1], [n2, n1])

    p1 = x1 / n1
    p2 = x2 / n2
    diff = p2 - p1

    results.append({
        "group": g,
        "p1": p1,
        "p2": p2,
        "diff": diff,
        "direction": "larger" if diff > 0 else "smaller" if diff < 0 else "same",
        "p_value": pval
    })

# multiple testing correction (important!)
pvals = [r["p_value"] for r in results]
adj = multipletests(pvals, method="bonferroni")[1]

for r, p_adj in zip(results, adj):
    r["p_adj"] = p_adj

for r in results:
    print(r)
    
#%% slope plot

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint

cond1 = subtype_cells["control"]
cond2 = subtype_cells["inflammation"]

groups = list(cond1.keys())

# totals per condition
n1 = sum(len(v) for v in cond1.values())
n2 = sum(len(v) for v in cond2.values())

x_pos = np.array([0, 1])  # condition positions

fig, ax = plt.subplots(figsize=(8, 5))

for g in groups:
    x1 = len(cond1[g])
    x2 = len(cond2[g])

    p1 = x1 / n1
    p2 = x2 / n2

    # 95% CI for each proportion (Wilson interval)
    ci1_low, ci1_high = proportion_confint(x1, n1, alpha=0.05, method="wilson")
    ci2_low, ci2_high = proportion_confint(x2, n2, alpha=0.05, method="wilson")

    # line connecting conditions
    ax.plot(x_pos, [p1, p2], marker="o", linewidth=2, alpha=0.8, label=g)

    # error bars
    ax.errorbar([0, 1], [p1, p2],
                yerr=[[p1 - ci1_low, p2 - ci2_low],
                      [ci1_high - p1, ci2_high - p2]],
                fmt="none", capsize=4, alpha=0.6)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Condition 1", "Condition 2"])
ax.set_ylabel("Proportion")
ax.set_title("Change in group proportions between conditions")
ax.set_ylim(0, max(
    max(len(v) for v in cond1.values()) / n1,
    max(len(v) for v in cond2.values()) / n2
) * 1.15)

ax.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

#%% grouped bar chart
import matplotlib.pyplot as plt
import numpy as np

cond1 = subtype_cells["control"]
cond2 = subtype_cells["inflammation"]

groups = list(cond1.keys())

# totals per condition
n1 = sum(len(v) for v in cond1.values())
n2 = sum(len(v) for v in cond2.values())

# proportions
p1 = [len(cond1[g]) / n1 for g in groups]
p2 = [len(cond2[g]) / n2 for g in groups]

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x - width/2, p1, width, label="Condition 1")
ax.bar(x + width/2, p2, width, label="Condition 2")

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("Proportion")
ax.set_title("Group proportions by condition")
ax.legend()

plt.tight_layout()
plt.show()
# %% distribution of lipiddroplets types in each cell

# add cellID back
lipid_labelled["inflammation"]["Metadata_CellID"] = all_sheets_filtered["inflammation"]["lipiddroplets"]["Metadata_CellID"]
plot.stacked_bxplt_count(
    lipid_labelled["inflammation"], "Metadata_CellID", "Subtype", 
    label_color_mapping=cmap_labelling, axis_labels = False,
    legend=False
)

plot.stacked_bxplt_count(
    lipid_labelled["inflammation"],
    "Metadata_CellID",
    "Subtype",
    label_color_mapping=cmap_labelling,
    normalize=True,
    axis_labels=False,
    legend=False
)

#%% per cell average proportion

proportions_per_cell = lipid_labelled.copy()
proportions_per_cell["control"]["Metadata_CellID"] = all_sheets_filtered["control"]["lipiddroplets"]["Metadata_CellID"]
proportions_per_cell["inflammation"]["Metadata_CellID"] = all_sheets_filtered["inflammation"]["lipiddroplets"]["Metadata_CellID"]


for condition in ["control", "inflammation"]:
    df = proportions_per_cell[condition]

    proportions_per_cell[condition] = (
        pd.crosstab(
            df["Metadata_CellID"],
            df["Subtype"],
            normalize="index"
        )
        .reset_index()
    )

long_df = pd.concat([
    df.assign(Condition=cond)
    for cond, df in proportions_per_cell.items()
])

long_df = long_df.melt(
    id_vars=["Metadata_CellID", "Condition"],
    var_name="Subtype",
    value_name="Proportion"
)

from scipy.stats import mannwhitneyu

results = []

for subtype, sub_df in long_df.groupby("Subtype"):
    ctrl = sub_df[sub_df["Condition"] == "control"]["Proportion"]
    infl = sub_df[sub_df["Condition"] == "inflammation"]["Proportion"]

    stat, p = mannwhitneyu(ctrl, infl, alternative="two-sided")

    results.append({
        "Subtype": subtype,
        "p_value": p,
        "control_mean": ctrl.mean(),
        "inflammation_mean": infl.mean()
    })

results = pd.DataFrame(results)

#%% types of lipiddroplets cells

lipid_cell_types = lipid_labelled.copy()
lipid_cell_types["control"]["Metadata_CellID"] = all_sheets_filtered["control"]["lipiddroplets"]["Metadata_CellID"]
lipid_cell_types["inflammation"]["Metadata_CellID"] = all_sheets_filtered["inflammation"]["lipiddroplets"]["Metadata_CellID"]

lipid_type = {}
cluster_df = {}
scaler = StandardScaler()

for treatment in ["control", "inflammation"]:
    # convert subtype composition per cell into cell measures
    lipid_type[treatment] = (
        lipid_cell_types[treatment]
        .groupby('Metadata_CellID')['Subtype']
        .value_counts(normalize=True)   # gives proportions instead of counts
        .unstack(fill_value=0)         # columns = types (1,2,3,4)
        #.add_prefix("lipiddroplets_")
        .reset_index()
    )


    comp_id = lipid_type[treatment][["Metadata_CellID"]].copy()
    comp_props = lipid_type[treatment].drop(columns=["Metadata_CellID"])
    
    # CLR transform on proportions
    comp_props_clr = clr_transform(comp_props)
    
    # optional: standardize after CLR
    if treatment == "control":
        comp_props_clr_scaled = pd.DataFrame(
            scaler.fit_transform(comp_props_clr),
            index=comp_props_clr.index,
            columns=comp_props_clr.columns
        )
    else: 
        comp_props_clr_scaled = pd.DataFrame(
            scaler.transform(comp_props_clr),
            index=comp_props_clr.index,
            columns=comp_props_clr.columns
        )
    
    # combine back
    lipid_clr = pd.concat([comp_id, comp_props_clr_scaled], axis=1)
    cluster_df[treatment] = lipid_clr.copy()

#%%
treatment = "control"

cell_labelled, _ = add_cluster_label(
    cluster_df[treatment],
    cluster_num=4,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_,_,cmap_labelling = label_color_mapping_dict(
    cell_labelled["Subtype"], palette="crest"
)

_, clustermap_cmap = plot.show_clustermap(
    cell_labelled, 
    color_row_cmap=cmap_labelling, 
    clustering_algorithm = "ward",
    feature_cluster = False,
    color_feat=None, 
    label_col="Subtype", 
    id_col=constants.CELLID_COL, 
    scale = False, 
    pltsize=(25,25), 
    score_range=(-1.5,1.5),
    cmap = "viridis",
    all_legends = False,
    #plot_title = f"{treatment} Clustermap"
)


labels = cell_labelled["Subtype"]

#%% composition - cell type control 
treatment = "inflammation"
order = [1, 2, 3, 4]
proportions = cell_labelled['Subtype'].value_counts(normalize=True)
proportions = proportions.reindex(order)

colors = [cmap_labelling[cat] for cat in proportions.index]

plt.figure(figsize=(3,4))
proportions.plot(kind='bar', color = colors)
#plt.ylabel('Proportion')
#plt.title('Category Proportions')
plt.xticks([])
plt.xlabel("")
plt.show()

#%%
cell_labelled = {}
subtype_cells = {}

clusternum = 4

cell_labelled["control"], _ = add_cluster_label(
    cluster_df["control"],
    cluster_num=clusternum,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

rf_model, data_scaler, scores, scores_mean, scores_std, feature_importance = (
    random_forest_train_model(
        cell_labelled["control"], "Subtype", savedir=None, scale = False, id_col = "Metadata_CellID"
    )
)

subpopulation_cells = {}
for i in list(range(1, clusternum + 1)):  # start subpopulation count at 1
    subpopulation_cells[i] = cell_labelled["control"].loc[
        cell_labelled["control"]["Subtype"] == i, "Metadata_CellID"
    ].tolist()

subtype_cells["control"] = subpopulation_cells

cell_labelled["inflammation"], subpopulation_lipid_rf, sorted_obj, feature_importance = random_forest_classification(
    rf_model, data_scaler, cluster_df["inflammation"], id_col="Metadata_CellID", scale=False
)
cell_labelled["inflammation"] = cell_labelled["inflammation"].rename(
    columns={'Random_forest_population': 'Subtype'}
) 

subpopulation_cells = {}
for i in list(range(1, clusternum + 1)):  # start subpopulation count at 1
    subpopulation_cells[i] = cell_labelled["inflammation"].loc[
        cell_labelled["inflammation"]["Subtype"] == i, "Metadata_CellID"
    ].tolist()

subtype_cells["inflammation"] = subpopulation_cells

#%% differences in lipid type composition with inflammation
pop = 1
pop_df = cell_labelled.copy()
pop_df["inflammation"] = pop_df["inflammation"][["Metadata_CellID", 1, 2, 3, 4, "Subtype"]]

df_combined_full = pd.concat(pop_df, names=["Treatment"]).reset_index(level=0)

cmap = {
        "control": (0.9934640522875817, 0.7477124183006535, 0.4352941176470587),
        "inflammation": (0.7477124183006538, 0.8980392156862746, 0.6274509803921569)
        }

# params 
comparison_col = "Type" # x axis comparison col
condition_col = "Treatment" # comparison within comparison col
measure = "CLR"

type_cols = [1, 2, 3, 4]
df_combined = df_combined_full.melt(
    id_vars=["Subtype", condition_col],
    value_vars=type_cols,
    var_name="Type",
    value_name="CLR"
)
# prepare data 
#df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()
#df_clean[measure] = pd.to_numeric(df_clean[measure], errors="coerce")
#df_clean = df_clean.dropna(subset=[measure]).copy()

df_combined = df_combined[df_combined["Subtype"] == pop]

subtypes = sorted(pd.unique(df_combined[comparison_col]).tolist())
hue_order = ["control", "inflammation"]

# helper for p-values
def mw_p(a_vals, b_vals):
    a_vals = np.asarray(a_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)

    a_vals = a_vals[~np.isnan(a_vals)]
    b_vals = b_vals[~np.isnan(b_vals)]

    if len(a_vals) < 1 or len(b_vals) < 1:
        return np.nan

    _, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided", method="auto")
    return p

# Build pair lists and p-value lists for each family
within_pairs = []          # ((Line, TP0), (Line, TP4))
within_pvals = []

# Within-line (0 vs 4)
for L in subtypes:
    a = df_combined[(df_combined[comparison_col] == L) & (df_combined[condition_col].astype(str) == "control")][measure].astype(float).values
    b = df_combined[(df_combined[comparison_col] == L) & (df_combined[condition_col].astype(str) == "inflammation")][measure].astype(float).values
    p = mw_p(a, b)
    if not np.isnan(p):
        within_pairs.append(((L, "control"), (L, "inflammation")))
        within_pvals.append(p)

# Apply FDR correction per family
within_corr = multipletests(within_pvals, method="fdr_bh")[1] if within_pvals else []

# Create single grouped violin plot: x=Line, hue=Timepoint (0 and 4)
plt.figure(figsize=(max(8, len(subtypes) * 1.2), 6))
ax = sns.violinplot(data=df_combined, x=comparison_col, y=measure, hue=condition_col,
                    order=subtypes, hue_order=hue_order, split=False, cut=0, palette=cmap)
sns.stripplot(data=df_combined, x=comparison_col, y=measure, hue=condition_col, order=subtypes,
              hue_order=hue_order, dodge=True, size=3, jitter=True, palette=["k","k"], alpha=0.6)

# The overlay above added a second legend; remove duplicate legend entries and keep single
handles, labels = ax.get_legend_handles_labels()
# keep only the first two (hue) and set the legend properly
if len(handles) >= 2:
    new_labels = ["Control", "Inflammation"]  # your custom legend names
   
    ax.legend(
       handles[:2],
       new_labels,
       title=condition_col,
       fontsize=12,        # legend text size
       title_fontsize=14   # legend title size
   )

if within_pairs:
    annot_within = Annotator(ax, within_pairs, data=df_combined, x=comparison_col, y=measure, hue=condition_col,
                             order=subtypes, hue_order=hue_order)
    annot_within.configure(text_format="star", loc="inside")
    annot_within.set_pvalues_and_annotate(list(within_pvals))

ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis='both', labelsize=14)
#ax.set_title(f"{measure}: Time {time0_label} vs {time4_label} per Line; within-line and between-line comparisons")
plt.tight_layout()
plt.show()

#%% cell lipiddroplets type proportion
#_, _, cmap_labelling = label_color_mapping_dict(
#cell_labelled["control"]["Subtype"], palette="crest"
#)

show_axes = False

proportions = {}

for condition, subtypes in subtype_cells.items():
    total = sum(len(values) for values in subtypes.values())
    proportions[condition] = {
        subtype: len(values) / total
        for subtype, values in subtypes.items()
    }

df = pd.DataFrame(proportions).T  # transpose for plotting

colors = [cmap_labelling[subtype] for subtype in df.columns]

fig, ax = plt.subplots(figsize=(4, 4))   # wider or narrower as you prefer

# plot on the provided axes (important!)
df.plot(
    kind="bar",
    stacked=True,
    color=colors,
    width=0.4,    # bar thickness; increase to make bars closer
    ax=ax,
)

# make x-ticks horizontal and nicely aligned
ax.set_xticklabels(df.index, rotation=0, ha="center")

if show_axes: 
    # ylabel and title
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Subtypes per Condition")
    
    
    # Put legend outside on the right and avoid clipping by reserving space
    leg = ax.legend(
        title="Subtype",
        bbox_to_anchor=(1.02, 0.5),   # x just outside axes, y centered
        loc="center left",
        borderaxespad=0
    )

else: 
    ax.legend().remove()
    ax.set_xticks([])

# Reserve space on the right for the legend — tweak the 0.78 value if needed
plt.subplots_adjust(right=0.78)

# Draw once
plt.show()

# compare proportions - no significant difference
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

def compare_group_proportions(data, cond1="control", cond2="inflammation",
                              p_adjust_method="holm"):
    d1 = data[cond1]
    d2 = data[cond2]

    groups = sorted(set(d1.keys()) | set(d2.keys()))
    n1 = sum(len(v) for v in d1.values())
    n2 = sum(len(v) for v in d2.values())

    rows = []

    for g in groups:
        x1 = len(d1.get(g, []))
        x2 = len(d2.get(g, []))

        # 2x2 table:
        #            group g   not group g
        # cond1         x1       n1 - x1
        # cond2         x2       n2 - x2
        table = [
            [x1, n1 - x1],
            [x2, n2 - x2],
        ]

        oddsratio, pval = fisher_exact(table, alternative="two-sided")

        p1 = x1 / n1 if n1 else float("nan")
        p2 = x2 / n2 if n2 else float("nan")
        diff = p2 - p1

        rows.append({
            "group": g,
            "count_control": x1,
            "count_inflammation": x2,
            "prop_control": p1,
            "prop_inflammation": p2,
            "diff": diff,
            "direction": "larger" if diff > 0 else "smaller" if diff < 0 else "same",
            "oddsratio": oddsratio,
            "p_value": pval
        })

    out = pd.DataFrame(rows)
    out["p_adj"] = multipletests(out["p_value"], method=p_adjust_method)[1]
    return out

results = compare_group_proportions(subtype_cells)
print(results)
#%% evaluation of clustering
individual_scores_df = silhouette_score_indiv(
    lipid_scaled["inflammation"],
    lipid_labelled["inflammation"]["Subtype"],
    scale=False,
    cellid_col="ObjectID",
)
silhouette_score_stats_dict = silhouette_score_stats(individual_scores_df, id_col = "ObjectID")
counts = lipid_labelled["inflammation"]['Subtype'].value_counts()


#%% example lipiddroplets per type

max_representative = 30
pop = 4
organelle = "lipiddroplets"

representative_obj = sorted_obj[pop]
#for pop in list(range(1, cluster_num)):
    
org.plot_selected(
    all_segmentation,
    organelle,
    silhouette_score_stats_dict = None,
    list_of_obj = representative_obj,
    readfile=False,
    max_display=max_representative,
    population=pop,
    xray_overlay=True,
    xray_paths=xray_paths,
    cropped_roi=True,
    cropped_xray = True, 
    lw = 2
)

#%% plot confidence 
confidence = cell_labelled["inflammation"]["RF_confidence"]
sorted_conf = np.sort(confidence)[::-1]  # high to low

plt.figure(figsize=(5, 5))
plt.plot(sorted_conf)
#plt.xlabel("Samples sorted by confidence")
#plt.ylabel("Max class probability")
#plt.title("Random Forest confidence dropoff")
plt.axhline(y=0.8, color='red', linestyle='--', linewidth=2)
plt.show()


plt.figure(figsize=(5, 4))
plt.hist(confidence, bins=50)
#plt.xlabel("Max class probability")
#plt.ylabel("Count")
#plt.title("Confidence distribution")
plt.axvline(x=0.85, color='red', linestyle='--', linewidth=2)

plt.show()

plt.figure(figsize=(8, 4))
plt.boxplot(confidence, vert=False)
plt.xlabel("Max class probability")
plt.title("Confidence spread")
plt.show()

#%% identify new classes
labelled_inf = cell_labelled["inflammation"].copy()
confidence = labelled_inf["RF_confidence"]

confidence_filtered =  labelled_inf[labelled_inf["RF_confidence"] > 0.85]
confidence_low =  labelled_inf[labelled_inf["RF_confidence"] <= 0.85]

low_confidence_df = confidence_low[["Metadata_CellID", 1, 2, 3, 4]]
outlier_labelled, _ = add_cluster_label(
    low_confidence_df,
    cluster_num=2,
    clustering_algorithm="HCA",
    id_col=constants.CELLID_COL,
    lbl_colname="Subtype",
)

_, clustermap_cmap = plot.show_clustermap(
    outlier_labelled, 
    #color_row_cmap=cmap_labelling, 
    clustering_algorithm = "ward",
    feature_cluster = False,
    color_feat=None, 
    label_col="Subtype", 
    id_col=constants.CELLID_COL, 
    scale = False, 
    pltsize=(25,25), 
    score_range=(-2,2),
    cmap = "viridis",
    all_legends = False,
    #plot_title = f"{treatment} Clustermap"
)

margins = labelled_inf["RF_margin"]
margins_high = confidence_filtered["RF_margin"]
margins_low = confidence_low["RF_margin"]

entropy = labelled_inf["RF_entropy"]
entropy_high = confidence_filtered["RF_entropy"]
entropy_low = confidence_low["RF_entropy"]

new_cells = list(confidence_low["Metadata_CellID"])
labelled_inf_new = labelled_inf[["Metadata_CellID", 1, 2, 3, 4, "Subtype"]]
labelled_inf_new = labelled_inf_new.copy()
labelled_inf_new["Subtype_new"] = labelled_inf_new["Subtype"]

labelled_inf_new.loc[labelled_inf_new['Metadata_CellID'].isin(new_cells), 'Subtype_new'] = 5

subpopulation_cells = {}
for i in list(range(1, 5 + 1)):  # start subpopulation count at 1
    subpopulation_cells[i] = labelled_inf_new.loc[
        labelled_inf_new["Subtype_new"] == i, "Metadata_CellID"
    ].tolist()

subtype_cells["inflammation"] = subpopulation_cells
subtype_cells["control"][5] = []
#%%

plt.figure(figsize=(6, 6))
plt.scatter(confidence, entropy, alpha=0.5)

plt.xlabel("Confidence (max probability)")
plt.ylabel("Margin (top1 - top2)")
plt.title("Confidence vs Margin")

plt.grid(True)
plt.show()

#%%

df = pd.DataFrame({
    "margins": np.concatenate([margins_high, margins_low]),
    "group": ["high"] * len(margins_high) + ["low"] * len(margins_low)
})

sns.histplot(data=df, x='margins', bins = 20, hue = "group", kde = False, color = "blue", alpha = 0.5, legend = False)

plt.ylabel("")
plt.xlabel("")
plt.show()

#%%
df = pd.DataFrame({
    "entropy": np.concatenate([entropy_high, entropy_low]),
    "group": ["high"] * len(entropy_high) + ["low"] * len(entropy_low)
})


# Example: df is your DataFrame and 'column_name' is your column
sns.histplot(data=df, x='entropy', bins = 20, hue = "group", kde = False, color = "blue", alpha = 0.5, legend = False)

plt.ylabel("")
plt.xlabel("")
plt.show()


#%% stacked bar plot of subtype combinations
_,_,cmap_labelling = label_color_mapping_dict(
    labelled_inf_new["Subtype_new"], palette="crest"
)
show_axes = True

proportions = {}

for condition, subtypes in subtype_cells.items():
    total = sum(len(values) for values in subtypes.values())
    proportions[condition] = {
        subtype: len(values) / total
        for subtype, values in subtypes.items()
    }

df = pd.DataFrame(proportions).T  # transpose for plotting

colors = [cmap_labelling[subtype] for subtype in df.columns]

fig, ax = plt.subplots(figsize=(4, 4))   # wider or narrower as you prefer

# plot on the provided axes (important!)
df.plot(
    kind="bar",
    stacked=True,
    color=colors,
    width=0.4,    # bar thickness; increase to make bars closer
    ax=ax,
)

# make x-ticks horizontal and nicely aligned
ax.set_xticklabels(df.index, rotation=0, ha="center")

if show_axes: 
    # ylabel and title
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Subtypes per Condition")
    
    
    # Put legend outside on the right and avoid clipping by reserving space
    leg = ax.legend(
        title="Subtype",
        bbox_to_anchor=(1.02, 0.5),   # x just outside axes, y centered
        loc="center left",
        borderaxespad=0
    )

else: 
    ax.legend().remove()
    ax.set_xticks([])

# Reserve space on the right for the legend — tweak the 0.78 value if needed
plt.subplots_adjust(right=0.78)

# Draw once
plt.show()

#%% plot example of new population 

for cell in new_cells:

    xray_img = tiff.imread(xray_paths[cell])
    lipid_seg_df = all_segmentation["lipiddroplets"]
    lipid_imgs = lipid_seg_df[lipid_seg_df["Metadata_CellID"] == cell]
    
    outlines =dict(zip('lipiddroplets_' + lipid_imgs['layer'].astype(str), lipid_imgs['image']))

    plot_overlays_masks_cell(base_img = xray_img,
                             outlines = outlines,
                             rotation = 90,
                             )

#%% spatial analysis
_, _, cmap_labelling = label_color_mapping_dict(
    lipid_labelled["control"]["Subtype"], palette="inferno"
)

original_df = all_sheets_original["inflammation"].copy()
sheets_df = all_sheets["inflammation"].copy()

classification_df = context_classification_lipid(
    original_df,
    all_segmentation,
    sheets_df["lipiddroplets"]["ObjectID"],
    #DEBUG = False
)

#if not in cell edge or perinuclear then cytoplasm 
classification_df['Classification_Cytoplasm'] = ((classification_df['Classification_Perinuclear'] == 0) & (classification_df['Classification_CellEdge'] == 0)).astype(int)
classification_df['Classification_CellEdge_only'] = ((classification_df['Classification_CellEdge'] == 1) & (classification_df['Classification_Perinuclear'] == 0)).astype(int)
classification_df['Classification_Perinuclear_only'] = ((classification_df['Classification_CellEdge'] == 0) & (classification_df['Classification_Perinuclear'] == 1)).astype(int)
classification_df['Classification_Squished'] = ((classification_df['Classification_CellEdge'] == 1) & (classification_df['Classification_Perinuclear'] == 1)).astype(int)


classification_list = [x for x in classification_df.columns if "Classification" in x]

#%% subtypes in each classification 
stack_order = [1, 2, 3, 4]
colors = [cmap_labelling[val] for val in stack_order]

classification_plot = classification_df.copy()
classification_plot = classification_df.merge(lipid_labelled["inflammation"][['ObjectID', 'Subtype']],
                                            on='ObjectID',
                                            how='left'
                                            )
#classification_plot = classification_plot[classification_plot["CellID"]=="CELL001"]

cols = ["Classification_Perinuclear_only", "Classification_Cytoplasm", "Classification_Cluster", "Classification_CellEdge_only"]#, "Classification_Squished"]
cols = ["Classification_Perinuclear_only", "Classification_Perimitochondrial", "Classification_Cytoplasm", "Classification_Cluster", "Classification_CellEdge_only"]#, "Classification_Squished"]

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

#%% test enrichment in subtype in location (no enrichment)
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
axes = True

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