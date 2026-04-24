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
    label_color_mapping_dict,
    random_forest_train_model,
    random_forest_classification,
    model_eval
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
import joblib
import sklearn.preprocessing as skprep
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from data_analysis_figs import plot_overlays_masks_cell
from statannotations.Annotator import Annotator

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

treatment = "inflammation"
run = "run4"
parentdir = Path(constants.PARENTDIR / f"{treatment}" / f"{run}")
keep_cols_reduced = pd.read_excel(
    parentdir / "selected_features_organelle.xlsx", sheet_name=None, header=None
)

segmentationdir = parentdir / "results"
xray_paths = load_path_into_dict(segmentationdir, keyword="xray")
all_segmentation = load_segmentation(
    segmentationdir, organelle=constants.ORGANELLE_CP_LIST
)
#rf_model = joblib.load(Path(constants.PARENTDIR / "control" / "run4" / "random_forest_model.joblib"))
#data_scaler = joblib.load(Path(constants.PARENTDIR / "control" / "run4" / "random_forest_data_scalar.joblib"))

#%% label inflamed populations - multi-copy organelles
clusternum = 4

cluster_df_inf_multi = {}
cluster_df_ctrl_multi = {}

multi_organelles = {"mitochondria": 4, "lipiddroplets": 4}

for organelle, clusternum in multi_organelles.items():
    
    subtype_cells = {}
    all_sheets = {}
    all_sheets_original = {}
    props = {}
    props_renamed = {}
    props_scaled = {}
    props_labelled = {}
    all_sheets_filtered = {}
    organelle_type = {}
    cluster_df = {}
    clr_scaler = StandardScaler()
    
    for treatment in ["control", "inflammation"]:
        
        datadir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
        filedir = datadir / "cellprofiler" / "measurements_updated"
        all_sheets_original[treatment] = combine_files_to_sheets(filedir)
    
        all_sheets[treatment] = all_sheets_original[treatment].copy()
        
        if organelle == "mitochondria" or organelle == "lipiddroplets":
            all_sheets[treatment][f"{organelle}"] = utils.update_cellID_objectnum(
                all_sheets[treatment][f"{organelle}"],
                "ImageNumber",
                "ObjectNumber",
                "Metadata_CellID",
            )
            all_sheets_filtered[treatment] = filter_dataframes(all_sheets[treatment], keep_cols_reduced)

        else: 
            all_sheets_filtered[treatment] = filter_dataframes(all_sheets[treatment], keep_cols_reduced)

    
    
        props[treatment] = all_sheets_filtered[treatment][f"{organelle}"].copy()
        props[treatment].loc[:, "ObjectID"] = all_sheets[treatment][f"{organelle}"]["ObjectID"]
        props[treatment] = props[treatment].drop("Metadata_CellID", axis=1)
    
        props_renamed[treatment] = rename_df_columns_with_keyword(props[treatment], rename_dict)
    
        feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]
    
        if treatment == "control": # new label
            props_scaled[treatment], scaler = scale_data_fit(props_renamed[treatment], id_col = "ObjectID")
            feature_groups = {}
            for feature in feature_types:
                cols = props_scaled[treatment].columns[
                    props_scaled[treatment].columns.str.contains(feature, case=False)
                ]
                
                feature_groups[feature] = cols
            
            props_normalised = apply_group_normalisation(props_scaled[treatment], feature_groups)
            
            props_labelled[treatment], _ = add_cluster_label(
                props_normalised,
                cluster_num=clusternum,
                clustering_algorithm="HCA",
                id_col="ObjectID",
                lbl_colname="Subtype",
            )
    
            rf_model, data_scaler, scores, scores_mean, scores_std, feature_importance = (
                random_forest_train_model(
                    props_labelled[treatment], "Subtype", savedir=datadir, scale = False, id_col = "ObjectID"
                )
            )
            
            X_scaled = props_normalised.drop(columns="ObjectID")
            Y = props_labelled[treatment]["Subtype"]
            
            eval_dict = model_eval(rf_model, X_scaled,Y)
                      
        elif treatment == "inflammation":
            props_scaled[treatment] = scale_data_transform(props_renamed[treatment], scaler, id_col = "ObjectID")
            props_normalised = apply_group_normalisation(props_scaled[treatment], feature_groups)
    
            props_labelled[treatment], subpopulation_lipid_rf, sorted_obj, feature_importance = random_forest_classification(
                rf_model, data_scaler, props_normalised, id_col="ObjectID", scale=False
            )
            props_labelled[treatment] = props_labelled[treatment].rename(
                columns={'Random_forest_population': 'Subtype'}
            )        
        
        pops_labelled_cell = props_labelled[treatment].copy()
        pops_labelled_cell["Metadata_CellID"] = all_sheets_filtered[treatment][f"{organelle}"]["Metadata_CellID"]
        
        # convert subtype composition per cell into cell measures
        organelle_type[treatment] = (
            pops_labelled_cell
            .groupby('Metadata_CellID')['Subtype']
            .value_counts(normalize=True)   # gives proportions instead of counts
            .unstack(fill_value=0)         # columns = types (1,2,3,4)
            .add_prefix(f"{organelle}_")
            .reset_index()
        )


        comp_id = organelle_type[treatment][["Metadata_CellID"]].copy()
        comp_props = organelle_type[treatment].drop(columns=["Metadata_CellID"])
        
        # CLR transform on proportions
        comp_props_clr = clr_transform(comp_props)
        
        # optional: standardize after CLR
        if treatment == "control": # must come first in the list to fit scaler
            comp_props_clr_scaled = pd.DataFrame(
                clr_scaler.fit_transform(comp_props_clr),
                index=comp_props_clr.index,
                columns=comp_props_clr.columns
            )
        else: 
            comp_props_clr_scaled = pd.DataFrame(
                clr_scaler.transform(comp_props_clr),
                index=comp_props_clr.index,
                columns=comp_props_clr.columns
            )
        
        # combine back
        clr = pd.concat([comp_id, comp_props_clr_scaled], axis=1)
        cluster_df[treatment] = clr.copy()
        
        subpopulation_cells = {}
        for i in list(range(1, clusternum + 1)):  # start subpopulation count at 1
            subpopulation_cells[i] = props_labelled[treatment].loc[
                props_labelled[treatment]["Subtype"] == i, "ObjectID"
            ].tolist()
        
        subtype_cells[treatment] = subpopulation_cells
    
        
    cluster_df_inf_multi[f"{organelle}"] = cluster_df["inflammation"]
    cluster_df_ctrl_multi[f"{organelle}"] = cluster_df["control"]

cluster_df_full_inf = cluster_df_inf_multi.copy()
cluster_df_full_ctrl = cluster_df_ctrl_multi.copy()
#%%
_, _, cmap_labelling = label_color_mapping_dict(
    props_labelled["control"]["Subtype"], palette="inferno"
)

#%% label inflamed populations - single copy organelle

cluster_df_inf_single = {}
cluster_df_ctrl_single = {}

single_organelle = { "nucleus": 5, "cell": 3, "cytoplasm": 3}

for organelle, clusternum in single_organelle.items():
    
    subtype_cells = {}
    all_sheets = {}
    all_sheets_original = {}
    props = {}
    props_renamed = {}
    props_scaled = {}
    props_labelled = {}
    all_sheets_filtered = {}
    organelle_type = {}
    cluster_df = {}
    clr_scaler = StandardScaler()
    
    for treatment in ["control", "inflammation"]:
        
        datadir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
        filedir = datadir / "cellprofiler" / "measurements_updated"
        all_sheets_original[treatment] = combine_files_to_sheets(filedir)
    
        all_sheets[treatment] = all_sheets_original[treatment].copy()
        all_sheets_filtered[treatment] = filter_dataframes(all_sheets[treatment], keep_cols_reduced)

        props[treatment] = all_sheets_filtered[treatment][f"{organelle}"].copy()    
        props_renamed[treatment] = rename_df_columns_with_keyword(props[treatment], rename_dict)
    
        feature_types = ["Geometric", "Densitometric", "Textural", "Structural"]
    
        if treatment == "control": # new label
            props_scaled[treatment], scaler = scale_data_fit(props_renamed[treatment], id_col = "Metadata_CellID")
            feature_groups = {}
            for feature in feature_types:
                cols = props_scaled[treatment].columns[
                    props_scaled[treatment].columns.str.contains(feature, case=False)
                ]
                
                feature_groups[feature] = cols
            
            props_normalised = apply_group_normalisation(props_scaled[treatment], feature_groups)
            
            props_labelled[treatment], _ = add_cluster_label(
                props_normalised,
                cluster_num=clusternum,
                clustering_algorithm="HCA",
                id_col="Metadata_CellID",
                lbl_colname="Subtype",
            )
    
            rf_model, data_scaler, scores, scores_mean, scores_std, feature_importance = (
                random_forest_train_model(
                    props_labelled[treatment], "Subtype", savedir=datadir, scale = False, id_col = "Metadata_CellID"
                )
            )
            
            X_scaled = props_normalised.drop(columns="Metadata_CellID")
            Y = props_labelled[treatment]["Subtype"]
            
            eval_dict = model_eval(rf_model, X_scaled,Y)
                      
        elif treatment == "inflammation":
            props_scaled[treatment] = scale_data_transform(props_renamed[treatment], scaler, id_col = "Metadata_CellID")
            props_normalised = apply_group_normalisation(props_scaled[treatment], feature_groups)
    
            props_labelled[treatment], subpopulation_lipid_rf, sorted_obj, feature_importance = random_forest_classification(
                rf_model, data_scaler, props_normalised, id_col="Metadata_CellID", scale=False
            )
            props_labelled[treatment] = props_labelled[treatment].rename(
                columns={'Random_forest_population': 'Subtype'}
            )        
        
        pops_labelled_cell = props_labelled[treatment].copy()
        
        # convert subtype composition per cell into cell measures
        
        organelle_type[treatment] = (
            pops_labelled_cell
            .groupby('Metadata_CellID')['Subtype']
            .value_counts(normalize=True)   # gives proportions instead of counts
            .unstack(fill_value=0)          # columns = types (1,2,3,4)
            .add_prefix(f"{organelle}_")
            .reset_index()
        )

        
        # combine back
        clr = pd.concat([comp_id, comp_props_clr_scaled], axis=1)
        cluster_df[treatment] = clr.copy()
        
        subpopulation_cells = {}
        for i in list(range(1, clusternum + 1)):  # start subpopulation count at 1
            subpopulation_cells[i] = props_labelled[treatment].loc[
                props_labelled[treatment]["Subtype"] == i, "Metadata_CellID"
            ].tolist()
        
        subtype_cells[treatment] = subpopulation_cells
    
    cluster_df_inf_single[f"{organelle}"] = organelle_type["inflammation"]
    cluster_df_ctrl_single[f"{organelle}"] = organelle_type["control"]

#%% context features

context_measures = ["mitochondria", "lipiddroplets", "cell"]

cluster_df_inf_context = {}
cluster_df_ctrl_context = {}
context_df_inf_context = {}
context_df_ctrl_context = {}

for organelle in context_measures:
    
    keep_cols_cell = {}
    sheet = {}
    corrected = {}
    context = {}
    
    for treatment in ["control", "inflammation"]:
        
        datadir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
        filedir = datadir / "cellprofiler" / "measurements_updated"
        keep_cols_cell[f"{organelle}"] = keep_cols_reduced[f"whole_cell_{organelle}"]

        all_sheets_original[treatment] = combine_files_to_sheets(filedir)
    
        sheet[f"{organelle}"] = all_sheets_original[treatment][f"{organelle}"].copy()
        all_sheets_filtered[treatment] = filter_dataframes(sheet, keep_cols_cell)

        props[treatment] = all_sheets_filtered[treatment][f"{organelle}"].copy()    
        props_renamed[treatment] = rename_df_columns_with_keyword(props[treatment], rename_dict)
        
        context[treatment] = (props_renamed[treatment]
                              .drop_duplicates(subset="Metadata_CellID")
                              .rename(columns=lambda c: c if c == "Metadata_CellID" else f"{organelle}_context_{c}"))
        
        if treatment == "control": # new label
            props_scaled[treatment], scaler = scale_data_fit(props_renamed[treatment], id_col = "Metadata_CellID")
        
        else:
            props_scaled[treatment] = scale_data_transform(props_renamed[treatment], scaler, id_col = "Metadata_CellID")
            
            
        if organelle == "mitochondria": #normalize by network measures
    
            mito_props_network_corrected = props_scaled[treatment].copy()
            network = mito_props_network_corrected.columns[mito_props_network_corrected.columns.str.contains("Network|Branch", case=False)]
            mito_props_network_corrected[network] /= np.sqrt(len(network)) 
            corrected[treatment] = (
                mito_props_network_corrected
                .drop_duplicates(subset="Metadata_CellID")
                .rename(columns=lambda c: c if c == "Metadata_CellID" else f"mitochondria_context_{c}")
                )
            
        elif organelle == "lipiddroplets":
            lipid_corrected = props_scaled[treatment].copy()
            corrected[treatment] = (
                lipid_corrected
                .drop_duplicates(subset="Metadata_CellID")
                .rename(columns=lambda c: c if c == "Metadata_CellID" else f"lipiddroplets_context_{c}")
                )
            
        elif organelle == "cell":
            corrected[treatment] = props_scaled[treatment].copy().rename(columns=lambda c: c if c == "Metadata_CellID" else f"cell_context_{c}")
    
    cluster_df_inf_context[f"context_{organelle}"] = corrected["inflammation"]
    cluster_df_ctrl_context[f"context_{organelle}"] = corrected["control"]
    context_df_ctrl_context[organelle] = context["control"]
    context_df_inf_context[organelle] = context["inflammation"]

context_df_all = {
    "control": context_df_ctrl_context,
    "inflammation": context_df_inf_context
    }
#%% combine into full dataframe
from functools import reduce
cluster_full = {}
context_full = {}

# combine inflammation 
cluster_full_multi_inf = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    cluster_df_inf_multi.values()
)

cluster_full_single_inf = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    cluster_df_inf_single.values()
)

cluster_full_context_inf = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    cluster_df_inf_context.values()
)

# combine control
cluster_full_multi_ctrl = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    cluster_df_ctrl_multi.values()
)

cluster_full_single_ctrl = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    cluster_df_ctrl_single.values()
)

cluster_full_context_ctrl = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    cluster_df_ctrl_context.values()
)

context_full["inflammation"] = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    context_df_inf_context.values()
)
context_full["control"] = reduce(
    lambda left, right: pd.merge(left, right, on='Metadata_CellID', how = "outer"),
    context_df_ctrl_context.values()
)

cluster_full["inflammation"] = cluster_full_multi_inf.merge(cluster_full_single_inf, on='Metadata_CellID').merge(cluster_full_context_inf, on='Metadata_CellID')
cluster_full["control"] = cluster_full_multi_ctrl.merge(cluster_full_single_ctrl, on='Metadata_CellID').merge(cluster_full_context_ctrl, on='Metadata_CellID')

cluster_full["inflammation"] = cluster_full["inflammation"].reindex(columns=cluster_full["inflammation"].columns.union(cluster_full["control"].columns), fill_value=0)


#%% cluster on whole cell

cell_labelled = {}
subtype_cells = {}

clusternum = 4

cell_labelled["control"], _ = add_cluster_label(
    cluster_full["control"],
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

column_match = [
    c for c in cluster_full["control"].columns
    if c not in ["Subtype"]
]
cluster_full["inflammation"] = cluster_full["inflammation"].reindex(columns=column_match)

cell_labelled["inflammation"], subpopulation_lipid_rf, sorted_obj, feature_importance = random_forest_classification(
    rf_model, data_scaler, cluster_full["inflammation"], id_col="Metadata_CellID", scale=False
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

#%% cell lipiddroplets type proportion
_, _, cmap_labelling = label_color_mapping_dict(
cell_labelled["control"]["Subtype"], palette="crest"
)

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

#%% plot confidence - too messy no clear cutoff
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
plt.hist(confidence, bins=20)
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

#%% identify new class - does not really work cannot tell difference 
new_cell_clusters = 2
subtype_cells_new = subtype_cells.copy()
labelled_inf = cell_labelled["inflammation"].copy()
confidence = labelled_inf["RF_confidence"]

confidence_filtered =  labelled_inf[labelled_inf["RF_confidence"] > 0.55]
confidence_low =  labelled_inf[labelled_inf["RF_confidence"] <= 0.55]

low_confidence_df = confidence_low.drop(columns = ["RF_confidence", "RF_margin", "RF_entropy", "Subtype"])
outlier_labelled, _ = add_cluster_label(
    low_confidence_df,
    cluster_num=new_cell_clusters,
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
labelled_inf_new = labelled_inf.drop(columns = ["RF_confidence", "RF_margin", "RF_entropy"])
labelled_inf_new = labelled_inf_new.copy()
labelled_inf_new["Subtype_new"] = labelled_inf_new["Subtype"]

labelled_inf_new.loc[labelled_inf_new['Metadata_CellID'].isin(new_cells), 'Subtype_new'] = 5

subpopulation_cells = {}
for i in list(range(1, len(subtype_cells["control"]) + new_cell_clusters)):  # start subpopulation count at 1
    subpopulation_cells[i] = labelled_inf_new.loc[
        labelled_inf_new["Subtype_new"] == i, "Metadata_CellID"
    ].tolist()

subtype_cells_new["inflammation"] = subpopulation_cells
subtype_cells_new["control"][5] = []

#%% compare compositions -individually

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


#%% feature heatmap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

plot_vertical = False
axes_show = False

df_combined = pd.concat(subtype_df, names=["Treatment"]).reset_index(level=0)

comparison_col = "Subtype"
condition_col = "Treatment"
group_a = "control"
group_b = "inflammation"

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

#%% differences in organelle type composition with inflammation (some changes in organelle type composition in certain populations)
pop = 4
pop_df = cell_labelled.copy()
plot_cols = "lipiddroplets"

plot_col_list = [col for col in pop_df["control"].columns if plot_cols in col and "Network" not in col]

pop_df["inflammation"] = pop_df["inflammation"][["Metadata_CellID"] + plot_col_list + ["Subtype"]]
df_combined_full = pd.concat(pop_df, names=["Treatment"]).reset_index(level=0)

cmap = {
        "control": (0.9934640522875817, 0.7477124183006535, 0.4352941176470587),
        "inflammation": (0.7477124183006538, 0.8980392156862746, 0.6274509803921569)
        }

# params 
comparison_col = "Type" # x axis comparison col
condition_col = "Treatment" # comparison within comparison col
measure = "CLR"

type_cols = plot_col_list

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
plt.figure(figsize=(10,5))

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
ax.tick_params(axis = "x", rotation = 90)
#ax.set_title(f"{measure}: Time {time0_label} vs {time4_label} per Line; within-line and between-line comparisons")
plt.tight_layout()
plt.show()

#%% differences in features per subpopulation (consider each subpopulation independent cells)

# read in original measures
for treatment in ["control", "inflammation"]:
    
    datadir = Path(constants.PARENTDIR / f"{treatment}" / f"{version}")
    filedir = datadir / "cellprofiler" / "measurements_updated"
    all_sheets_original[treatment] = combine_files_to_sheets(filedir)

    all_sheets[treatment] = all_sheets_original[treatment].copy()
    all_sheets_filtered[treatment] = filter_dataframes(all_sheets[treatment], keep_cols_reduced)
        

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

organelle = "mitochondria"

organelle_df = {}
context_df_labelled = {}
props_renamed["control"] = rename_df_columns_with_keyword(all_sheets_filtered["control"][organelle], rename_dict)
props_renamed["inflammation"] = rename_df_columns_with_keyword(all_sheets_filtered["inflammation"][organelle], rename_dict)

organelle_df["control"] = props_renamed["control"].copy()
organelle_df["inflammation"] = props_renamed["inflammation"].copy()

all_features = list(organelle_df["inflammation"].columns)

# label original dataframe with subtype
organelle_df["control"] = organelle_df["control"].merge(cell_labelled["control"][['Metadata_CellID', 'Subtype']], on='Metadata_CellID', how='left')
organelle_df["inflammation"] = organelle_df["inflammation"].merge(cell_labelled["inflammation"][['Metadata_CellID', 'Subtype']], on='Metadata_CellID', how='left')

context_df_labelled = context_full.copy()
context_df_labelled["control"] = context_df_labelled["control"].merge(cell_labelled["control"][['Metadata_CellID', 'Subtype']], on='Metadata_CellID', how='left')
context_df_labelled["inflammation"] = context_df_labelled["inflammation"].merge(cell_labelled["inflammation"][['Metadata_CellID', 'Subtype']], on='Metadata_CellID', how='left')

df_combined = pd.concat(organelle_df, names=["Treatment"]).reset_index(level=0)
df_combined = pd.concat(context_df_labelled, names=["Treatment"]).reset_index(level=0)

df_combined["Geometric_Area"] = df_combined["Geometric_Area"] * (0.01 ** 2)

dfs_prefixed = [
    df.rename(columns=lambda c: f"{k}_{c}" if c != "Metadata_CellID" else c)
    for k, df in context_df_all["control"].items()
]

context_df = {}
context_df["control"] = reduce(
    lambda left, right: pd.merge(left, right, on="Metadata_CellID", how="outer"),
    dfs_prefixed
)

df_combined_context = pd.concat(context_df, names=["Treatment"]).reset_index(level=0)

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

#%% feature heatmap (all features)
_, _, cmap_labelling = label_color_mapping_dict(
    df_combined["Subtype"], palette="crest"
)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

plot_vertical = True
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
    figsize = (7,13)
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
    #cbar_pos=cbar_pos,
    cbar_pos=None,
    dendrogram_ratio = (0.02, 0.02)
)

# Remove x-axis labels completely
def clean_feature_name(name):
    #name = name.split("_",1)[-1]
    #name = " ".join([name.split("_")[0], name.split("_")[-1]]).capitalize()
    parts = name.split("_")
    name = f"{parts[0][0].upper() + parts[0][1:]} {parts[-1]}"
    # replace keywords
    name = name.replace("Densitometric", "Intensity")

    return name

new_labels = [clean_feature_name(feature) for feature in features]

if plot_vertical:
    # effect_mat_plot = effect_mat.T
    x_labels = list(g.data2d.columns)   # subtypes -> 4 labels
    y_labels = [clean_feature_name(f) for f in g.data2d.index]  # features -> many labels
    g.ax_heatmap.set_yticklabels(y_labels, rotation=90, fontsize = 14)
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

#%% plot scale bar separately
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(0.25, 5))

norm = mpl.colors.Normalize(vmin=-0.3, vmax=0.3)
cmap = mpl.cm.get_cmap("vlag")

cb = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='vertical'
)

plt.show()
#%% example cell per type

max_representative = 10
pop = 4
organelle = "cell"

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
    cropped_xray = False, 
    lw = 10,
    scale_bar = True
)
