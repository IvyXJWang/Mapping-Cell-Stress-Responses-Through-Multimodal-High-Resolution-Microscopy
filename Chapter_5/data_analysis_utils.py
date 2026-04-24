import pandas as pd

pd.DataFrame.iteritems = pd.DataFrame.items  # iteritems removed in pandas version 2.0
import numpy as np

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as skprep
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import levene, kruskal, false_discovery_control
import matplotlib.cm as cm

import seaborn as sns  # version 0.11.2 requ res matplotlib version 3.7.3)
import matplotlib.pyplot as plt  # version 3.7.3
from PIL import Image
from scikit_posthocs import posthoc_dunn
import cv2
import os
import constants


def ANOVA_Ftest(dataframe, y):
    """
    ANOVA F-test for feature relevance - check if there are lots of redundant features (features with greatest contribution not found with MRMR suggests large contribution minimum relevance)

    """

    X_scaled = StandardScaler().fit_transform(dataframe)

    F, pvals = f_classif(X_scaled, y)

    feature_scores_Fstat = pd.DataFrame(
        {"Feature": dataframe.columns, "F_stat": F, "p_value": pvals}
    ).sort_values(by="F_stat", ascending=False)

    feature_list_Fstat = feature_scores_Fstat["Feature"]

    return feature_list_Fstat


def filter_cols(dataframe, col_list):
    # Keep only the features specified in keep_cols
    existing_features = [col for col in col_list if col in dataframe.columns]
    # df_clean = df[df.columns.intersection(keep_cols)]

    # Extract the matching columns
    filtered_df = dataframe[existing_features]

    return filtered_df


def summary_stat_all(df, group_col="Metadata_CellID", std_threshold=0.0):
    numeric_cols = df.select_dtypes(include="number").columns.difference(
        ["ImageNumber"], sort=False
    )
    grouped = df.groupby(group_col)[numeric_cols]

    agg = grouped.agg(["mean", "median", "std"])
    skew_df = grouped.skew()  # DataFrame indexed by group, columns = numeric_cols
    std_df = agg.xs("std", axis=1, level=1)

    skew_df = skew_df.where(std_df >= std_threshold, 0.0)
    agg.columns = ["_".join(col).strip() for col in agg.columns.to_flat_index()]
    skew_df = skew_df.rename(columns=lambda c: f"{c}_skew")

    agg_reset = agg.reset_index()
    skew_reset = skew_df.reset_index()

    stats = pd.merge(agg_reset, skew_reset, on=group_col, how="left")
    stats = stats.copy()

    return stats

def summary_stat(df, group_col="Metadata_CellID", std_threshold=0.0, DEBUG = False):
    
    numeric_cols = df.select_dtypes(include="number").columns.difference(
    ["ImageNumber"], sort=False
    )
    grouped = df.groupby(group_col)[numeric_cols]

    # Always compute mean
    mean_df = grouped.mean()

    features = df.select_dtypes(include=[np.number]).columns.difference([group_col])
    nunique = df.groupby(group_col)[features].nunique(dropna=False)  # change dropna if you want NaN treated differently
    constant_every_group = nunique.eq(1).all(axis=0)  # boolean Series indexed by feature

    # Compute other stats
    median_all = df.groupby(group_col)[features].median()
    std_all = df.groupby(group_col)[features].std()
    skew_all = df.groupby(group_col)[features].skew()
    
    median_df = median_all.loc[:, ~constant_every_group]#.reset_index()
    std_df = std_all.loc[:, ~constant_every_group]#.reset_index()
    skew_df = skew_all.loc[:, ~constant_every_group]#.reset_index()
    
    if DEBUG:
        return std_df
    
    # Rename columns
    mean_df = mean_df.rename(columns=lambda c: f"{c}_mean")
    median_df = median_df.rename(columns=lambda c: f"{c}_median")
    std_df = std_df.rename(columns=lambda c: f"{c}_std")
    skew_df = skew_df.rename(columns=lambda c: f"{c}_skew")

    # Combine
    stats = pd.concat([mean_df, median_df, std_df, skew_df], axis=1).reset_index()
    #only calculate mean
    #stats = mean_df.reset_index()

    return stats


def calculate_summary_stats(dataframe_dict):
    summary = {}

    for sheet_name, df in dataframe_dict.items():
        print(f"Processing ........... {sheet_name}")

        # Check for multiple objects per cellID for summary statistics
        all_cellID_unique = df["Metadata_CellID"].is_unique

        if all_cellID_unique:
            summary[sheet_name] = df.copy()
            continue  # Only one measurements per cell skipping summary stat calculations

        # calculate summary statistics per CellID
        stats = summary_stat(df, group_col="Metadata_CellID", std_threshold=0)

        summary[sheet_name] = stats

    return summary


def extract_columns(dataframe, keywords, method="or"):
    """
    Extract columns from dataframe containing keyword(s)
    method: and (contains all keywords), or (contains one of the keywords)
    """

    cols = dataframe.columns

    if method == "or":
        cols = [
            col
            for col in dataframe.columns
            if any(keyword in col for keyword in keywords)
        ]

    elif method == "and":
        cols = [
            col
            for col in dataframe.columns
            if all(keyword in col for keyword in keywords)
        ]

    filtered_dataframe = dataframe[cols]

    return filtered_dataframe


def combine_dataframes_by_CellID(dataframe_dict):
    # Rename columns to include sheet (key) name (to avoid duplicates)
    renamed_sheets = []
    merged_df = None

    for sheet_name, df in dataframe_dict.items():
        df_renamed = (
            df.set_index(["Metadata_CellID"]).add_prefix(f"{sheet_name}_").reset_index()
        )
        renamed_sheets.append(df_renamed)

        # Merge sheets horizontally (column-wise based on cellID)
        if merged_df is None:
            merged_df = df_renamed
        else:
            merged_df = pd.merge(
                merged_df, df_renamed, on="Metadata_CellID", how="outer"
            )  # or 'inner'

    return merged_df


def drop_empty_cols(dataframe, threshold=0.5):
    """
    threshold (int): percentage of column that can be empty

    """

    is_empty = dataframe.isna()  # | (dataframe == 0) might be a relevant
    empty_fraction = is_empty.mean()
    df_clean = dataframe.loc[:, empty_fraction <= threshold]
    df_empty = dataframe.loc[:, empty_fraction > threshold].columns.tolist()

    return df_clean, df_empty


def drop_no_variation_cols(dataframe, scale=False, threshold=0):
    df_numeric = dataframe.select_dtypes(include="number")
    df_cellID = dataframe["Metadata_CellID"]
    # df_clean = df_numeric.loc[:, df_numeric.std() > threshold]
    # df_empty = df_numeric.loc[:, df_numeric.std() <= threshold]

    # scaling change standard deviation
    if scale:
        data_scaler = skprep.StandardScaler()
        df_scaled = pd.DataFrame(
            data_scaler.fit_transform(df_numeric),
            columns=df_numeric.columns,
            index=df_numeric.index,
        )
    else:
        df_scaled = df_numeric

    keep_cols = df_scaled.std() > threshold # variation across feature/col 

    df_clean = df_numeric.loc[:, keep_cols].assign(Metadata_CellID=df_cellID)
    df_empty = df_numeric.loc[:, ~keep_cols].columns.tolist()

    return df_clean, df_empty


def keyword_occurrence_list(string_list, keywords, method="all"):
    """
    count number of occurrences of keywords in list of strings
    """

    count = 0
    for string in string_list:
        if method == "all":
            found = all(keyword in string for keyword in keywords)
        elif method == "any":
            found = any(keyword in string for keyword in keywords)
        if found:
            count += 1

    return count


def copy_list_keyboard(variable):
    df = pd.DataFrame(variable)
    df.to_clipboard(index=False, header=False)
    print("variable copied to clipboard!")


def extract_df_subset_sequential(dataset_full, subset=[], constant_key=""):
    dataset_filtered = dataset_full.copy()
    subset_copy = subset

    for subset_list in subset_copy:
        subset_list.append(constant_key)
        dataset_filtered = extract_columns(dataset_filtered, keywords=subset_list)
        subset_list.remove(constant_key)

    return dataset_filtered


def remove_outliers(dataframe):
    df = dataframe.copy()
    outlier_cols = []

    for col in list(df.columns):
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue  # skip non-numeric columns

        Q1 = df[str(col)].quantile(0.05)
        Q3 = df[str(col)].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Check if column has any values outside the bounds
        if ((df[col] < lower_bound) | (df[col] > upper_bound)).any():
            outlier_cols.append(col)

        df = df[(df[str(col)] >= lower_bound) & (df[str(col)] <= upper_bound)]

    return df, outlier_cols


def delete_outlier_rows(dataset, dataset_full, data_no_outliers, remove=True):
    """
    no longer needed - remove outliers removes rows and maintains cellID
    """

    if remove == True:
        removed_cell_list = []
        if len(dataset) > len(data_no_outliers):
            removed_cells = pd.merge(
                dataset, data_no_outliers, how="outer", indicator=True
            )["_merge"]
            removed_cells_idx = removed_cells[removed_cells == "left_only"].index[:]

            full_dataset_no_outliers = dataset_full[
                ~dataset_full.index.isin(list(removed_cells_idx))
            ]
            removed_cell_list = removed_cells_idx.astype(int)

        else:
            print("no outliers found :D")
            full_dataset_no_outliers = dataset_full

    elif remove == False:
        removed_cell_list = []
        full_dataset_no_outliers = dataset_full

    return full_dataset_no_outliers, removed_cell_list


def scale_data(dataframe):
    """
    dataframe: takes any pandas dataframe but only scales numeric columns
    scales numeric columns in dataframe and adds non-numeric columns back into dataframe
    """
    # split dataset into numeric and non-numeric
    dataframe_numeric = dataframe.select_dtypes(include=np.number)
    dataframe_non_numeric = dataframe.select_dtypes(exclude=["number"])

    data_scaler = skprep.StandardScaler()
    scaled = data_scaler.fit_transform(dataframe_numeric)

    # convert NaN values to 0
    scaled_data = np.nan_to_num(scaled, nan=0.0)

    # Convert back to DataFrame
    scaled_df = pd.DataFrame(
        scaled_data, columns=dataframe_numeric.columns, index=dataframe.index
    )

    # Combine scaled numeric columns and untouched non-numeric columns
    final_df = pd.concat([scaled_df, dataframe_non_numeric], axis=1)

    return final_df


def weighted_df(dataframe, weight_dict):
    # feature weighting: multiply normalized data 0-1
    dataframe_weighted = dataframe.copy()
    for feat, weight in weight_dict.items():
        dataframe_weighted[feat] = dataframe_weighted[feat] * weight

    return dataframe_weighted


def save_svg(plt, outputdir, filename):
    outputdir.mkdir(parents=True, exist_ok=True)

    image_format = "svg"
    image_name = outputdir / filename
    plt.savefig(image_name, format=image_format, dpi=1200, bbox_inches="tight")

    print(f"Saved plot to {outputdir}")


def compare_dataframes(dataframe1, dataframe2, return_variables="dataframes"):
    common_cols = dataframe1.columns.intersection(dataframe2.columns).tolist()
    dataframe1_only = dataframe1.columns.difference(dataframe2.columns)
    dataframe2_only = dataframe2.columns.difference(dataframe1.columns)
    not_in_common = dataframe1_only.union(dataframe2_only)

    dataframe1_filtered = dataframe1[common_cols]
    dataframe2_filtered = dataframe2[common_cols]

    if return_variables == "dataframes":
        return dataframe1_filtered, dataframe2_filtered
    else:
        return dataframe1_filtered, dataframe2_filtered, common_cols, not_in_common


def save_tiff(img, outputdir, filename):
    im = Image.fromarray(img.astype(np.uint8))
    full_filename = filename + ".tiff"
    im.save(outputdir / full_filename, "TIFF")

    return


def dataset_metrics(dataframe):
    numeric_dataset = dataframe.select_dtypes(include=np.number)

    stdev_df = np.std(numeric_dataset, axis=0)
    mean_df = np.mean(numeric_dataset, axis=0)

    return mean_df, stdev_df


def compare_two_df(dataframe1, dataframe2, stat_test="levene"):
    # Example: per-feature Levene's test (robust to non-normality)
    # Optional: Adjust for multiple testing (e.g., Benjamini-Hochberg)

    dataframe1, dataframe2, _, dropped_cols = compare_dataframes(
        dataframe1, dataframe2, return_variables="all"
    )

    numeric_dataset1 = dataframe1.select_dtypes(include=np.number)
    numeric_dataset2 = dataframe2.select_dtypes(include=np.number)

    p_vals = {}
    significant_features = []

    for col in numeric_dataset1.columns:
        if stat_test == "levene":
            stat, p = levene(numeric_dataset1[col], numeric_dataset2[col])
            p_vals[col] = p
            if p < 0.05:
                significant_features.append(col)

    return significant_features, p_vals, dropped_cols


def variance_analysis(control_df, treatment_df, plot=True, verbose=False):
    _, p_val_dict = compare_two_df(control_df, treatment_df)
    results = pd.DataFrame(p_val_dict)

    numeric_control_df = control_df.select_dtypes(include=np.number)
    numeric_treatment_df = treatment_df.select_dtypes(include=np.number)

    # Calculate variance for each feature in both groups
    var_control = numeric_control_df.var()
    var_exp = numeric_treatment_df.var()

    # Compute the difference (treatment - control)
    delta_var = var_exp - var_control

    # Show only features with significant p-values (assuming you've stored them)
    significant_features = results[results["P_Value"] < 0.05]["Feature"]

    # Display direction of variance change
    if verbose:
        for feature in significant_features:
            print(
                f"{feature}: ΔVar = {delta_var[feature]:.4f} "
                f"({'↑ in Treatment' if delta_var[feature] > 0 else '↓ in Treatment'})"
            )

    results["Var_Control"] = var_control.values
    results["Var_Treatment"] = var_exp.values
    results["Delta_Var"] = results["Var_Treatment"] - results["Var_Control"]
    results["Direction"] = results["Delta_Var"].apply(
        lambda x: "↑ in Treatment" if x > 0 else "↓ in Treatment"
    )

    if plot:
        # Only include significant features
        sig_results = results[results["P_Value"] < 0.05]

        plt.figure(figsize=(12, 6))
        sns.barplot(data=sig_results, x="Feature", y="Delta_Var", palette="coolwarm")

        plt.axhline(0, color="gray", linestyle="--")
        plt.xticks(rotation=90)
        plt.ylabel("Variance Difference (Treatment - Control)")
        plt.title("Direction and Magnitude of Feature Variance Differences")
        plt.tight_layout()
        plt.show()

    return


def plot_silscore_clusters_v1(
    scaled_data, clusters, n_clusters, datashape="nD", plotdataframe=None
):
    silhouette_indiv = silhouette_samples(scaled_data, clusters)
    silhouette_avg = silhouette_score(scaled_data, clusters)

    y_lower = 10
    # scatter_y = np.ones(len(data)).reshape(-1,1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_indiv[clusters == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot for the various clusters.")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    if datashape == "2D":
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(clusters.astype(float) / n_clusters)
        ax2.scatter(
            plotdataframe.iloc[:, 0],
            plotdataframe.iloc[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        ax2.set_title("Visualization of the clustered data.")
        ax2.set_xlabel("Principal component 1")
        ax2.set_ylabel("Principal component 2")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

    return


def kw_single(dataframe, feature_name, cluster_col=None):
    if cluster_col is None:
        cluster_col = dataframe.columns[-1]

    data = []
    populations = np.unique(dataframe[cluster_col])

    for p in populations:
        data.append(dataframe[dataframe[cluster_col] == p][feature_name])

    stat, pval = kruskal(*data)

    return stat, pval


def kw_dunn(dataframe, cluster_col=None, alpha=0.05, p_adjust="fdr_bh"):
    if cluster_col is None:
        cluster_col = dataframe.columns[-1]

    # get list of features
    features = dataframe.columns.drop(cluster_col)
    numeric_feats = (
        dataframe[features].select_dtypes(include=[np.number]).columns.tolist()
    )

    significant_features = []
    molten_dunn_df = {}
    dunn_df = {}
    for feature in numeric_feats:
        stat, pval = kw_single(dataframe, feature)

        if pval < alpha:
            significant_features.append(feature)
            dunn_df[feature] = posthoc_dunn(
                dataframe, val_col=feature, group_col=cluster_col, p_adjust=p_adjust
            )
            remove = np.tril(np.ones(dunn_df[feature].shape), k=0).astype("bool")
            dunn_df[feature][remove] = np.nan

            molten_dunn_df[feature] = (
                dunn_df[feature].melt(ignore_index=False).reset_index().dropna()
            )

    return significant_features, molten_dunn_df, dunn_df


def kw_dunn_group_diffs(dataframe, cluster_col=None, alpha=0.05, p_adjust="fdr_bh"):
    """
    Same as before but fixes label-mismatch lookup bug when reading pairwise p-values
    from the Dunn p-value DataFrame.
    Returns:
        significant_features, molten_dunn_df, dunn_df, group_feature_table, group_summary
    """
    if cluster_col is None:
        cluster_col = dataframe.columns[-1]

    features = dataframe.columns.drop(cluster_col)
    numeric_feats = dataframe[features].select_dtypes(include=[np.number]).columns.tolist()

    significant_features = []
    molten_dunn_df = {}
    dunn_df = {}
    rows = []

    for feature in numeric_feats:
        stat, pval = kw_single(dataframe, feature)

        if pval < alpha:
            significant_features.append(feature)
            pmat = posthoc_dunn(
                dataframe, val_col=feature, group_col=cluster_col, p_adjust=p_adjust
            )

            # keep full square p-value matrix
            dunn_df[feature] = pmat.copy()

            # create molten (masked lower triangle as before)
            remove = np.tril(np.ones(pmat.shape), k=0).astype(bool)
            masked = pmat.copy()
            masked[remove] = np.nan
            molten_dunn_df[feature] = masked.melt(ignore_index=False).reset_index().dropna()

            # ---------- robust per-group classification ----------
            # Use the original labels (do NOT coerce to str)
            groups = list(pmat.index)

            def get_p(a, b):
                """Return p-value for pair (a,b) trying both orders; return NaN if not found."""
                # direct try
                try:
                    return pmat.loc[a, b]
                except Exception:
                    pass
                # try reversed
                try:
                    return pmat.loc[b, a]
                except Exception:
                    pass
                # fallback: if labels are present but in columns/rows swapped
                if (a in pmat.index) and (b in pmat.columns):
                    try:
                        return pmat.loc[a, b]
                    except Exception:
                        pass
                if (b in pmat.index) and (a in pmat.columns):
                    try:
                        return pmat.loc[b, a]
                    except Exception:
                        pass
                return np.nan

            for g in groups:
                partners = []
                for other in groups:
                    if other == g:
                        continue
                    pv = get_p(g, other)
                    # if pv is array-like (shouldn't be), coerce to scalar
                    if isinstance(pv, (pd.Series, np.ndarray)):
                        try:
                            pv = float(pv.squeeze())
                        except Exception:
                            pv = np.nan
                    if (not pd.isna(pv)) and (pv < alpha):
                        partners.append(other)
                n_sig = len(partners)
                if n_sig >= 2:
                    category = "differs_from_both"
                elif n_sig == 1:
                    category = "differs_from_exactly_one"
                else:
                    category = "differs_from_none"
                rows.append((feature, g, n_sig, partners, category))

    # Build DataFrame
    if rows:
        group_feature_table = pd.DataFrame.from_records(
            rows, columns=["feature", "group", "n_significant", "partners", "category"]
        ).set_index(["feature", "group"])
    else:
        group_feature_table = pd.DataFrame(
            columns=["n_significant", "partners", "category"]
        ).astype(object)

    # Build summary dict
    group_summary = {}
    for (feat, g), row in group_feature_table.iterrows():
        group_summary.setdefault(g, {"differs_from_both": [], "differs_from_exactly_one": [], "differs_from_none": []})
        group_summary[g][row["category"]].append(feat)

    return significant_features, molten_dunn_df, dunn_df, group_feature_table, group_summary

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


def kw_dunn_effects(dataframe, cluster_col=None, alpha=0.05, p_adjust="fdr_bh"):
    if cluster_col is None:
        cluster_col = dataframe.columns[-1]

    features = dataframe.columns.drop(cluster_col)
    numeric_feats = dataframe[features].select_dtypes(include=[np.number]).columns.tolist()

    significant_features = []
    pairwise_rows = []
    dunn_df = {}

    for feature in numeric_feats:
        stat, pval = kw_single(dataframe, feature)

        if pval < alpha:
            significant_features.append(feature)

            dunn = posthoc_dunn(
                dataframe,
                val_col=feature,
                group_col=cluster_col,
                p_adjust=p_adjust
            )
            dunn_df[feature] = dunn

            groups = list(dunn.index)

            for i, g1 in enumerate(groups):
                x = dataframe.loc[dataframe[cluster_col] == g1, feature].dropna()

                for g2 in groups[i + 1:]:
                    y = dataframe.loc[dataframe[cluster_col] == g2, feature].dropna()

                    pairwise_rows.append({
                        "feature": feature,
                        "group1": g1,
                        "group2": g2,
                        "pair": f"{g1} vs {g2}",
                        "p_adj": dunn.loc[g1, g2],
                        "cliffs_delta": cliffs_delta(x, y),
                    })

    pairwise_df = pd.DataFrame(pairwise_rows)

    return significant_features, pairwise_df, dunn_df

def find_all_none(dataframe, cluster_col=None, alpha=0.05, p_adjust="fdr_bh"):
    """
    Use your kw_dunn_group_diffs to classify features:
      - 'all'     : every pairwise comparison significant
      - 'none'    : Kruskal-Wallis not significant (no pairwise tests run)
      - 'partial' : some but not all pairwise comparisons significant

    Returns:
        all_features       : list of features where all pairs differ
        none_features      : list of features with no global difference
        summary_df         : DataFrame with columns ['feature', 'category']
    """
    # run your function once to get all outputs
    sig_feats, molten, dunn_df_dict, group_feature_table, group_summary = kw_dunn_group_diffs(
        dataframe, cluster_col=cluster_col, alpha=alpha, p_adjust=p_adjust
    )

    # determine numeric features (same logic as your function)
    if cluster_col is None:
        cluster_col = dataframe.columns[-1]
    features = dataframe.columns.drop(cluster_col)
    numeric_feats = dataframe[features].select_dtypes(include=[np.number]).columns.tolist()

    all_features = []
    none_features = []
    partial_features = []

    for feat in numeric_feats:
        # if KW was not significant, kw_dunn_group_diffs did not compute Dunn -> classify as 'none'
        if feat not in sig_feats:
            none_features.append(feat)
            continue

        # Dunn matrix for feature (square DataFrame indexed/cols by group)
        pmat = dunn_df_dict.get(feat)
        if pmat is None:
            # Unexpected — treat as partial to be conservative
            partial_features.append(feat)
            continue

        # Mask lower triangle and diagonal (we only want unique unordered pairs)
        mask = np.tril(np.ones(pmat.shape), k=0).astype(bool)
        upper = pmat.mask(mask)

        # Flatten remaining p-values to a 1D array of scalars (drop NaNs)
        flat = upper.values.flatten()
        flat = flat[~pd.isna(flat)]

        # If no pairwise values found (e.g., only one group), classify as 'none' (no comparisons)
        if flat.size == 0:
            none_features.append(feat)
            continue

        # all pairs significant if every remaining pval < alpha
        if np.all(flat < alpha):
            all_features.append(feat)
        else:
            partial_features.append(feat)

    # build summary DataFrame
    rows = []
    for f in all_features:
        rows.append((f, "all"))
    for f in none_features:
        rows.append((f, "none"))
    for f in partial_features:
        rows.append((f, "partial"))

    summary_df = pd.DataFrame(rows, columns=["feature", "category"]).set_index("feature")

    return all_features, none_features, summary_df


def significant_kruskal_columns_v1(
    df,
    cluster_col=None,
    alpha=0.05,
    min_group_size=3,
    p_adjust="bh",
    return_summary=False,
):
    """
    Run Kruskal-Wallis for each column and return a list of columns with significant
    differences between the groups defined in `cluster_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    cluster_col : str or int, optional
        Column name or index that contains group labels. If None, uses last column.
    alpha : float, default 0.05
        Significance threshold (applied to adjusted p-values if p_adjust is not None).
    numeric_only : bool, default True
        If True, test only numeric columns; otherwise try to coerce.
    min_group_size : int, default 3
        Minimum number of non-NaN observations per group to include that group in testing.
        A feature is skipped if fewer than two groups have >= min_group_size.
    p_adjust : {'bh','bonferroni', 'by', 'holm', 'sidak', None}, default 'bh'
        Method to adjust the collection of p-values across features.
    return_summary : bool, default False
        If True, also return a pandas.DataFrame with feature, raw p, p_adj, n_groups_used.

    Returns
    -------
    significant_cols : list
        List of column names (strings) that are significant at the given alpha.
    (summary_df) : pandas.DataFrame, optional
        If return_summary True, also returns the DataFrame summarizing results.
    ************* Same results as v2 *************
    """
    if cluster_col is None:
        cluster_col = df.columns[-1]
    labels = df[cluster_col]
    features = df.columns.drop(cluster_col)

    # choose features
    numeric_feats = df[features].select_dtypes(include=[np.number]).columns.tolist()

    results = []
    raw_pvals = []

    for feat in numeric_feats:
        col = df[feat]
        # collect arrays for groups that have enough data
        group_arrays = []
        groups_used = []
        for g in pd.Index(labels.dropna().unique()):
            vals = col[labels == g].dropna().values
            if vals.size >= min_group_size:
                group_arrays.append(vals)
                groups_used.append(g)
        if len(group_arrays) < 2:
            # not enough groups with data to run test
            results.append(
                {"feature": feat, "p": np.nan, "n_groups": len(group_arrays)}
            )
            raw_pvals.append(np.nan)
            continue

        # remove any NaNs inside arrays (just in case) and run kruskal
        try:
            clean_arrays = [a[~np.isnan(a)] for a in group_arrays]
            if len(clean_arrays) < 2 or any(a.size == 0 for a in clean_arrays):
                pval = np.nan
            else:
                stat, pval = kruskal(*clean_arrays)
        except Exception:
            # safe fallback
            pval = np.nan

        results.append({"feature": feat, "p": pval, "n_groups": len(group_arrays)})
        raw_pvals.append(pval)

    # adjust p-values, preserving order
    if p_adjust is not None:
        pvals_arr = np.array(raw_pvals, dtype=float)
        valid_mask = ~np.isnan(pvals_arr)
        adj = np.full_like(pvals_arr, np.nan, dtype=float)
        if valid_mask.any():
            adj_vals = false_discovery_control(pvals_arr[valid_mask], method=p_adjust)
            adj[valid_mask] = adj_vals

        # build summary dataframe
        summary = pd.DataFrame(results)
        summary["p_adj"] = adj
    else:
        summary = pd.DataFrame(results)

    # determine significance
    if p_adjust is not None:
        summary["significant"] = summary["p_adj"].notna() & (summary["p_adj"] < alpha)
    else:
        summary["significant"] = summary["p"].notna() & (summary["p"] < alpha)

    sig_cols = summary.loc[summary["significant"], "feature"].tolist()

    if return_summary and p_adjust is not None:
        return sig_cols, summary.sort_values("p_adj").reset_index(drop=True)
    elif return_summary and p_adjust is None:
        return sig_cols, summary.sort_values("p").reset_index(drop=True)

    return sig_cols


def analyze_dunn_results(molten_dunn_dict, alpha=0.05):
    """
    Analyze the molten Dunn pairwise p-values to classify how populations differ for each feature.

    Parameters
    ----------
    molten_dunn_dict : dict
        Output from kw_dunn(); keys are features and values are a melted DataFrame
        containing pairwise comparisons. Expected columns after your melt:
        ['index', 'variable', 'value'] where:
          - 'index' is group A
          - 'variable' is group B
          - 'value' is the p-value for A vs B
    alpha : float
        significance threshold (default 0.05)

    Returns
    -------
    summary_df : pd.DataFrame
        One row per feature with columns:
          - feature: feature name
          - n_groups: number of unique populations found
          - classification: one of 'all_differ', 'two_differ_from_all',
                            'one_differ_from_all', 'only_pairwise', 'mixed'
          - significant_pairs: list of tuples (A,B) that are significant
          - counts: dict mapping group->number of significant differences vs others
    details : dict
        Mapping feature -> dict with:
            - 'signif_matrix' : pd.DataFrame (boolean adjacency matrix)
            - 'pairs' : list of significant (A,B)
            - 'counts' : dict group->count
    """
    summary_rows = []
    details = {}

    for feature, df in molten_dunn_dict.items():
        # normalize column names if needed
        cols = [c.lower() for c in df.columns]
        # try to infer expected columns
        if {"index", "variable", "value"}.issubset(set(df.columns)):
            left_col, right_col, val_col = "index", "variable", "value"
        elif {"level_0", "variable", "value"}.issubset(set(df.columns)):
            left_col, right_col, val_col = "level_0", "variable", "value"
        else:
            # try generic fallback: first three columns
            left_col, right_col, val_col = df.columns[:3]

        # ensure strings
        df_local = df[[left_col, right_col, val_col]].copy()
        df_local[left_col] = df_local[left_col].astype(str)
        df_local[right_col] = df_local[right_col].astype(str)

        # get full list of groups
        groups = sorted(set(df_local[left_col]).union(set(df_local[right_col])))

        # build an empty boolean adjacency matrix (rows/cols = groups)
        mat = pd.DataFrame(False, index=groups, columns=groups)

        significant_pairs = []
        # fill upper-triangle from melted df (the melt kept only one triangle)
        for _, row in df_local.iterrows():
            a = row[left_col]
            b = row[right_col]
            p = row[val_col]
            try:
                pval = float(p)
            except Exception:
                # skip non-numeric (defensive)
                continue
            if pval < alpha:
                mat.at[a, b] = True
                mat.at[b, a] = True  # symmetric
                significant_pairs.append((a, b))

        # count how many other groups each group differs from
        counts = {g: int(mat.loc[g].sum()) for g in groups}

        k = len(groups)
        counts_list = list(counts.values())

        # classification logic
        if all(c == (k - 1) for c in counts_list):
            classification = "all_differ"
        else:
            n_full = sum(1 for c in counts_list if c == (k - 1))
            n_one = sum(1 for c in counts_list if c == 1)
            n_more_than_one = sum(1 for c in counts_list if c > 1 and c < (k - 1))

            # specifically tuned for 3-group case (common)
            if k == 3:
                # possible counts per group are 0,1,2
                if n_full == 2:
                    classification = (
                        "two_differ_from_all"  # two groups differ from both others
                    )
                elif n_full == 1:
                    classification = "one_differ_from_all"
                elif all(c <= 1 for c in counts_list):
                    # no group differs from both others; only pairwise differences (possibly disjoint)
                    classification = "only_pairwise"
                else:
                    classification = "mixed"
            else:
                # general multi-group heuristic
                if n_full >= 1 and n_more_than_one == 0 and n_one == (k - n_full):
                    # e.g., one or more groups differ from everyone, the rest differ from only them
                    if n_full == 1:
                        classification = "one_differ_from_all"
                    elif n_full == 2:
                        classification = "two_differ_from_all"
                    else:
                        classification = "mixed"
                elif all(c <= 1 for c in counts_list):
                    classification = "only_pairwise"
                else:
                    classification = "mixed"

        summary_rows.append(
            {
                "feature": feature,
                "n_groups": k,
                "classification": classification,
                "significant_pairs": significant_pairs,
                "counts": counts,
            }
        )

        details[feature] = {
            "signif_matrix": mat,
            "pairs": significant_pairs,
            "counts": counts,
        }

    summary_df = pd.DataFrame(summary_rows).set_index("feature")

    return summary_df, details


def map_imagenum_to_layer(series):
    # sorted unique imagenums for this group
    cats = sorted(series.unique())
    # create categorical with that category order and return integer codes (0-based)
    return pd.Categorical(series, categories=cats, ordered=True).codes


def update_cellID_objectnum(dataframe, imagenum_col, object_col, id_col):
    dataframe["Layer"] = dataframe.groupby(id_col)[imagenum_col].transform(
        map_imagenum_to_layer
    )
    dataframe_updated = dataframe.copy()
    dataframe_updated["ObjectID"] = (
        dataframe_updated[id_col]
        + "_"
        + dataframe_updated["Layer"].astype(str)
        + "_"
        + dataframe_updated[object_col].astype(str)
    )

    #dataframe_updated["Metadata_CellID"] = dataframe[id_col]

    return dataframe_updated


def show_object(segmentation_dict, organelle, object_id):
    id_split = object_id.split("_")

    if organelle == constants.MITOCHONDRIA:
        cellID = id_split[0]
        layer = int(id_split[1])
        obj = int(id_split[2])

        layer_img = segmentation_dict.loc[
            (segmentation_dict["Metadata_CellID"] == cellID)
            & (segmentation_dict["layer"] == layer),
            "image",
        ].iloc[0]
        object_mask = layer_img == obj

    elif organelle == "lipiddroplets":
        cellID = id_split[0]
        obj = int(id_split[2])

        object_mask = segmentation_dict[cellID] == obj

    else:
        cellID = id_split[0]
        object_mask = segmentation_dict[cellID]

    return object_mask, cellID


def overlay_mask_xray(
    obj_mask,
    xray,
    color=(0, 0, 255),
    alpha=0.8,
    lw=1,
    fill=False,
    savedir="",
    filename="overlay.png",
    show=True,
    display_window=(0, 100),
):
    """
    Overlay a colored outline (or filled mask) of obj_mask on a grayscale xray image.

    Parameters
    - obj_mask: (H,W) binary mask. Can be bool, 0/1, or 0/255.
    - xray: (H,W) grayscale uint8 image.
    - color: BGR tuple, e.g. (0,0,255) for red.
    - alpha: float in [0,1], blending weight of the colored overlay.
    - lw: integer outline thickness (uses dilation iterations or contour thickness).
    - fill: if True, fill the mask region with color; if False (default), draw only outline.
    - savedir: optional folder to save the result image; if empty, won't save.
    - filename: output filename when saved.
    - show: whether to call cv2.imshow (set False for headless/script use).
    Returns: result image (H,W,3) uint8 (BGR).
    """

    # ---- Validate & normalize mask ----
    if obj_mask.ndim == 3 and obj_mask.shape[2] == 1:
        obj_mask = obj_mask[..., 0]
    # convert to boolean mask
    mask_bool = obj_mask > 0

    h, w = mask_bool.shape
    if xray.ndim != 2:
        raise ValueError("xray must be a grayscale image with shape (H, W).")

    if xray.shape[0] != h or xray.shape[1] != w:
        raise ValueError("xray and obj_mask must have the same spatial dimensions.")

    # ---- Prepare display-scaled grayscale image ----
    win_min, win_max = display_window
    # convert xray to float for scaling
    x = xray.astype(np.float32)

    # Clip to window and map to 0-255
    x_clipped = np.clip(x, win_min, win_max)
    # Avoid division by zero if win_min == win_max
    denom = (win_max - win_min) if (win_max - win_min) != 0 else 1.0
    x_scaled = (x_clipped - win_min) / denom  # 0..1
    x_disp = (x_scaled * 255.0).clip(0, 255).astype(np.uint8)  # uint8 0..255

    # ---- Convert grayscale to BGR so we can colorize ----
    base_bgr = cv2.cvtColor(x_disp.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # ---- Create colored overlay image ----
    overlay = base_bgr.copy()
    color = tuple(int(c) for c in color)  # ensure ints

    if fill:
        # Fill entire mask region with color
        overlay[mask_bool] = color
    else:
        # Create outline: dilate(mask) - mask
        kernel = np.ones((3, 3), np.uint8)
        iterations = max(1, int(round(lw)))  # ensure positive int
        dilated = cv2.dilate(
            mask_bool.astype(np.uint8) * 255, kernel, iterations=iterations
        )
        outline = cv2.subtract(dilated, (mask_bool.astype(np.uint8) * 255))
        edges_bool = outline > 0

        # Apply color only on outline pixels
        overlay[edges_bool] = color

    # ---- Blend overlay with base image ----
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    result = cv2.addWeighted(overlay, alpha, base_bgr, 1.0 - alpha, 0)

    # ---- Optionally save ----
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        out_path = os.path.join(savedir, filename)
        cv2.imwrite(out_path, result)

    # ---- Optionally show ----
    if show:
        cv2.imshow("Overlay", result)
        # waitKey with small delay so GUI updates; change to 0 for blocking
        cv2.waitKey(1)

    return result


def delete_file(filedir):
    try:
        os.remove(filedir)
    except OSError:
        pass

    print(f"{filedir.stem} file cleared")

    return
