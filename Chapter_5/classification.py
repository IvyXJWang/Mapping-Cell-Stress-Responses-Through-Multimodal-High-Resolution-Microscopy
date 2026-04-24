# general modules
import pandas as pd
import numpy as np  # requires numpy 1.26.4

pd.DataFrame.iteritems = pd.DataFrame.items  # iteritems removed in pandas version 2.0

import joblib
from pathlib import Path

# calculation modules
import sklearn.cluster as skclus  # requires scikit-learn 1.5.1
import sklearn.metrics as skmet
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, cut_tree
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.preprocessing as skprep
from sklearn.ensemble import RandomForestClassifier
from feature_engine.selection import MRMR

# plotting modules
import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns

# project modules
import data_analysis_utils as utils
import plotting as plot
import constants
from scipy.stats import entropy

# %% Helper functions


def gap_statistic(X, max_k=10, B=10, random_state=None):
    from scipy.spatial.distance import cdist

    """
    Compute the Gap Statistic for an optimal number of clusters.

    Parameters:
        X : numpy array of shape (n_samples, n_features)
            The data to cluster.
        max_k : int
            Maximum number of clusters to test.
        B : int
            Number of reference datasets to generate.
        random_state : int or None
            Random state for reproducibility.

    Returns:
        gaps : list
            Gap values for each k from 1 to max_k.
        optimal_k : int
            Optimal number of clusters based on Gap Statistic.
    """
    np.random.seed(random_state)
    gaps = []
    deviations = []

    # Bounds for generating reference data within min/max of each feature
    shape_bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]

    # Loop over each number of clusters
    for k in range(1, max_k + 1):
        # Fit k-means on the actual data
        kmeans = skclus.KMeans(n_clusters=k, random_state=random_state).fit(X)
        Wk = np.log(
            sum(np.min(cdist(X, kmeans.cluster_centers_, "euclidean"), axis=1) ** 2)
            / X.shape[0]
        )

        # Generate B reference datasets and compute Wk for each
        Wk_refs = []
        for _ in range(B):
            random_data = np.array(
                [
                    np.random.uniform(low, high, X.shape[0])
                    for (low, high) in shape_bounds
                ]
            ).T
            kmeans_ref = skclus.KMeans(n_clusters=k, random_state=random_state).fit(
                random_data
            )
            Wk_refs.append(
                np.log(
                    sum(
                        np.min(
                            cdist(
                                random_data, kmeans_ref.cluster_centers_, "euclidean"
                            ),
                            axis=1,
                        )
                        ** 2
                    )
                    / random_data.shape[0]
                )
            )

        # Compute Gap(k) as the mean difference between Wk_refs and Wk
        gap_k = np.mean(Wk_refs) - Wk
        gaps.append(gap_k)

        # Compute the standard deviation of the gaps
        sk = np.sqrt(np.sum((Wk_refs - np.mean(Wk_refs)) ** 2) / B) * np.sqrt(1 + 1 / B)
        deviations.append(sk)

    # Determine optimal k
    optimal_k = next(
        k for k in range(1, max_k) if gaps[k - 1] >= gaps[k] - deviations[k]
    )

    return optimal_k


def elbow_method(X, max_k=10, random_state=None):
    """
    Implement the Elbow Method for determining optimal number of clusters.

    Parameters:
        X : numpy array of shape (n_samples, n_features)
            The data to cluster.
        max_k : int
            Maximum number of clusters to test.
        random_state : int or None
            Random state for reproducibility.

    Returns:
        sse : list
            Sum of squared errors for each k from 1 to max_k.
    """
    sse = []

    # Loop over each number of clusters
    for k in range(1, max_k + 1):
        kmeans = skclus.KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        sse.append(
            kmeans.inertia_
        )  # Sum of squared distances to the nearest cluster center

    # Plot the SSE for each k
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), sse, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Sum of squared errors (SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.xticks(range(1, max_k + 1))
    plt.show()

    return sse


def silhouette_score_indiv(dataframe, labels, scale=True, cellid_col=None):
    if scale:
        scaled_dataset = utils.scale_data(dataframe)
    else:
        scaled_dataset = dataframe.copy()

    numeric_dataset = scaled_dataset.select_dtypes(include=np.number)
    individual_scores = silhouette_samples(numeric_dataset, labels)

    if cellid_col is not None:
        individual_scores_df = pd.DataFrame(
            {
                cellid_col: dataframe[cellid_col].values,
                "Population": labels,
                "Silhouette_score": individual_scores,
            },
            index=dataframe.index,
        )
    else:
        individual_scores_df = pd.DataFrame(
            {
                "Population": labels,
                "Silhouette_score": individual_scores,
            },
            index=dataframe.index,
        )

    return individual_scores_df


def silhouette_score_indiv_linkage(dataframe, clusternum, cellid_col=None):
    numeric_dataset = dataframe.select_dtypes(include=np.number)
    scaled_dataset = utils.scale_data(numeric_dataset)

    linkage_matrix = linkage(scaled_dataset, method="ward", metric="euclidean")
    labels = fcluster(linkage_matrix, t=clusternum, criterion="maxclust")

    individual_scores_df = silhouette_score_indiv(
        numeric_dataset, labels, cellid_col=cellid_col
    )

    return individual_scores_df


def silhouette_score_best_cluster_num(
    dataframe,
    clustering_algorithm="ward",
    id_col="Metadata_CellID",
    distance_metric="euclidean",
    clusternum_min=2,
    clusternum_max=10,
    plot=False,
    scale=True,
    threshold=0,
    individual_scores=False,
):
    numeric_dataset = dataframe.select_dtypes(include=np.number)
    non_numeric_cols = dataframe.select_dtypes(exclude=["number"]).columns.tolist()
    print(f"dropping {non_numeric_cols} for calculating silhouette score")

    if scale:
        scaled_dataset = utils.scale_data(numeric_dataset)

    else:
        scaled_dataset = numeric_dataset.to_numpy()

    linkage_matrix = linkage(
        scaled_dataset, method=clustering_algorithm, metric=distance_metric
    )

    best_score = -1
    best_k = 0

    for k in range(clusternum_min, clusternum_max):
        if clustering_algorithm == "KMeans":
            kmeans = skclus.KMeans(n_clusters=k)
            labels = kmeans.fit_predict(scaled_dataset)
        else:
            labels = fcluster(linkage_matrix, t=k, criterion="maxclust")

        score = silhouette_score(scaled_dataset, labels)
        if score > best_score:
            best_score = score
            best_k = k

    if plot:
        # set up plotting of all cluster number silhouette plots
        return

    if (
        best_score < threshold
    ):  # minimum average silhouette score for dataset to be considered clustered
        best_k = 1

    if individual_scores:
        individual_scores_df = silhouette_score_indiv(
            dataframe, best_k, cellid_col=id_col
        )
        return best_k, best_score, individual_scores_df

    else:
        return best_k, best_score


def silhouette_plot(silhouette_individual, labels, savedir="", title=""):
    cluster_num = len(np.unique(labels))
    silhouette_avg = silhouette_individual.mean()
    colors = sns.color_palette("hls", cluster_num)
    labels_unique = sorted(list(labels.unique()))

    y_lower = 10

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, color in zip(labels_unique, colors):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = silhouette_individual[
            labels == label
        ].to_numpy()

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        trans = ax.get_xaxis_transform()
        plt.text(
            silhouette_avg + 0.05,
            0.05,
            "silhouette average: " + str(round(silhouette_avg, 4)),
            transform=trans,
        )  # Label line with silhouette score

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    if title == "":
        ax.set_title(
            "Silhouette analysis for clustering with n_clusters = %d" % cluster_num,
        )
    else:
        ax.set_title(
            f"{title} Silhouette analysis clustering with n_clusters = %d"
            % cluster_num,
        )

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    # plt.text(10.1, 0, silhouette_avg)  # Label line with silhouette score

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xlim([-0.1, 1.0])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    if savedir == "":
        print("Plot not saved")
    else:
        filename = "silhouette_score_ncluster" + str(cluster_num) + ".png"

        plt.savefig(savedir / filename)
        print("Plot saved")

    plt.show()

    return


def silhouette_score_stats(silhouette_score_df, id_col="Metadata_CellID"):
    silhouette_score_stats_dict = {}

    silhouette_score_stats_dict["silhouette_score_std"] = silhouette_score_df[
        "Silhouette_score"
    ].std()
    silhouette_score_stats_dict["silhouette_score_avg"] = silhouette_score_df[
        "Silhouette_score"
    ].mean()

    # get most representative cells of subpopulations
    silhouette_score_stats_dict["subpopulation_ranked_cells"] = (
        silhouette_score_df.sort_values(
            ["Population", "Silhouette_score"], ascending=[True, False]
        )
        .groupby("Population", group_keys=False)
        .apply(
            lambda g: list(zip(g[id_col], g["Silhouette_score"])),
            include_groups=False,
        )
        .to_dict()
    )

    silhouette_score_stats_dict["subpopulation_silhouette_score_avg"] = (
        silhouette_score_df.groupby("Population")["Silhouette_score"].mean().to_dict()
    )

    return silhouette_score_stats_dict


def filter_dataframes(all_sheets: dict, keep_cols: dict, savedir=""):
    all_sheets_filtered = {}
    for organelle, df in all_sheets.items():
        col_list = list(keep_cols[organelle][0])
        all_sheets_filtered[organelle] = utils.filter_cols(df, col_list)

    if savedir != "":
        with pd.ExcelWriter(savedir / "updated_organelle_measures.xlsx") as writer:
            for sheet, df in all_sheets_filtered.items():
                df.to_excel(writer, sheet_name=sheet, index=False)

    return all_sheets_filtered


def prepare_HCA_sheet(
    all_sheets: dict, keep_cols: dict, savedir="", variable_features=True, filter_col = False
):
    if filter_col:
        all_sheets_filtered = filter_dataframes(all_sheets, keep_cols, savedir="")
    else:
        all_sheets_filtered = all_sheets

    summary = utils.calculate_summary_stats(all_sheets_filtered)

    # save summary file
    if savedir != "":
        with pd.ExcelWriter(savedir / "summary_stats.xlsx") as writer:
            for sheet, df in summary.items():
                df.to_excel(writer, sheet_name=sheet, index=False)

    combined_df = utils.combine_dataframes_by_CellID(summary)
    combined_df_non_empty = combined_df.dropna(axis=1, how="all")
    
    '''
    combined_df_non_empty, dropped_cols_empty = utils.drop_empty_cols(
        combined_df, threshold=0
    )  # 0.2 = 20% empty
    '''
    
    if variable_features:
        combined_df_clean, dropped_cols_clean = utils.drop_no_variation_cols(
            combined_df_non_empty, threshold= 0 #0.1 # removes 800 features
        )
    else:
        combined_df_clean = combined_df_non_empty
        dropped_cols_clean = 0

    if savedir != "":
        combined_df_clean.to_excel(savedir / "HCA.xlsx", index=False)

    return combined_df_clean, dropped_cols_clean, #dropped_cols_empty,


def add_cluster_label(
    dataframe,
    cluster_num=1,
    identify_clusternum_method=None,
    clustering_algorithm="KMeans",
    id_col=None,
    lbl_colname="",
):
    """
    Assigns cluster label to dataframe
    dataframe: scaled dataframe

    """

    if id_col is not None:
        numeric_df = dataframe.drop(columns=id_col)
    else:
        numeric_df = dataframe.copy()

    if cluster_num == 0:
        if identify_clusternum_method == "gap":
            # No defined cluster number then find optimal from gap_statistic or silhouette score
            data_array = numeric_df.to_numpy()
            cluster_num = gap_statistic(data_array, max_k=10, B=10, random_state=42)
        elif identify_clusternum_method == "silhouette":
            cluster_num, _ = silhouette_score_best_cluster_num(
                numeric_df, individual_scores=False, scale=False
            )
        else:
            print("No valid cluster method specified - defaulting to 1 cluster")
            cluster_num = 1

    if lbl_colname == "":
        lbl_colname = f"Classification_{clustering_algorithm}"

    dataframe_cluster = dataframe.copy()
    if len(numeric_df) >= 2:
        # numeric_dataset = dataframe.select_dtypes(include=np.number) # if scaled should all be numeric
        if clustering_algorithm == "KMeans":
            kmeans = skclus.KMeans(n_clusters=cluster_num, random_state=42)
            labels = kmeans.fit_predict(numeric_df)
            dataframe_cluster[lbl_colname] = labels

        elif clustering_algorithm == "HCA":
            clustered = linkage(numeric_df, method="ward", metric="euclidean")

            labels = cut_tree(clustered, n_clusters=cluster_num).reshape(-1)
            dataframe_cluster[lbl_colname] = labels

        # start grouping label at 1 instead of 0
        dataframe_cluster[lbl_colname] = dataframe_cluster[lbl_colname] + 1

    else:
        dataframe_cluster = dataframe.copy()
        dataframe_cluster[lbl_colname] = 1

    if id_col is not None:
        classification_df = dataframe_cluster[[id_col, lbl_colname]]
    else:
        id_col = pd.Series(list(range(1, len(dataframe) + 1)), name="ObjectNumber")
        labels = dataframe_cluster[lbl_colname]
        classification_df = pd.concat([id_col, labels], axis=1)

    return dataframe_cluster, classification_df


# %% HCA


def HCA_clustering(
    dataset_full: pd.DataFrame,
    title="",
    subset=[],
    color_by="organelle",
    clusternum=1,
    savedir="",
    plot_median=False,
    pltsize=(60, 30),
    color_row=True,
    lbl_colname="Subtype",
    filename="",
    cmap=None,
):
    filename = ""
    if subset != []:
        fullstring = ""
        flat_subset = [x for xs in subset for x in xs]
        for w in flat_subset:
            fullstring += "_" + w

        filename = "_" + fullstring

    if subset != []:
        dataset_processing = utils.extract_df_subset_sequential(
            dataset_full, subset=subset, constant_key=constants.CELLID_COL
        )
    else:
        dataset_processing = dataset_full

    # Clustering and labelling dataframe
    dataframe_scaled = utils.scale_data(dataset_processing)
    dataframe_labeled, _ = add_cluster_label(
        dataframe_scaled,
        clusternum,
        clustering_algorithm="HCA",
        lbl_colname=lbl_colname,
        id_col=constants.CELLID_COL,
    )

    # get CellID belonging to each subpopulation
    subpopulation_cells = {}
    for i in list(range(1, clusternum + 1)):  # start subpopulation count at 1
        subpopulation_cells["Supopulation_" + str(i)] = dataframe_labeled.loc[
            dataframe_labeled[lbl_colname] == i, constants.CELLID_COL
        ].tolist()

    if plot_median:
        heatmap = utils.show_sorted_heatmap(dataset_processing, plot_title=f"{title}")
        if savedir != "":
            utils.save_svg(heatmap, savedir, f"heatmap_{title}.svg")

        heatmap.show()

    clustermap = plot.show_clustermap(
        dataframe_labeled,
        plot_title=title,
        color_feat=color_by,
        pltsize=pltsize,
        color_row=color_row,
        id_col=constants.CELLID_COL,
        label_col=dataframe_labeled.columns[-1],
        color_row_cmap=cmap,
    )
    clustermap.fig
    if savedir != "":
        utils.save_svg(plt, savedir, f"clustered_heatmap_{title}{filename}.svg")

    plt.show()

    return dataframe_labeled, subpopulation_cells


# %% KMeans
# Kmeans cluster and silhouette plot function
def KMeans_2D(
    dataframe,
    cluster_num=0,
    identify_clusternum_method="silhouette",
    savedir="",
    plot=True,
    id_col=None,
):
    """
    K means clustering of dataset

    Input
        dataframe: scaled dataframe
        cluster_num: number of clusters to group data into (default: will use gap statistic defined number of clusters)
        savedir: save directory to save plot of silhouette score and data clustering (default: does not save plot)

    Output
        silhouette_individual: silhouette scores of each datapoint
        silhouette_avg: average silhouette score

    """
    if "Metadata_CellID" in dataframe.columns:
        dataframe_numerical = dataframe.drop(columns="Metadata_CellID")
    else:
        dataframe_numerical = dataframe

    scaler = StandardScaler()
    dataframe_scaled = scaler.fit_transform(dataframe_numerical)

    # format dataframe to be compatible with KMeans
    # data = dataframe_scaled.values.reshape(-1, 1)
    kmeans = skclus.KMeans(n_clusters=cluster_num)
    labels = kmeans.fit_predict(dataframe_scaled)
    dataframe_labelled = dataframe.copy()
    dataframe_labelled["KMeans_Population"] = labels

    silhouette_avg = skmet.silhouette_score(dataframe_scaled, labels)
    silhouette_individual_df = silhouette_score_indiv_linkage(dataframe, cluster_num)
    silhouette_individual = silhouette_individual_df["Silhouette_score"]

    if plot:
        silhouette_plot(silhouette_individual, labels, savedir=savedir)

    return dataframe_labelled, silhouette_individual_df, silhouette_avg


# %% dimension reduction


def MRMR_dataframe(dataframe, target, max_features=0, title="", id_col=None, axes_show=True):
    if id_col is not None:
        dataframe = dataframe.drop(columns=id_col)

    features_original = dataframe.columns.tolist()
    feature_num_original = {}
    for keyword in constants.ORGANELLE_LIST:
        feature_num_original[keyword] = utils.keyword_occurrence_list(
            features_original, [keyword]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        dataframe, target, test_size=0.3, random_state=0
    )

    if max_features == 0:
        sel = MRMR(method="FCQ", regression=False)
    else:
        sel = MRMR(method="FCQ", regression=False, max_features=max_features)

    sel.fit(dataframe, target)
    #sel.relevance_
    feature_scores = pd.DataFrame()
    feature_scores["Feature"] = sel.variables_
    feature_scores["Score"] = sel.relevance_
    # dropped = sel.features_to_drop_

    Xtr = sel.transform(X_test)
    features_sel = Xtr.columns.tolist()

    # Plot types of feature organelles selected by MRMR
    feature_num = {}
    for keyword in constants.ORGANELLE_LIST:
        feature_num[keyword] = utils.keyword_occurrence_list(features_sel, [keyword])

    colors = [constants.ORGANELLE_CMAP[k] for k in feature_num.keys()]
    plt.figure(figsize=(5, 5))
    plt.bar(feature_num.keys(), feature_num.values(), 0.5, color=colors)
    #plt.title(f"Number of Features per Organelle Selected by MRMR {title}")
    plt.grid(False)

    ax = plt.gca()
    plot.bar_plot_formatting(ax)
    if axes_show == False:
        plt.xticks([])

    plt.show()

    #%% plot score of feature organelles selected by MRMR
    feature_score_organelle = {}
    for keyword in constants.ORGANELLE_LIST:
        feature_score_organelle[keyword] = feature_scores.loc[
        feature_scores["Feature"].str.contains(keyword, case=False, na=False),
        "Score"
    ]

    total_score = {}
    for organelle, scores in feature_score_organelle.items():
        total_score[organelle] = scores.sum()
        #average_contribution_organelle[organelle] = scores.mean()

    colors = [constants.ORGANELLE_CMAP[k] for k in total_score.keys()]
    plt.figure(figsize=(5, 5))
    plt.bar(total_score.keys(), total_score.values(), 0.5, color=colors)
    #plt.title(f"Total Relevance Score of Features per Organelle Selected by MRMR {title}")
    plt.grid(False)

    ax = plt.gca()
    plot.bar_plot_formatting(ax)
    plt.show()

    featuretype_num = {}
    for keyword in [
        #"AreaShape",
        #"RadialDistribution",
        #"Intensity",
        #"Texture",
        #"Structure",
        "Geometric",
        "Densitometric",
        "Textural",
        "Structural",
    ]:
        featuretype_num[keyword] = utils.keyword_occurrence_list(
            features_sel, [keyword]
        )

    colors = [constants.FEATURE_TYPE_CMAP[k] for k in featuretype_num.keys()]
    
    plt.figure(figsize=(5, 5))
    plt.bar(featuretype_num.keys(), featuretype_num.values(), 0.5, color=colors)
    #plt.title(f"Number of Features by Type Selected by MRMR {title}")
    plt.xticks(rotation=45)
    plt.grid(False)

    ax = plt.gca()
    plot.bar_plot_formatting(ax)
    plt.show()

    combined_df = pd.DataFrame()
    for organelle in constants.ORGANELLE_LIST:
        for feature_type in [
            #"AreaShape",
            #"RadialDistribution",
            #"Intensity",
            #"Texture",
            #"Structure",
            "Geometric",
            "Densitometric",
            "Textural",
            "Structural",
        ]:
            keywords = [organelle, feature_type]

            combo_df = pd.DataFrame(
                {
                    "Organelle": [organelle],
                    "Type": [feature_type],
                    "Count": [utils.keyword_occurrence_list(features_sel, keywords)],
                }
            )

            combined_df = pd.concat([combined_df, combo_df], ignore_index=True)
    
    pivot_df = combined_df.pivot(index="Organelle", columns="Type", values="Count")
    colors = [constants.FEATURE_TYPE_CMAP[col] for col in pivot_df.columns]

    ax = pivot_df.plot(kind="bar", stacked=True, figsize=(5, 5), color=colors, edgecolor="none", linewidth=0)
    plt.grid(False)

    plot.bar_plot_formatting(ax)
    # label bars with counts
    #for container in ax.containers:
        #ax.bar_label(container, label_type="center", fmt="%d")  # '%d' for integers

    #plt.ylabel("Number of Features")
    #plt.title(f"Organelle and Feature Types Selected by MRMR {title}")
    plt.legend(title="Feature Type")
    plt.tight_layout()
    if axes_show == False:
        plt.xticks([])
        plt.xlabel("")
    plt.show()

    return features_sel, sel


def plot_feature_relevance(sel, keyword_cmap=None):
    s = pd.Series(sel.relevance_, index=sel.variables_).sort_values(ascending=False)

    if keyword_cmap is None:
        ax = s.plot.bar(figsize=(7, 7))

        plt.title("Most Relevant Features for Clustering")
        plt.ylabel("F-statistic/Correlation Ratio")
        plt.xlabel("Feature")
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

        return plt

    colors = []
    legend_items = {}  # legend_label -> color (avoid duplicates)

    for feat in s.index:
        feat_str = str(feat).lower()

        # default fallback
        assigned_color = "tab:blue"
        legend_label = "Other"

        matched = False
        for key, (col, lbl) in keyword_cmap.items():
            if key.lower() in feat_str:
                assigned_color = col
                legend_label = lbl
                matched = True
                break

        # Track legend items uniquely
        if legend_label not in legend_items:
            legend_items[legend_label] = assigned_color

        colors.append(assigned_color)

    plt.figure(figsize=(7, 7))
    plt.bar(s.index, s.values, color=colors)

    plt.title("Most Relevant Features for Clustering")
    plt.ylabel("F-statistic/Correlation Ratio")
    plt.xlabel("Feature")
    plt.xticks(rotation=90)

    patches = [Patch(color=col, label=lbl) for lbl, col in legend_items.items()]
    plt.legend(handles=patches, title="Shape descriptor type")

    plt.tight_layout()
    plt.show()

    return plt


def extract_col_row_subset(
    dataframe_full,
    col_subset=None,
    row_subset=None,
    row_filter_col=constants.CELLID_COL,
    scale=True,
):
    """
    select subset of columns
    scales data column-wise (across all rows) if scale = True
    select subset of rows
    returns filtered dataframe

    :param dataframe_full: Description
    :param col_subset: Description
    :param row_subset: Description
    """
    if col_subset is None and row_subset is None:
        print(
            "No subset specified, returning full unlabelled dataframe - use add_cluster_label"
        )
        return dataframe_full
    else:
        if col_subset is not None:
            dataset_filtered = utils.extract_df_subset_sequential(
                dataframe_full, subset=col_subset, constant_key=constants.CELLID_COL
            )

    if scale:
        dataset_filtered_scaled = utils.scale_data(dataset_filtered)
    else:
        dataset_filtered_scaled = dataset_filtered.copy()

    # select row subset of dataset after scaling to all rows
    if row_subset is not None:
        dataset_filtered_final = dataset_filtered_scaled[
            dataset_filtered_scaled[row_filter_col].isin(row_subset)
        ]

    return dataset_filtered_final


def extract_top_features(dataframe_labelled, clusternum, name="", max_features=0):
    """
    Extracts most relevant features contributing to labelling

    :param dataframe_labelled: labelled UNSCALED dataframe
    :param name: Description
    :param max_features: Description
    """

    # MRMR cannot handle empty columns
    dataset_nonempty, dataset_nan = utils.drop_empty_cols(
        dataframe_labelled.drop(columns=constants.CELLID_COL), threshold=0
    )
    X = dataset_nonempty.loc[:, dataset_nonempty.std(axis=0, skipna=True) != 0]

    y = dataframe_labelled["Subtype"]

    selected_features, sel = MRMR_dataframe(X, y, title=name, max_features=max_features)

    relevance_noNaN = np.nan_to_num(sel.relevance_, nan=0, posinf=0, neginf=0)

    features_sorted = pd.Series(relevance_noNaN, index=sel.variables_).sort_values(
        ascending=False
    )

    # copy_list_keyboard(features_sorted[0:len(selected_features)].index.to_list())
    # utils.copy_list_keyboard(features_sorted[0:30].index.to_list())

    ax = features_sorted[0:20].plot.bar(figsize=(15, 4))
    plt.ylabel(
        "relevance / redundancy ratio (F-statistic / Pearson’s correlation coefficient)"
    )
    ax.yaxis.set_label_coords(-0.03, 0.15)

    plt.xlabel(f"{name} feature")
    plt.title(
        f"top 20 relevant features for separation into {clusternum} {name} subpopulations"
    )
    plt.show()

    return selected_features, features_sorted, sel


def HCA_top_features(
    HCA_filtered,
    keyword_filter_list,
    name="",
    best_k=0,
    savedir="",
    title="",
    color_row=True,
    cmap=None,
):
    keyword_filter_flat_list = [x for xs in keyword_filter_list for x in xs]
    keyword_list_name = "_".join(keyword_filter_flat_list)

    dataset_subset = utils.extract_df_subset_sequential(
        HCA_filtered, subset=keyword_filter_list, constant_key="Metadata_CellID"
    )

    if best_k == 0:
        best_k, best_score, indiv_score = utils.silhouette_score_best_cluster_num(
            dataset_subset, clusternum_max=5, individual_scores=True
        )

    HCA_df_labelled, subpopulation_cells = HCA_clustering(
        dataset_subset,
        subset=keyword_filter_list,
        color_by="type",
        clusternum=best_k,
        savedir=savedir,
        pltsize=(50, 30),
        title=title,
        color_row=color_row,
        cmap=cmap,
        lbl_colname="Cluster",
    )

    selected_features, features_sorted, sel = extract_top_features(
        dataset_subset,
        best_k,
        subset=keyword_filter_list,
        name=f"{name} {keyword_list_name}",
        max_features=20,
    )

    if best_k != 1:
        indiv_score = silhouette_score_indiv(
            dataset_subset,
            HCA_df_labelled["Cluster"],
            cellid_col=constants.CELLID_COL,
        )
        silhouette_scores_dict = silhouette_score_stats(indiv_score)
    else:
        silhouette_scores_dict = {}

    # plot sorted MRMR scores
    plt_feat = features_sorted.plot()
    plt_feat.xaxis.set_ticklabels([])
    plt_feat.set_ylabel("F-statistic : Correlation ratio")
    plt_feat.set_xlabel("Features (sorted)")
    plt_feat.set_title("Features sorted by MRMR statistic")
    plt.show()

    return (
        HCA_df_labelled,
        subpopulation_cells,
        selected_features,
        silhouette_scores_dict,
    )


def PCA_n_components(dataframe, percent_threshold=0.95):
    """
    Determine number of PCA components needed to describe threshold variance
    """
    if "Metadata_CellID" in dataframe.columns:
        dataframe_numerical = dataframe.drop(columns="Metadata_CellID")
    else:
        dataframe_numerical = dataframe

    scaled = StandardScaler().fit_transform(dataframe_numerical)
    # scaled_data = np.nan_to_num(scaled, nan=0.0)

    pca = PCA(n_components=percent_threshold)
    reduced = pca.fit_transform(scaled)

    n_dimensions = reduced.shape[1]

    return n_dimensions


def label_color_mapping_dict(label_series, palette="hls", label_name=False):
    cluster_num = label_series.unique().shape[0]
    labels_unique = sorted(list(label_series.unique()))
    colors = sns.color_palette(palette, cluster_num)
    label_color_mapping = {int(lab): col for lab, col in zip(labels_unique, colors)}

    return labels_unique, colors, label_color_mapping


def PCA_components(
    dataframe_full,
    n_PCA_components=2,
    cluster_num=1,
    legend=False,
    plot=False,
    title="",
    cluster_method=None,
    id_col=None,
    scale=True,
):
    if id_col in dataframe_full.columns:
        dataframe_numerical = dataframe_full.drop(columns=id_col)
    else:
        dataframe_numerical = dataframe_full

    if scale:
        scaled = StandardScaler().fit_transform(dataframe_numerical)
        scaled_df = pd.DataFrame(
            scaled, columns=dataframe_numerical.columns, index=dataframe_numerical.index
        )

    # scaled_data = np.nan_to_num(scaled, nan=0.0)
    pca = PCA(n_components=n_PCA_components)
    principalComponents = pca.fit_transform(scaled)

    column_name_list = [f"PC{i}" for i in list(range(1, n_PCA_components + 1))]
    principal_df = pd.DataFrame(data=principalComponents, columns=column_name_list)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_matrix = pd.DataFrame(
        loadings, columns=column_name_list, index=dataframe_numerical.columns
    )
    explained_variance = pca.explained_variance_ratio_

    fig = None

    if plot:
        if cluster_method == "HCA":
            df_labelled, _ = add_cluster_label(
                principal_df,
                cluster_num=cluster_num,
                clustering_algorithm="HCA",
            )
            combined_df = pd.concat(
                [principal_df, df_labelled[df_labelled.columns[-1]]], axis=1
            )

        elif cluster_method == "KMeans":
            df_labelled, _ = add_cluster_label(
                principal_df,
                cluster_num=cluster_num,
                clustering_algorithm="KMeans",
            )
            combined_df = pd.concat(
                [principal_df, df_labelled[df_labelled.columns[-1]]], axis=1
            )

        else:
            combined_df = principal_df.copy()
            combined_df["Cluster"] = [0] * len(principal_df)

    label_series = combined_df[combined_df.columns[-1]]
    labels, colors, label_color_mapping = label_color_mapping_dict(
        label_series, palette="inferno"
    )

    # Format axis labels with explained variance %
    def pc_label(idx):
        return f"PC{idx + 1} ({explained_variance[idx] * 100:.1f}%)"

    if plot and n_PCA_components == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(pc_label(0), fontsize=15)
        ax.set_ylabel(pc_label(1), fontsize=15)
        ax.set_title(
            f"PCA - total explained variance {sum(explained_variance) * 100:.2f}% {title}",
            fontsize=20,
        )

        for label, color in zip(labels, colors):
            indicesToKeep = combined_df[combined_df.columns[-1]] == label
            ax.scatter(
                combined_df.loc[indicesToKeep, "PC1"],
                combined_df.loc[indicesToKeep, "PC2"],
                c=color,
                s=50,
            )

        ax.legend(labels)  # , title="Treatment")
        ax.grid()

    elif n_PCA_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(pc_label(0), fontsize=12)
        ax.set_ylabel(pc_label(1), fontsize=12)
        ax.set_zlabel(pc_label(2), fontsize=12)
        ax.set_title(
            f"3D PCA - total explained variance {sum(explained_variance[:3]) * 100:.2f}% {title}",
            fontsize=16,
        )

        for label, color in zip(labels, colors):
            indicesToKeep = combined_df[combined_df.columns[-1]] == label
            ax.scatter(
                combined_df.loc[indicesToKeep, "PC1"],
                combined_df.loc[indicesToKeep, "PC2"],
                combined_df.loc[indicesToKeep, "PC3"],
                c=[color],
                s=40,
                depthshade=True,
            )
        ax.legend(labels)

    else:
        fig = None

    return (
        fig,
        principal_df,
        pca,
        loadings_matrix,
        combined_df,
        colors,
        labels,
        label_color_mapping,
    )


# %% random forest
def random_forest_train_model(
    dataset_full,
    clustercol,
    id_col="Metadata_CellID",
    savedir=None,
    weighted=False,
    weights=[],
    scale = True,
):
    data_scaler = skprep.StandardScaler()

    X = dataset_full.drop(columns=[clustercol, id_col])
    Y = dataset_full[clustercol]

    if weighted:
        X_train_augmented = np.column_stack([X, weights])
        X_scaled = data_scaler.fit_transform(X_train_augmented)
    elif not scale:
        X_scaled = X.values
    else:
        # scale data
        X_scaled = data_scaler.fit_transform(X)


    rf_model = RandomForestClassifier(
        n_estimators=100,  # number of trees (can tune later)
        max_depth=None,  # depth of trees (None = grow fully)
        random_state=42,  # reproducibility
        n_jobs=3,  # use 3 CPU cores
    )

    rf_model.fit(X_scaled, Y)  # scores ∈ [0, 1]

    if savedir is not None:
        # save model
        joblib.dump(rf_model, savedir / "random_forest_model.joblib")
        # Save scaler
        joblib.dump(data_scaler, savedir / "random_forest_data_scalar.joblib")

    # random forest model evaluation (cross validate)
    scores = cross_val_score(rf_model, X_scaled, Y, cv=5, scoring="accuracy")

    # confusion matrix - where model makes mistakes
    # classification report for models
    # check feature importance
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    return (
        rf_model,
        data_scaler,
        scores,
        scores.mean(),
        scores.std(),
        feature_importance,
    )


def random_forest_classification(
    model,
    data_scaler,
    dataset_full,
    id_col="Metadata_CellID",
    weighted=False,
    weights=[],
    clustercol="Random_forest_population",
    confidencecol="RF_confidence",
    scale=True,
):
    dataset_full_numeric = dataset_full.drop(columns=id_col)

    if weighted:
        dataset_full_numeric_weighted = np.column_stack([dataset_full_numeric, weights])
    else:
        dataset_full_numeric_weighted = dataset_full_numeric

    if scale:
        X_scaled = data_scaler.transform(
            dataset_full_numeric_weighted
        )  # important: use SAME scaler
    else:
        X_scaled = dataset_full_numeric_weighted.values

    y_predicted = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    confidence = y_proba.max(axis=1)
    sorted_proba = np.sort(y_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    entropy_scores = entropy(y_proba.T)

    dataset_clustered = dataset_full.copy()
    dataset_clustered[clustercol] = y_predicted
    dataset_clustered[confidencecol] = confidence
    dataset_clustered["RF_margin"] = margin
    dataset_clustered["RF_entropy"] = entropy_scores

    ncluster = dataset_clustered[clustercol].nunique()
    
    subpopulation_cells = {}
    for i in set(y_predicted):
        subpopulation_cells[int(i)] = dataset_clustered.loc[
            dataset_clustered[clustercol] == i, id_col
        ].tolist()

    representative_cells = (
        dataset_clustered[[id_col, clustercol, confidencecol]]
        .sort_values([clustercol, confidencecol], ascending=[True, False])
        .groupby(clustercol)[id_col]
        .apply(list)
        .to_dict()
    )

    feature_importance = pd.Series(
        model.feature_importances_, index=dataset_full_numeric.columns
    )
    feature_importance = feature_importance.sort_values(ascending=False)

    return dataset_clustered, subpopulation_cells, representative_cells, feature_importance


def import_model(datadir: Path, model_name: str, scaler_name: str):
    rf_model = joblib.load(datadir / model_name)
    data_scaler = joblib.load(datadir / scaler_name)

    return rf_model, data_scaler


def model_eval(X_scaled, Y, rf_model = None):
    # Define the model
    if rf_model is None:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=3)

    # Cross-validate
    scores = cross_val_score(rf_model, X_scaled, Y, cv=5, scoring="accuracy")

    eval_dict = {
        "Cross-validated accuracies": scores,
        "Mean accuracy": scores.mean(),
        "Standard deviation": scores.std(),
    }

    return eval_dict

# dendrogram select clusternum 
def compute_linkage(X, method='ward'):
    """
    X: array-like (n_samples, n_features) or pandas DataFrame
    method: 'ward','single','complete','average',...
    returns: linkage matrix (scipy format)
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    Z = linkage(X, method=method)
    return Z

def plot_dendrogram(Z, truncate_mode=None, p=30, figsize=(10,5), annotate_above=None):
    """
    Plot dendrogram. Optionally truncate for long trees.
    annotate_above: annotate node distances above this value
    """
    plt.figure(figsize=figsize)
    dn = dendrogram(Z, truncate_mode=truncate_mode, p=p)
    if annotate_above is not None:
        # annotate node heights
        for i, d, c in zip(dn['icoord'], dn['dcoord'], dn['ivl']):
            # this only annotates the final leaves when truncate_mode is not None;
            pass
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

def flat_clusters_from_distance(Z, threshold):
    """
    Cut the dendrogram at distance=threshold and return cluster labels (1..k)
    """
    labels = fcluster(Z, t=threshold, criterion='distance')
    return labels

def suggest_threshold_by_max_gap(Z, top_gap_index=0):
    """
    Heuristic: find the largest gap between successive linkage distances.
    Z: linkage matrix (n-1 rows). Returns (suggested_threshold, suggested_n_clusters, gap_index)
    top_gap_index: 0 -> largest gap, 1 -> second largest gap, etc.
    """
    distances = Z[:, 2]  # distances are sorted non-decreasing
    diffs = np.diff(distances)  # length n-2
    if len(diffs) == 0:
        # trivial small dataset
        return (None, 1, None)
    # index of the sorted diffs (descending)
    sorted_idx = np.argsort(diffs)[::-1]
    idx = sorted_idx[top_gap_index]  # index i where gap is between distances[i] and distances[i+1]
    # suggested cut threshold is midpoint between distances[i] and distances[i+1]
    thr = (distances[idx] + distances[idx+1]) / 2.0
    n_obs = Z.shape[0] + 1
    # number of clusters after cutting between distances[idx] and distances[idx+1]:
    # after i+1 merges you have n - (i+1) clusters
    suggested_k = n_obs - (idx + 1)
    return (thr, suggested_k, idx)

def choose_k_by_silhouette(X, Z=None, k_min=2, k_max=10, linkage_method_for_agglomerative='ward'):
    """
    Try k in [k_min..k_max] and compute silhouette score using AgglomerativeClustering.
    Returns DataFrame with k and silhouette, plus the best k.
    Note: silhouette requires at least 2 clusters and <= n-1.
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    n = X_arr.shape[0]
    k_max = min(k_max, n-1)
    results = []
    for k in range(k_min, max(k_min, k_max)+1):
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method_for_agglomerative)
        labels = model.fit_predict(X_arr)
        # silhouette can fail if a cluster has a single sample -> catch exceptions
        try:
            s = silhouette_score(X_arr, labels)
        except Exception:
            s = np.nan
        results.append({'k': k, 'silhouette': s})
    df_res = pd.DataFrame(results)
    best_row = df_res.loc[df_res['silhouette'].idxmax()] if df_res['silhouette'].notna().any() else None
    return df_res, best_row