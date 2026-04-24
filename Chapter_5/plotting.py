import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
from pathlib import Path
import plotly.io as pio
import json

import random
import colorsys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import data_analysis_utils as utils
from matplotlib import gridspec
import classification as clus
from itertools import combinations
import webbrowser
from matplotlib.patches import Patch
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D

import constants

import re

try:
    from scipy.stats import gaussian_kde

    _HAS_SCIPY = True
except Exception:
    gaussian_kde = None
    _HAS_SCIPY = False

def bar_plot_formatting(ax,
                   y_major=5, y_minor=1,
                   y_min=0, y_max=None,
                   spine_color="black", spine_width=1,
                   y_tick_labelsize=10):

    plt.grid(False)

    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
        spine.set_color(spine_color)

    ax.tick_params(
    axis='y',
    which='major',
    left=True,        # show tick marks
    labelleft=True,   # show tick labels
    length=5,         # tick length
    width=1,        # tick thickness
    labelsize=y_tick_labelsize      # font size
    )
    
    return ax

def get_color(label, color_by ="organelle"):
    if color_by == "organelle":
        color_map = constants.ORGANELLE_CMAP
    elif color_by == "feature_type":
        color_map = constants.FEATURE_TYPE_CMAP
    else:
        color_map = {}

    for keyword, color in color_map.items():
        if keyword.lower() in label.lower():
            return color
        
    return "black"

def generate_color_list_RGB(num_colors, shuffle=False, lightness=0.5, saturation=0.9):
    """
    Generate a list of visually distinct RGB colors.

    Parameters
    ----------
    n : int
    Number of colors to generate.
    shuffle : bool
    Whether to shuffle the colors (otherwise evenly spaced around HSV wheel).
    lightness : float
    Brightness of the colors (0–1).
    saturation : float
    Color saturation (0–1).

    Returns
    -------
    list of tuple
    List of (R, G, B) values in range 0–255.
    """

    hues = [i / num_colors for i in range(num_colors)]
    if shuffle:
        random.shuffle(hues)

    color_list = [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h, saturation, lightness))
        for h in hues
    ]

    return color_list


def col_color_list(col_names, color_dict):
    # create a list of column colors based on keywords
    col_colors = []
    for name in col_names:
        name_lower = name.lower()
        for keyword, color in color_dict.items():
            keyword_lower = keyword.lower()
            if keyword_lower in name_lower:
                col_colors.append(color)
                break
        else:
            col_colors.append("black")

    return col_colors


def sns_colors_to_hex(sns_colors):
    """Convert a list of sns RGB tuples (0–1 floats) into hex strings."""
    return [
        "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in sns_colors
    ]


def _point_density_using_kde(x, y, bw_method=None):
    """Return density values for each (x,y) using gaussian_kde."""
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bw_method)
    dens = kde.evaluate(xy)
    return dens


def _point_density_hist2d(x, y, bins=100):
    """
    Fallback density estimate using 2D histogram:
    returns density for each point by looking up its histogram bin count.
    """
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    # map each point to its bin
    x_idx = np.searchsorted(xedges, x, side="right") - 1
    y_idx = np.searchsorted(yedges, y, side="right") - 1
    # clip indices
    x_idx = np.clip(x_idx, 0, counts.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, counts.shape[1] - 1)
    dens = counts[x_idx, y_idx]
    # normalize
    if np.max(dens) > 0:
        dens = dens / np.max(dens)
    return dens

def rgb_to_hex(c):
    if isinstance(c, str): return c
    r, g, b = c
    if all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in (r, g, b)):
        r, g, b = [int(v*255) for v in (r, g, b)]
    else:
        r, g, b = [int(v) for v in (r, g, b)]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def scatter_plot(
    x,
    y,
    axis_label=["X values", "Y values"],
    title="",
    labels=None,
    cluster_colors=None,
    color_points=None,
    density_cmap="viridis",
    kind="density",
    s=40,
    alpha=0.9,
    bw_method=None,
    hist_bins=100,
    per_label_density=False,
    figsize=(8, 8),
    show_grid=True,
):
    """
    if labels is not None and len(labels) > 0:
        labelnum = len(set(labels))
        color_list = sns.color_palette("hls", labelnum)
        label_to_color = {lbl: color_list[i] for i, lbl in enumerate(set(labels))}
        point_colors = [label_to_color[lbl] for lbl in labels]
    else:
        point_colors = color_points if color_points is not None else "black"
    """
    # If labels AND cluster_colors AND cluster label names are provided:
    if cluster_colors is not None and labels is not None:
        marker_color = [cluster_colors[l] for l in labels]
        label_to_color = {lab: cluster_colors[lab] for lab in sorted(set(labels))}

    # Fall back to your original behavior
    elif labels is not None:
        marker_color = labels
    elif color_points is not None:
        marker_color = color_points
    else:
        marker_color = "black"

    plt.figure(figsize=figsize)

    if kind == "density":
        # global density across all points
        if _HAS_SCIPY and gaussian_kde is not None:
            dens = _point_density_using_kde(x, y, bw_method=bw_method)
            # normalize
            dens = dens / np.nanmax(dens)
        else:
            dens = _point_density_hist2d(x, y, bins=hist_bins)

        idx = np.argsort(dens)
        x_sorted, y_sorted, dens_sorted = x[idx], y[idx], dens[idx]

        # plot points per label but color mapped by density colormap for visibility
        sc = plt.scatter(
            x_sorted, y_sorted, c=dens_sorted, cmap=density_cmap, s=s, alpha=alpha
        )
        plt.colorbar(sc, label="point density")

    else:  # default normal points
        plt.scatter(x, y, c=marker_color)

    # Add titles and labels
    plt.title(title)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])

    # Show grid (optional)
    if show_grid:
        plt.grid(True)

    # Add legend if labels exist
    if labels is not None and len(labels) > 0:
        for lbl, color in label_to_color.items():
            plt.scatter([], [], color=color, label=lbl)
        plt.legend()

    # Display the plot
    plt.show()

    return plt


def scatter_plot_interactive(
    x,
    y,
    axis_label=["X values", "Y values"],
    title="",
    labels=None,
    color_points=None,
    ID_col=None,
    showfig=True,
    savedir="",
    cluster_colors=None,
):
    """
    Create interactive scatter (go.Scatter) and produce an HTML file that
    downloads selected points as selected_points.csv (Excel-friendly CRLF + BOM).
    Returns: (fig, html_path)
    """
    # Build dataframe
    df = pd.DataFrame({"x": x, "y": y})
    if ID_col is not None:
        df["ID"] = ID_col
    if labels is not None:
        df["label"] = labels
    if labels is None and color_points is not None:
        df["color"] = color_points

    # Keep column order for JS mapping
    col_keys = df.columns.tolist()
    keys_json = json.dumps(col_keys)

    # Prepare customdata as list-of-dicts (guarantees column names survive)
    customdata = df.to_dict("records")

    # If labels AND cluster_colors AND cluster label names are provided:
    if labels is not None and cluster_colors is not None:
        # convert sns (0–1 floats) → hex strings
        #hex_colors = sns_colors_to_hex(cluster_colors)
        
        # Map each label to its corresponding cluster color (if list of colors)
        #label_to_color = {
            #lab: hex_colors[i+1] for i, lab in enumerate(sorted(set(labels)))
        #}
        #marker_color = [label_to_color[l] for l in labels]
        marker_color = [cluster_colors[l] for l in labels]
        marker_color = [rgb_to_hex(c) for c in marker_color] # convert to hex

    # Fall back to your original behavior
    elif labels is not None and cluster_colors is None:
        marker_color = labels
    elif color_points is not None:
        marker_color = color_points
    else:
        marker_color = "black"

    # Build figure with graph objects so customdata is preserved
    trace = go.Scatter(
        x=df["x"],
        y=df["y"],
        mode="markers",
        marker=dict(size=8, color=marker_color),
        customdata=customdata,
        hovertemplate=(
            "x=%{x}<br>y=%{y}"
            + (
                ("<br>ID=%{customdata.ID}" if "ID" in df.columns else "")
                + ("<br>label=%{customdata.label}" if "label" in df.columns else "")
            )
            + "<extra></extra>"
        ),
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(title=title, xaxis_title=axis_label[0], yaxis_title=axis_label[1])

    # Determine output HTML path
    if savedir:
        out_path = Path(savedir)
        if out_path.is_dir() or str(out_path).endswith(("/", "\\")):
            out_path = out_path / f"{title}_scatterplot.html"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path("plot_selection.html")

    # Generate HTML string
    html = pio.to_html(fig, full_html=True, include_plotlyjs="cdn")

    # JS template (placeholder __CUSTOM_KEYS__ will be replaced)
    inject_js_template = r"""
<script>
(function() {
  const CUSTOM_KEYS = __CUSTOM_KEYS__;  // replaced by Python
  const gd = document.querySelectorAll('.js-plotly-plot')[0];
  if(!gd) {
    console.warn('Plot div not found.');
    return;
  }

  function parseCustom(p) {
    if (p.customdata === undefined || p.customdata === null) {
      return { x: p.x, y: p.y };
    }
    if (typeof p.customdata === 'object' && !Array.isArray(p.customdata)) {
      return p.customdata;
    }
    if (Array.isArray(p.customdata)) {
      if (p.customdata.length === 0) return { x: p.x, y: p.y };
      if (typeof p.customdata[0] === 'object') {
        return p.customdata[0];
      }
      const obj = {};
      for (let i = 0; i < p.customdata.length; i++) {
        const key = (i < CUSTOM_KEYS.length) ? CUSTOM_KEYS[i] : ('custom' + i);
        obj[key] = p.customdata[i];
      }
      return obj;
    }
    if (typeof p.customdata === 'string') {
      try {
        const parsed = JSON.parse(p.customdata);
        if (typeof parsed === 'object') return parsed;
      } catch(e) {}
    }
    return { x: p.x, y: p.y };
  }

  function buildUnionKeys(objs) {
    const keys = new Set();
    objs.forEach(o => Object.keys(o).forEach(k => keys.add(k)));
    const preferred = ['x','y','ID','label'];
    const ordered = [];
    preferred.forEach(k => { if (keys.has(k)) { ordered.push(k); keys.delete(k); }});
    keys.forEach(k => ordered.push(k));
    return ordered;
  }

  function objsToCSV(objs) {
    if(!objs || objs.length===0) return '';
    const keys = buildUnionKeys(objs);
    const rows = [keys.join(',')];
    for (const o of objs) {
      const vals = keys.map(k => {
        let v = o[k];
        if (v === null || v === undefined) v = '';
        return '"' + String(v).replace(/"/g,'""') + '"';
      });
      rows.push(vals.join(','));
    }
    // Excel-friendly: UTF-8 BOM + CRLF
    return '\uFEFF' + rows.join('\r\n');
  }

  function downloadFile(filename, text, mime) {
    const blob = new Blob([text], {type: mime || 'text/csv;charset=utf-8;'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  gd.on('plotly_selected', function(eventData) {
    if(!eventData || !eventData.points || eventData.points.length === 0) {
      console.log('No points selected.');
      return;
    }
    const objs = eventData.points.map(p => parseCustom(p));
    const csv = objsToCSV(objs);
    if (csv) downloadFile('selected_points.csv', csv, 'text/csv');
  });

})();
</script>
"""

    # Replace placeholder with actual JSON-encoded keys
    inject_js = inject_js_template.replace("__CUSTOM_KEYS__", keys_json)

    # Inject JS and write HTML
    html = html.replace("</body>", inject_js + "\n</body>")
    out_path.write_text(html, encoding="utf-8")
    html_path = str(out_path.resolve())

    # Open the saved HTML page so the injected JS is present
    if showfig:
        webbrowser.open(out_path.resolve().as_uri())

    return fig, html_path


def plot_scatter_feat_combo(
    dataset,
    feat_list,
    hover=False,
    id_col=None,
    labels=None,
    savedir="",
    cluster_colors=None,
    kind="density",
):
    if id_col in feat_list:
        feat_list.remove(id_col)

    for feature1, feature2 in combinations(feat_list, 2):
        x = dataset[feature1]
        y = dataset[feature2]

        if hover and id_col is not None:
            scatterplt = scatter_plot_interactive(
                x,
                y,
                axis_label=[feature1, feature2],
                title=f"{feature1} vs {feature2}",
                ID_col=dataset[id_col],
                savedir=savedir,
                labels=labels,
                cluster_colors=cluster_colors,
            )
        elif kind == "density":
            scatterplt = scatter_plot(
                x,
                y,
                axis_label=[feature1, feature2],
                title=f"{feature1} vs {feature2}",
                kind="density",
            )

        elif labels is not None and cluster_colors is not None and kind != "density":
            scatterplt = scatter_plot(
                x,
                y,
                axis_label=[feature1, feature2],
                title=f"{feature1} vs {feature2}",
                kind="",
                labels=labels,
                cluster_colors=cluster_colors,
            )

        else:
            scatterplt = scatter_plot(
                x,
                y,
                axis_label=[feature1, feature2],
                title=f"{feature1} vs {feature2}",
                kind="",
            )

    return scatterplt


def show_boxplot(df):
    plt.rcParams["figure.figsize"] = [14, 6]
    sns.boxplot(data=df, orient="v")
    plt.title("Outliers Distribution", fontsize=16)
    plt.ylabel("Range", fontweight="bold")
    plt.xlabel("Attributes", fontweight="bold")

    return


def correlation_matrix(dataframe, thresholds=(0.85, 0.95), plot=False, title=""):
    # run correlation matrix and plot
    corr = dataframe.corr()

    if plot:
        f, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            corr,
            mask=np.zeros_like(corr, dtype=bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True,
            ax=ax,
            xticklabels=False,
            yticklabels=False,
        )

        ax.set_title(title)

    else:
        f = None

    strong_corr_thresh = thresholds[0]
    extremely_strong_corr = thresholds[1]

    thresholded_positive = (corr > strong_corr_thresh) & (corr < extremely_strong_corr)
    rows, cols = thresholded_positive.values.nonzero()
    positive_corr = [
        (corr.index[i], corr.columns[j], corr.iloc[i, j]) for i, j in zip(rows, cols)
    ]

    thresholded_strong_positive = (corr > extremely_strong_corr) & (corr < 0.99)
    rows, cols = thresholded_strong_positive.values.nonzero()
    strong_positive_corr = [
        (corr.index[i], corr.columns[j], corr.iloc[i, j]) for i, j in zip(rows, cols)
    ]

    thresholded_negative = (corr < -strong_corr_thresh) & (
        corr > -extremely_strong_corr
    )
    rows, cols = thresholded_negative.values.nonzero()
    negative_corr = [
        (corr.index[i], corr.columns[j], corr.iloc[i, j]) for i, j in zip(rows, cols)
    ]

    thresholded_strong_negative = (corr < -extremely_strong_corr) & (corr > -0.99)
    rows, cols = thresholded_strong_negative.values.nonzero()
    strong_negative_corr = [
        (corr.index[i], corr.columns[j], corr.iloc[i, j]) for i, j in zip(rows, cols)
    ]

    return f, strong_positive_corr, positive_corr, negative_corr, strong_negative_corr


def correlation_matrix_hover(
    dataframe, thresholds=(0.5, 0.8), plot=True, figsize=(1600, 1400), savedir=""
):
    """
    Compute correlation matrix, return thresholded correlation lists and optionally an interactive Plotly heatmap.
    - thresholds: tuple (strong_corr_thresh, extremely_strong_corr)
    - plot: if True returns a Plotly Figure (and also shows it in notebooks)
    Returns: corr, positive_corr, strong_positive_corr, negative_corr, strong_negative_corr, fig_or_none
    """
    featurenum = dataframe.shape[1]  # number of columns
    # 1) correlation matrix
    corr = dataframe.corr()

    # 2) mask the upper triangle (including diagonal) so we only list each pair once
    mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=0)  # k=0 masks diagonal too
    corr_lower = corr.mask(mask_upper)  # NaN in upper triangle and diagonal

    # 3) threshold lists (only from lower triangle -- avoids duplicates)
    strong_corr_thresh = thresholds[0]
    extremely_strong_corr = thresholds[1]

    # Use .stack() to get pairs (this ignores NaNs automatically)
    stacked = corr_lower.stack()  # MultiIndex series: (row, col) -> value

    # Positive (moderate)
    thresholded_positive = stacked[
        (stacked > strong_corr_thresh) & (stacked < extremely_strong_corr)
    ]
    positive_corr = [(i, j, float(v)) for (i, j), v in thresholded_positive.items()]

    # Positive (extremely strong but less than 0.99 to avoid perfect)
    thresholded_strong_positive = stacked[
        (stacked > extremely_strong_corr) & (stacked < 0.99)
    ]
    strong_positive_corr = [
        (i, j, float(v)) for (i, j), v in thresholded_strong_positive.items()
    ]

    # Negative (moderate)
    thresholded_negative = stacked[
        (stacked < -strong_corr_thresh) & (stacked > -extremely_strong_corr)
    ]
    negative_corr = [(i, j, float(v)) for (i, j), v in thresholded_negative.items()]

    # Negative (extremely strong)
    thresholded_strong_negative = stacked[
        (stacked < -extremely_strong_corr) & (stacked > -0.99)
    ]
    strong_negative_corr = [
        (i, j, float(v)) for (i, j), v in thresholded_strong_negative.items()
    ]

    fig = None
    if plot:
        # 4) prepare matrix for plotting: mask upper to NaN so hover only shows lower triangle
        plot_matrix = corr.copy()
        # plot_matrix.values[mask_upper] = np.nan

        # x = columns, y = index (showing full labels)
        x = list(plot_matrix.columns)
        y = list(plot_matrix.index)
        z = plot_matrix.values

        sns_palette = sns.diverging_palette(220, 10, as_cmap=False, s=90, l=60, n=11)
        # Convert to Plotly format [[position, 'rgb(r,g,b)'], ...]
        colorscale = []
        for i, rgb in enumerate(sns_palette):
            pos = i / (len(sns_palette) - 1)
            colorscale.append([pos, f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"])

        # Create custom hovertemplate: show x, y, and coefficient (z). Format to 3 decimals.
        hovertemplate = (
            "Feature X: %{x}<br>Feature Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                zmin=-1,
                zmax=1,
                colorscale=colorscale,  # diverging colorscale
                colorbar=dict(title="Pearson r"),
                hovertemplate=hovertemplate,
                # showvalues=False  # we rely on hover to show values
            )
        )

        fig.update_layout(
            title=f"Correlation matrix (feature number: {featurenum})",
            width=figsize[0],
            height=figsize[1],
            xaxis=dict(tickmode="array"),
            yaxis=dict(
                autorange="reversed"
            ),  # keep visual orientation similar to seaborn
            template="plotly_white",
        )

        # In notebooks this will display the interactive plot; otherwise return fig to user.
        fig.show()

        if savedir != "":
            fig.write_html(savedir)

    return (
        corr,
        positive_corr,
        strong_positive_corr,
        negative_corr,
        strong_negative_corr,
        fig,
    )


def show_sorted_heatmap(
    dataframe, clustering_algorithm="ward", feature_cluster=True, plot_title=""
):
    cellID = dataframe["Metadata_CellID"]
    dataset = dataframe.drop(columns="Metadata_CellID")
    features_selected = dataset.columns.tolist()

    scaled_dataset = utils.scale_data(dataset)

    median_profile = np.median(scaled_dataset, axis=0)
    median_df = pd.DataFrame(
        [median_profile], columns=features_selected, index=["Median"]
    )

    clustermap = sns.clustermap(
        scaled_dataset, method=clustering_algorithm, metric="euclidean", cbar=False
    )

    # Extract row order from clustermap
    row_order = clustermap.dendrogram_row.reordered_ind
    reordered_data = pd.DataFrame(
        scaled_dataset[row_order],
        columns=features_selected,
        index=np.array(cellID)[row_order],
    )

    fig = plt.figure(figsize=(30, 30))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[10, 0.5])

    ax0 = fig.add_subplot(gs[0])
    sns.heatmap(
        reordered_data,
        cmap="coolwarm",
        vmin=-3,
        vmax=3,
        xticklabels=False,
        yticklabels=True,
        ax=ax0,
        cbar=False,
    )

    ax0.set_title(plot_title, loc="center", pad=20, y=1, fontsize=60)

    ax0.set_xlabel("Cell Features", fontsize=50, labelpad=50)
    ax0.set_ylabel("Cell ID", fontsize=50, rotation=270, labelpad=50)
    ax0.tick_params(axis="x", labelsize=40)
    ax0.tick_params(axis="y", labelsize=25)
    ax0.yaxis.tick_right()
    ax0.yaxis.set_label_position("right")
    ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0)
    for label in ax0.get_yticklabels():
        label.set_horizontalalignment("left")  # or 'center' if preferred

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    sns.heatmap(
        median_df,
        cmap="coolwarm",
        center=0,
        cbar=False,
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=True,
        ax=ax1,
    )

    ax1.set_yticklabels(["Median"], rotation=0, fontsize=50)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    ax1.set_yticklabels(["Median"], rotation=0, fontsize=50)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    cbar = fig.colorbar(
        ax0.collections[0],
        ax=[ax0, ax1],
        orientation="vertical",
        fraction=0.02,
        pad=0.04,
    )
    cbar.ax.set_position([0.07, 0.25, 0.03, 0.6])  # [top, bottom, left, right]
    cbar.set_ticks([-3, 0, 3])
    cbar.set_ticklabels([-3, 0, 3])

    cbar.ax.tick_params(labelsize=50)

    return plt


def subpopulation_median_profiles(
    dataframe, clustering_algorithm="ward", subpopulation_number=1, plot_title=""
):
    dataframe_labeled = clus.add_cluster_label(
        dataframe, subpopulation_number, clustering_algorithm=clustering_algorithm
    )
    # Group by cluster and calculate median profile
    df_median_profiles = dataframe_labeled.groupby(
        clustering_algorithm + " Cluster"
    ).median()

    # Put the scaled data back into a DataFrame for plotting
    df_scaled_profiles = pd.DataFrame(
        df_median_profiles,
        index=df_median_profiles.index,
        columns=df_median_profiles.columns,
    )

    # Plot heatmap
    plt.figure(
        figsize=(
            len(df_scaled_profiles.columns) * 0.25,
            len(df_scaled_profiles.index) * 0.8,
        )
    )
    sns.heatmap(
        df_scaled_profiles, cmap="coolwarm", annot=False, linewidths=0.5, center=0
    )
    plt.title(plot_title)
    plt.xlabel("Cell Features")
    plt.ylabel("Subpopulation")


def show_clustermap(
    dataframe_labelled,
    clustering_algorithm="ward",
    feature_cluster=True,
    plot_title="",
    color_feat=None,
    pltsize=(90, 30),
    color_row=True,
    id_col=constants.CELLID_COL,
    label_col=None,
    color_row_cmap=None,
    scale = True,
    score_range = None,
    cmap="coolwarm",
    all_legends = False,
):
    # clustering algorithms: complete, ward, average, single

    if label_col is None:  # use last column as label column if none provided
        label_col = dataframe_labelled.columns[-1]

    cells = dataframe_labelled[constants.CELLID_COL]
    dataset = dataframe_labelled[
        [col for col in dataframe_labelled.columns if col not in [id_col, label_col]]
    ]

    if scale:
        dataset_scaled = utils.scale_data(dataset)
        vmin = -3
        vmax = 3
    else:
        dataset_scaled = dataset.copy()
        vmin = 0.5 * dataset_scaled.min().min()
        vmax = 0.5 * dataset_scaled.max().max()

    if score_range is not None:
        vmin, vmax = score_range

    features = np.array(dataset.columns.values.tolist())

    subpopulations = dataframe_labelled[label_col]
    if color_row_cmap is None:
        _, _, color_row_cmap = clus.label_color_mapping_dict(subpopulations, palette="inferno")

    row_colors = subpopulations.map(color_row_cmap).to_numpy()

    if not color_row:
        row_colors = None

    # color features by organelle
    if color_feat == "organelle":
        keyword_colors = constants.ORGANELLE_CP_CMAP
        col_colors = col_color_list(dataset.columns.tolist(), constants.ORGANELLE_CP_CMAP)
        handles_col = [
            Patch(facecolor=keyword_colors[organelle], label=f"{organelle}")
            for organelle in keyword_colors.keys()
        ]

    # color features by feature type
    elif color_feat == "type":
        keyword_colors = constants.FEATURE_TYPE_CMAP
        col_colors = col_color_list(
            dataset.columns.tolist(), constants.FEATURE_TYPE_CMAP
        )

        handles_col = [
            Patch(facecolor=keyword_colors[feat_type], label=f"{feat_type}")
            for feat_type in keyword_colors.keys()
        ]
        
    else:
        col_colors = None
        keyword_colors = {"": "black"}

    if feature_cluster:
        dendrogram_ratio = (0.1, 0.2)
    else:
        dendrogram_ratio = (0.1, 0.02)   # or even (0.1, 0.0)
    number_xaxis = False
    if len(dataset) >= 2 and len(dataset.columns) > 60:
        # Plot row colors
        plot_comp = sns.clustermap(
            dataset_scaled.to_numpy(),
            method=clustering_algorithm,
            cmap=cmap,
            row_colors=row_colors,
            colors_ratio=(0.02, 0.02),  # [row colorbar, column colorbar]
            dendrogram_ratio=dendrogram_ratio,  # [row dendrogram, column dendrogram]
            figsize=pltsize,
            yticklabels=cells,
            col_cluster=feature_cluster,
            vmin=vmin,
            vmax=vmax,
            metric="euclidean",
            tree_kws=dict(linewidths=5),
            col_colors=col_colors,
        )

        number_xaxis = False

    elif len(dataset) >= 2 and len(dataset.columns) <= 60:
        # Plot row colors
        plot_comp = sns.clustermap(
            dataset_scaled.to_numpy(),
            method=clustering_algorithm,
            cmap=cmap,
            row_colors=row_colors,
            colors_ratio=(0.02, 0.02),  # [row colorbar, column colorbar]
            dendrogram_ratio=dendrogram_ratio,  # [row dendrogram, column dendrogram]
            figsize=pltsize,
            yticklabels=cells,
            xticklabels=features,
            col_cluster=feature_cluster,
            vmin=vmin,
            vmax=vmax,
            metric="euclidean",
            tree_kws=dict(linewidths=5),
            col_colors=col_colors,
        )
    else:
        plot_comp = sns.clustermap(
            dataset_scaled.to_numpy(),
            method=clustering_algorithm,
            cmap=cmap,
            row_colors=row_colors,
            colors_ratio=(0.02, 0.02),  # [row colorbar, column colorbar]
            dendrogram_ratio=dendrogram_ratio,  # [row dendrogram, column dendrogram]
            figsize=pltsize,
            yticklabels=cells,
            row_cluster=False,
            col_cluster=feature_cluster,
            vmin=vmin,
            vmax=vmax,
            metric="euclidean",
            tree_kws=dict(linewidths=5),
            col_colors=col_colors,
        )

    handles = [
        Patch(facecolor=color_row_cmap[label], label=f"Subpopulation {label}")
        for label in color_row_cmap.keys()
    ]

    if all_legends:
        # add legend to the figure
        pop_legend = plot_comp.ax_heatmap.legend(
            handles=handles,
            title="Subpopulations",
            loc="upper right",
            bbox_to_anchor=(1.25, 1),  # move legend outside heatmap
            fontsize=50,
            title_fontsize=60,
        )
    
        plot_comp.ax_heatmap.add_artist(pop_legend)
    
        # add second legend for feature type
        plot_comp.ax_heatmap.legend(
            handles=handles_col,
            title=f"Feature type ({color_feat})",
            loc="lower right",
            bbox_to_anchor=(1.25, 0.4),  # move legend outside heatmap
            fontsize=50,
            title_fontsize=50,
        )
    

    #plot_comp.ax_heatmap.set_xlabel("Cell Features", fontsize=50, labelpad=50)
    #plot_comp.ax_heatmap.set_ylabel("Cell ID", fontsize=50, rotation=270, labelpad=50)
    plot_comp.ax_heatmap.tick_params(axis="y", labelsize=30)
    plot_comp.ax_heatmap.tick_params(axis="x", labelsize=20)
    plot_comp.ax_heatmap.yaxis.tick_right()
    plot_comp.ax_heatmap.yaxis.set_label_position("right")
    plot_comp.ax_heatmap.set_yticklabels(
        plot_comp.ax_heatmap.get_yticklabels(), rotation=0
    )

    plot_comp.ax_heatmap.set_xticks([])  # remove tick marks
    plot_comp.ax_heatmap.set_xticklabels([])  # remove tick labels
    
    # remove y axis labels (CellID)
    plot_comp.ax_heatmap.set_yticks([])  # remove tick marks
    plot_comp.ax_heatmap.set_yticklabels([])  # remove tick labels

    if number_xaxis:
        for label in plot_comp.ax_heatmap.get_xticklabels():
            label.set_color(
                "black"
            )  # set default color if feature not in color dictionary
            for o in keyword_colors.keys():
                feature = features[int(label.get_text())]
                if o in feature:
                    label.set_color(keyword_colors[o])
    else:
        for label in plot_comp.ax_heatmap.get_xticklabels():
            label.set_color(
                "black"
            )  # set default color if feature not in color dictionary
            for o in keyword_colors.keys():
                if o in str(label):
                    label.set_color(keyword_colors[o])

    for label in plot_comp.ax_heatmap.get_yticklabels():
        label.set_horizontalalignment("left")

    if plot_title:
        y_pos = 1.3 if feature_cluster else 1
        plot_comp.ax_heatmap.set_title(
            plot_title, loc="center", pad=20, y=y_pos, fontsize=60
        )

    plot_comp.cax.set_position([1.02, 0.4, 0.02, 0.4])  # [left, bottom, width, height]
    plot_comp.cax.tick_params(labelsize=60)  # [left, bottom, width, height]
    plot_comp.ax_heatmap.set_xticklabels(
        plot_comp.ax_heatmap.get_xticklabels(),
        fontsize=60,
    )

    # plot_comp.show()

    return plot_comp, color_row_cmap


def pie_chart(population_list, legend=False, title=""):
    # Count occurrences
    counts = population_list.value_counts()

    cluster_num = population_list.nunique()

    labels = sorted(list(population_list.unique()))
    counts = counts.reindex(labels)
    label_names = legend if legend else [f"Cluster {l}" for l in labels]

    colors = sns.color_palette("hls", cluster_num)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot pie WITHOUT labels
    wedges, texts, autotexts = ax.pie(
        counts.values,
        colors=colors,  # same colors as PCA
        autopct="%1.1f%%",
        labels=None,  # important
    )

    for autotext in autotexts:
        autotext.set_fontsize(18)

    # Add legend using your PCA label_names
    ax.legend(
        wedges,
        label_names,
        title="Subtypes",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    plt.title(title)
    plt.show()

    return


def PCA_clustering_silhouette(
    dataset_plot, title, cluster_num, cluster_method, n_PCA_components=3
):
    legendlist = []
    for i in list(range(1, cluster_num + 1)):
        legendlist.append(f"Subtype {i}")

    (
        fig,
        principal_df,
        pca,
        loadings_matrix,
        combined_df,
        colors,
        label_names,
        label_color_mapping,
    ) = clus.PCA_components(
        dataset_plot,
        n_PCA_components=n_PCA_components,
        cluster_num=cluster_num,
        legend=legendlist,
        plot=True,
        title=title,
        cluster_method=cluster_method,
        id_col="Metadata_CellID",
    )

    # compare silhouette scores
    labels = combined_df[combined_df.columns[-1]]
    individual_scores_df = clus.silhouette_score_indiv(
        principal_df, labels, cellid_col=None, scale=False
    )
    clus.silhouette_plot(
        individual_scores_df["Silhouette_score"],
        individual_scores_df["Population"],
        title=title,
    )

    # pie chart of population distribution
    pie_chart(labels, legend=legendlist, title="Distribution of Subtypes")

    return combined_df, colors, label_names, label_color_mapping


def violin_by_keyword(
    df,
    text_col,
    value_col,
    group_map,
    fallback_group="Other",
    include_fallback=True,
    order=None,
    palette=None,        # seaborn color spec (dict, list, name)
    show_points="swarm",  # None, "swarm", or "strip"
    figsize=(5, 5),
    violin_kwargs=None,
    point_kwargs=None
):
    """
    df: pandas DataFrame
    text_col: column containing strings to search for keywords
    value_col: numeric column to plot
    group_map: dict mapping keyword -> group_label (priority = dict order)
               e.g. {'mitochondria': 'Mito', 'nucleus': 'Nuc'}
               If you want many keywords to map to same group, put them all in keys.
    fallback_group: group name for rows that match none of the keywords
    order: explicit order of groups for plotting (list); if None, inferred from group_map + fallback
    palette: colors; can be a dict mapping group->color or a seaborn palette name
    show_points: overlay points, "swarm" (better) or "strip" or None
    violin_kwargs: dict of kwargs passed to sns.violinplot
    point_kwargs: dict of kwargs passed to sns.swarmplot/stripplot
    """

    violin_kwargs = {} if violin_kwargs is None else dict(violin_kwargs)
    point_kwargs = {} if point_kwargs is None else dict(point_kwargs)

    # Build a single regex that captures keywords in priority order
    # Use alternation group with word boundaries where appropriate
    # We'll also remember a mapping from matched substring -> group label
    pattern_parts = []
    map_lower = {}
    for keyword, group_label in group_map.items():
        # escape keyword for regex; keep it case-insensitive
        k_esc = re.escape(keyword)
        pattern_parts.append(k_esc)
        map_lower[keyword.lower()] = group_label

    if pattern_parts:
        pattern = re.compile("(" + "|".join(pattern_parts) + ")", flags=re.IGNORECASE)
    else:
        pattern = None

    def find_group(text):
        if pattern is None:
            return fallback_group
        m = pattern.search("" if pd.isna(text) else str(text))
        if not m:
            return fallback_group
        matched = m.group(0).lower()
        # map matched text (lower) back to group label
        # We stored original keywords lower-cased in map_lower; however if multiple original keywords
        # differ only by case, this still works because we used escape of original keywords and
        # map_lower uses .lower()
        return map_lower.get(matched, fallback_group)

    # assign groups (vectorized)
    df = df.copy()
    df["_group"] = df[text_col].apply(find_group)
    
    if not include_fallback:
        df = df[df["_group"] != fallback_group]
    
    # compute plot order
    if order is None:
        # preserve the order of group_map values, then fallback last (if present)
        group_vals = []
        seen = set()
        for k in group_map:
            g = group_map[k]
            if g not in seen:
                group_vals.append(g); seen.add(g)
        if fallback_group not in seen:
            group_vals.append(fallback_group)
        order = group_vals

    if not include_fallback and fallback_group in order:
        order = [g for g in order if g != fallback_group]

    # optionally build palette dict if user gave e.g. FEATURE_TYPE_CMAP where keys are groups
    # if palette is a dict mapping group->color, use it; otherwise pass palette to seaborn
    palette_to_pass = palette
    if isinstance(palette, dict):
        # ensure palette includes all groups in `order`, fallback to a default color if missing
        palette_to_pass = {g: palette.get(g, "#333333") for g in order}

    # Plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df,
        x="_group",
        y=value_col,
        order=order,
        palette=palette_to_pass,
        hue = "_group",
        legend=False,
        ax=ax,
        **violin_kwargs
    )

    # optionally overlay points
    if show_points == "swarm":
        sns.swarmplot(
            data=df,
            x="_group",
            y=value_col,
            order=order,
            color="k",
            alpha=0.6,
            ax=ax,
            **point_kwargs
        )
    elif show_points == "strip":
        sns.stripplot(
            data=df,
            x="_group",
            y=value_col,
            order=order,
            color="k",
            alpha=0.6,
            ax=ax,
            **point_kwargs
        )

    ax.set_xlabel("")
    ax.set_ylabel(value_col)
    plt.tight_layout()
    return fig, ax

def subpopulation_violplt(
    dataframe, pval_matrix, cluster_col, feature, cmap=None, id_label=False, id_col=None, axes = True
):
    populations = list(np.unique(dataframe[cluster_col]))

    remove = np.tril(np.ones(pval_matrix.shape), k=0).astype("bool")
    pval_matrix[remove] = np.nan

    molten_df = pval_matrix.melt(ignore_index=False).reset_index().dropna()

    if cmap is None:
        cmap = clus.label_color_mapping_dict(dataframe[cluster_col], palette="hls")

    fig, ax = plt.subplots()
    sns.violinplot(
        data=dataframe,
        x=cluster_col,
        y=feature,
        order=populations,
        palette=cmap,
        hue=cluster_col,
        legend=False,
        ax=ax,

    )
    
    if id_label and id_col is not None:
        sns.stripplot(data=dataframe, x=cluster_col, y=feature, color="white", ax=ax)

        for _, row in dataframe.iterrows():
            ax.text(
                x=populations.index(row[cluster_col]),
                y=row[feature],
                s=row[id_col],
                fontsize=6,
                ha="center",
                va="center",
            )

    # feature_split = feature.split("_")
    # Overlay the individual points
    sns.stripplot(
        data=dataframe,
        x=cluster_col,
        y=feature,
        dodge=True,  # separate points by hue
        alpha=0.6,  # transparency for visibility
        size=3,  # smaller points
        color="black",
    )

    if not axes:
        ax.set_ylabel("")
        ax.set_xlabel("")
    else:
        ax.set_ylabel(f"{feature}", fontsize=8)
        ax.set_xlabel("Subpopulation")


    plt.legend([], [], frameon=False)  # optional: hide duplicate legend

    pairs = [(i[1]["index"], i[1]["variable"]) for i in molten_df.iterrows()]
    p_values = [i[1]["value"] for i in molten_df.iterrows()]

    annotator = Annotator(
        ax, pairs, data=dataframe, x=cluster_col, y=feature, order=populations
    )
    annotator.configure(text_format="star", loc="inside")
    annotator.set_pvalues_and_annotate(p_values)

    plt.tight_layout()

    subpopulation_cells = {}
    for i in list(range(1, len(populations) + 1)):  # start subpopulation count at 1
        subpopulation_cells["Supopulation_" + str(i)] = dataframe.loc[
            dataframe[cluster_col] == i, constants.CELLID_COL
        ].tolist()

    return subpopulation_cells


def cluster_bxplt(
    dataframe,
    clusters="",
    feature_idx=0,
    savedir="",
    significance=False,
    stat_test="Mann-Whitney",
    sort_clusters=False,
):
    """
    plots features based on their cluster in boxplot

    input:
        dataframe: original data (not normalized) in DataFrame with column names and assigned cluster column
        clusters (str): column name with cluster labels (default is last column)
        feature_idx: column with feature to plot
        savedir: output directory to save images to (if blank will not save any image)
        significance: performs and plots results of test for significance between groups
        stat_test: specify what stat test to perform on clusters (default: Mann-Whitney)
        sort_clusters: sort the clusters if numerical from smallest to largest (default: False)

    output:
        boxplot of feature distribution of each cluster


    """

    Path(savedir).mkdir(parents=True, exist_ok=True)
    header = dataframe.columns.values.tolist()  # Get column headings

    if clusters == "":
        clusters = header[-1]

    unique_colvals = dataframe.nunique()
    clusternum = unique_colvals[clusters]

    bxplt = sns.boxplot(
        x=clusters,
        y=header[feature_idx],
        data=dataframe,
        palette="colorblind",
        medianprops=dict(color="red", alpha=0.7),
    )
    bxplt.set_xticklabels(dataframe[clusters].unique())

    if significance == True:  # plot significance
        ls = list(range(clusternum))
        combinations = [
            (ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))
        ]
        print(combinations)

        annotator = Annotator(
            bxplt, combinations, data=dataframe, x=clusters, y=header[feature_idx]
        )
        annotator.configure(test=stat_test, text_format="star")
        annotator.apply_and_annotate()

    if savedir == "":
        print("Boxplot not saved as image")
    else:
        image_format = "svg"
        image_name = (
            savedir + "bxplt_clusters_" + clusters + "_" + header[feature_idx] + ".svg"
        )
        plt.savefig(image_name, format=image_format, dpi=1200)
        print("Boxplot saved to " + savedir)

    plt.show()


def stacked_bxplt_count(
    dataframe, group_col, data_col, label_color_mapping=None, normalize=False, legend=True, axis_labels = True
):
    unique_labels = dataframe[data_col].unique()
    labels_sorted = list(np.sort(unique_labels))  # keep label order consistent

    counts = dataframe.groupby([group_col, data_col]).size().unstack(fill_value=0)

    if label_color_mapping is not None:
        color_list = [
            label_color_mapping.get(col, (0.7, 0.7, 0.7)) for col in counts.columns
        ]
    else:
        color_list = None

    # Ensure columns are ordered in ascending order
    counts = counts[labels_sorted]

    if normalize:
        # avoid division by zero: if a row sums to zero, leave as zeros
        row_sums = counts.sum(axis=1).replace({0: np.nan})
        proportions = counts.div(row_sums, axis=0).fillna(0)
        plot_df = proportions
        ylabel = "Proportion"
    else:
        plot_df = counts
        ylabel = "Count"

    if legend:
        ax = plot_df.plot(kind="bar", stacked=True, color=color_list, figsize=(10, 6))
    else:
        ax = plot_df.plot(kind="bar", stacked=True, color=color_list, figsize=(10, 6), legend=False)
    
    if axis_labels:
        ax.set_xlabel("Cell ID")
        ax.set_ylabel(ylabel)
        ax.set_title("Subtype Composition per Cell")
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])

    if legend: 
        if label_color_mapping is not None:
            legend_handles = [
                Patch(facecolor=color_list[i], label=f"Subtype {counts.columns[i]}")
                for i in range(len(counts.columns))
            ]
            ax.legend(handles=legend_handles, title="Subtype")

    plt.tight_layout()
    plt.show()

    return


def scatter_density_3D(x, y, z, labels=("X", "Y", "Z")):
    points = np.vstack([x, y, z]).T

    # Fit kernel density model
    kde = KernelDensity(bandwidth=0.5).fit(points)
    dens = np.exp(kde.score_samples(points))

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    img = ax.scatter(x, y, z, c=dens, cmap="viridis", s=15)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    fig.colorbar(img, ax=ax, label="Density")
    plt.show()

    return plt

def create_table(df, body_fontsize=5, header_fontsize=5, fig_size=(5, 2.5), title=""):

    '''
    df is a pandas DataFrame with columns = table columns and rows = table rows. The function creates a styled table figure with journal-style formatting and booktabs-like horizontal rules.
    '''

    # -----------------------
    # 2. Create Figure
    # -----------------------
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    # -----------------------
    # 3. Style the Table
    # -----------------------

    table.auto_set_font_size(False)
    #table.set_fontsize(9)

    # Remove all cell borders (journal style)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = 0.02

        if row == 0:
            # Header row
            cell.set_text_props(weight='bold', fontsize=header_fontsize)
        else:
            # Body rows
            cell.set_text_props(fontsize=body_fontsize)

    # Make header bold
    for col in range(len(df.columns)):
        table[(0, col)].set_text_props(weight='bold')


    # Left align first column (Group)
    #for row in range(1, len(df) + 1):
        #table[(row, 0)].get_text().set_ha('left')

    # -----------------------
    # 4. Add Horizontal Rules (Booktabs Style)
    # -----------------------
    plt.draw()

    # Get table bounding box
    bbox = table.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transAxes.inverted()
    bbox = inv.transform_bbox(bbox)

    x0, x1 = bbox.x0, bbox.x1
    y_top = bbox.y1
    y_bottom = bbox.y0

    # Header bottom line
    header_height = table[(0, 0)].get_window_extent(fig.canvas.get_renderer())
    header_height = inv.transform_bbox(header_height)
    y_header_bottom = header_height.y0

    # Draw lines
    ax.plot([x0, x1], [y_top, y_top], linewidth=1, transform=ax.transAxes, color = "black")
    ax.plot([x0, x1], [y_header_bottom, y_header_bottom], linewidth=0.7, transform=ax.transAxes,  color = "black")
    ax.plot([x0, x1], [y_bottom, y_bottom], linewidth=1, transform=ax.transAxes,  color = "black")

    plt.title(title, fontsize=header_fontsize + 2, weight='bold', pad=20)
    plt.tight_layout()

    return