import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def _frame_to_index(frame_key, frame_order=None):
    """
    Convert a frame key (like 't010' or integer) to numeric frame index.
    If frame_order is provided (list of ordered keys), returns the index in that list.
    If frame_key already numeric, returns it as int.
    If frame_key is a string of digits, converts to int.
    """
    if frame_order is not None:
        # safe: if frame_key not in frame_order, raise KeyError
        return int(frame_order.index(frame_key))
    # else try numeric conversion
    if isinstance(frame_key, (int, np.integer)):
        return int(frame_key)
    s = str(frame_key)
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits != "":
        return int(digits)
    raise ValueError(f"Cannot convert frame key {frame_key!r} to numeric index. Provide frame_order.")

def lifetimes_dict_to_df(lifetimes, frame_order=None, fps=None):
    """
    Convert lifetimes dict -> DataFrame with columns:
      label, start_frame (index), end_frame (index), duration_frames, start_time (sec), end_time (sec)
    lifetimes: { label: {'first_frame': t_first, 'last_frame': t_last, 'frames_present': n, ... }, ...}
    frame_order: optional list defining chronological order of keys
    fps: optional frames-per-second for time conversion
    """
    rows = []
    for lab, info in lifetimes.items():
        first = info.get("first_frame")
        last = info.get("last_frame")
        frames_list = info.get("frames_list", None)
        # fallback: if last_frame missing, infer from frames_list
        if first is None and frames_list:
            first = frames_list[0]
        if last is None and frames_list:
            last = frames_list[-1]
        if first is None or last is None:
            continue
        try:
            s_idx = _frame_to_index(first, frame_order=frame_order)
            e_idx = _frame_to_index(last, frame_order=frame_order)
        except Exception:
            # if conversion fails, skip this label
            continue
        duration = e_idx - s_idx + 1  # inclusive count of frames
        row = {
            "label": int(lab),
            "start_frame": s_idx,
            "end_frame": e_idx,
            "duration_frames": duration
        }
        if fps is not None and fps > 0:
            row["start_time_s"] = s_idx / float(fps)
            row["end_time_s"] = e_idx / float(fps)
            row["duration_s"] = duration / float(fps)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def plot_lifetimes_gantt(
    lifetimes,
    frame_order=None,
    fps=None,
    max_labels=200,
    sort_by="start",   # "start" or "duration"
    figsize=(8,5),
    cmap="viridis",
    show_label_ticks=False,
    title="Label lifetimes (Gantt-like)",
    xlabel="Frame"   # or "Time (s)" if fps given
):
    """
    Create a Gantt-like horizontal bar chart from lifetimes dict.

    Parameters
    ----------
    lifetimes : dict
        output of quantify_label_lifetimes
    frame_order : list or None
        ordered list of time keys (if your first/last_frame are strings). If None, numeric conversion is attempted.
    fps : float or None
        optional frames-per-second to convert frames -> seconds; if provided, xlabel will be "Time (s)".
    max_labels : int
        if you have many labels, limit plotted rows; top rows selected by sort_by.
    sort_by : "start" or "duration"
        how to order y-axis rows.
    show_label_ticks : bool
        whether to show label ids on y-axis (may be unreadable if many).
    """
    df = lifetimes_dict_to_df(lifetimes, frame_order=frame_order, fps=fps)
    if df.empty:
        raise ValueError("No valid lifetimes entries found in lifetimes dict.")

    # sort and optionally limit rows
    if sort_by == "duration":
        df = df.sort_values("duration_frames", ascending=False)
    else:
        df = df.sort_values("start_frame", ascending=True)

    if len(df) > max_labels:
        df_plot = df.iloc[:max_labels].copy()
    else:
        df_plot = df.copy()

    # prepare plotting coordinates
    y_positions = np.arange(len(df_plot))
    starts = df_plot["start_frame"].values.astype(float)
    durations = df_plot["duration_frames"].values.astype(float)

    # x axis in seconds if fps provided
    if fps is not None and fps > 0:
        starts_plot = starts / float(fps)
        widths_plot = durations / float(fps)
        xlabel_plot = "Time (s)"
    else:
        starts_plot = starts
        widths_plot = durations
        xlabel_plot = xlabel

    # color by duration (normalized)
    norm = mpl.colors.Normalize(vmin=df_plot["duration_frames"].min(), vmax=df_plot["duration_frames"].max())
    cmap = mpl.cm.get_cmap(cmap)
    colors = cmap(norm(df_plot["duration_frames"].values))

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y_positions, widths_plot, left=starts_plot, height=0.8, color=colors, edgecolor="k", linewidth=0.3)

    # y ticks -> label ids if requested, else blank
    if show_label_ticks:
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df_plot["label"].astype(str))
    else:
        ax.set_yticks([])

    ax.set_xlabel("")
    ax.set_ylabel("") 
    #ax.set_title(title)
    
    # colorbar for durations
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    #cbar.set_label("Duration (minutes)")

    plt.tight_layout()
    return fig, ax, df_plot  # return df_plot in case you want to inspect what was plotted


# plot SEM

def agg_pct_change(df,
                   value_col,
                   time_col="Timepoint_min",
                   id_col_candidates=None,
                   normalize = True, 
                   include_zeros = True):
    """
    Compute per-cell percent change then aggregate across cells.
    Returns DataFrame with columns: [time_col, mean_pct, min_pct, max_pct, sem_pct].
    """
    df = df.copy()
    if id_col_candidates is None:
        id_col_candidates = ['Cell', 'Cell_ID', 'cell_id', 'CellID', 'cellID', 'id', 'track_id', 'trackID']

    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    if id_col is None:
        df['__pseudo_cell'] = np.arange(len(df))
        id_col = '__pseudo_cell'

    df = df.sort_values([id_col, time_col])
    
    if normalize:
        # compute baseline per cell
        baseline_map = {}
        for cell, g in df.groupby(id_col):
            t0_vals = g.loc[g[time_col] == 0, value_col]
            if len(t0_vals) > 0:
                baseline = t0_vals.mean()
            else:
                baseline = g.iloc[0][value_col]
            baseline_map[cell] = baseline if baseline != 0 else np.nan  # avoid zero baseline
    
        df['baseline'] = df[id_col].map(baseline_map)
    
        # percent change (cells with baseline NaN will produce NaN)
        nonzero = ~df['baseline'].isna()
        df['pct_change'] = np.nan
        df.loc[nonzero, 'pct_change'] = (df.loc[nonzero, value_col] - df.loc[nonzero, 'baseline']) / df.loc[nonzero, 'baseline'] * 100  
        
        agg_col = 'pct_change'

    else:
        df['value_for_agg'] = df[value_col]
        agg_col = 'value_for_agg'
    
    if include_zeros:
        # aggregate
        agg = df.groupby(time_col)[agg_col].agg(['mean', 'min', 'max', 'sem']).reset_index().rename(
            columns={'mean': 'mean_pct', 'min': 'min_pct', 'max': 'max_pct', 'sem': 'sem_pct'}
        )
    else:
        agg = (
        df.assign(**{agg_col: df[agg_col].replace(0, np.nan)})
          .groupby(time_col)[agg_col]
          .agg(['mean', 'min', 'max', 'sem'])
          .reset_index()
          .rename(columns={
              'mean': 'mean_pct',
              'min': 'min_pct',
              'max': 'max_pct',
              'sem': 'sem_pct'
          })
    )

    return agg

def plot_groups_with_band(lines_uninfected,
                          lines_infected,
                          cell_properties_uninfected,
                          cell_properties_infected,
                          feature_plot,
                          band="range",          # "range" or "sem"
                          percentile=(10,90),   # future-proofing if you want percentiles
                          cmap_map=None,
                          figsize=(8,5),
                          legend_outside=False,
                          labels = True,
                          normalize = True,
                          all_conditions = False,
                          include_zeros = True,
                          ylimit = 100):
    """
    band: "range" (min->max) or "sem" (mean +/- sem)
    cmap_map: dict mapping line -> color (e.g., constants.CMAP). If None, matplotlib default cycle used.
    """
    plt.figure(figsize=figsize)
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_line(agg, label, linestyle, color):
        plt.plot(agg['Timepoint_min'], agg['mean_pct'], linewidth=1.5, linestyle=linestyle, label=label, color=color)
        if band == "range":
            lower = agg['min_pct']
            upper = agg['max_pct']
        elif band == "sem":
            lower = agg['mean_pct'] - agg['sem_pct']
            upper = agg['mean_pct'] + agg['sem_pct']
        else:
            raise ValueError("band must be 'range' or 'sem'")

        plt.fill_between(agg['Timepoint_min'], lower, upper, alpha=0.2, color=color, linewidth=0)
        
    if all_conditions:
    # Uninfected (dashed)
        for i, line in enumerate(lines_uninfected):
            df_line = cell_properties_uninfected[line]
            agg = agg_pct_change(df_line, value_col=feature_plot, time_col="Timepoint_min", normalize = normalize, include_zeros = include_zeros)
            if cmap_map and line in cmap_map:
                color = cmap_map[line]
            else:
                color = prop_cycle[i % len(prop_cycle)]
            plot_line(agg, f"{line} (uninfected)", linestyle='--', color=color)

    # Infected (solid)
    offset = len(lines_uninfected)
    for j, line in enumerate(lines_infected):
        df_line = cell_properties_infected[line]
        agg = agg_pct_change(df_line, value_col=feature_plot, time_col="Timepoint_min", normalize = normalize, include_zeros = include_zeros)
        if cmap_map and line in cmap_map:
            color = cmap_map[line]
        else:
            color = prop_cycle[(j + offset) % len(prop_cycle)]
        plot_line(agg, f"{line} (infected)", linestyle='-', color=color)
    
    if labels:
        plt.xlabel("Timepoint (min)")
        plt.ylabel(f"Percent change in {feature_plot} from baseline (%)")
        plt.title(f"Percent change from baseline ({feature_plot}) — per-cell normalized, then averaged")
    else:
        plt.xlabel("")
        plt.ylabel("")
        plt.title("")

    if legend_outside:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        plt.legend()
        
    plt.ylim(top = ylimit)
    plt.show()

def line_plot_over_time(df, feature_plot, time_col, group_col, selected_line = None, 
                        plot_all = False, separate_plots = False, 
                        full_tracks = False, plot_one = False):
    import constants
    
    ylimit = np.max(df[feature_plot])*1.5
    if plot_all and not separate_plots:
        plt.figure()
        
        # Plot each cell with transparency
        for cell_id, group in df.groupby("CellID"):
            group = group.sort_values("Timepoint_min")
            plt.plot(group["Timepoint_min"], group[feature_plot], 
                     alpha=0.3, linewidth=1)
        
        # Add mean line
        mean_df = (
            df.groupby("Timepoint_min")[feature_plot]
            .mean()
            .reset_index()
        )
        
        plt.plot(mean_df["Timepoint_min"], mean_df[feature_plot], 
                 linewidth=1)
        
        plt.xlabel("Timepoint (min)")
        plt.ylabel(f"{feature_plot}")
        plt.title(f"{feature_plot} over Time")
        plt.show()
    
    elif plot_all and separate_plots:
        
        for cell_id, group in df.groupby("CellID"):
            plt.figure()

            group = group.sort_values("Timepoint_min")
            
            if plot_one and cell_id == plot_one:
                
                plt.plot(group["Timepoint_min"], group[feature_plot], 
                         alpha=0.3, linewidth=3, color = constants.CMAP[selected_line])
                
                plt.ylim(top = ylimit)
                '''
                plt.xlabel("Timepoint (min)")
                plt.ylabel(f"{feature_plot}")
                plt.title(f"{feature_plot} {cell_id} over Time")
                '''
                plt.show()
                
            elif plot_one and cell_id != plot_one:
                continue
            
            else:
                plt.plot(group["Timepoint_min"], group[feature_plot], 
                         alpha=0.3, linewidth=3, color = constants.CMAP[selected_line])
                
                plt.ylim(top = ylimit)
                plt.xlabel("Timepoint (min)")
                plt.ylabel(f"{feature_plot}")
                plt.title(f"{feature_plot} {cell_id} over Time")
                plt.show()
        
        
    else:
        
        if full_tracks:
            all_timepoints = sorted(df[time_col].unique())
    
            pivot = df.pivot(index=group_col, columns=time_col, values=feature_plot)
            pivot = pivot.reindex(columns=all_timepoints)
            
            pivot_complete = pivot.dropna()
            
            mean_df = pivot_complete.mean().reset_index()
            mean_df.columns = [time_col, feature_plot]
        
        else: 
            mean_df = (
                df.groupby("Timepoint_min")[feature_plot]
                .mean()
                .reset_index()
            )
            
        plt.figure()
        plt.plot(mean_df["Timepoint_min"], mean_df[feature_plot], 
                 linewidth=1)
        
        plt.xlabel("Timepoint (min)")
        plt.ylabel(f"Mean {feature_plot}")
        plt.title(f"Mean {feature_plot} over Time")
        plt.show()
