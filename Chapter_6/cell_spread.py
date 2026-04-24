from pathlib import Path
import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import utils
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment
import constants
import cv2
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops_table
from tqdm import tqdm
#%% IO
lines_ordered = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]

parentdir = Path(r"C:/Users/IvyWork/Desktop/projects/dataset4D/3D_Overview")

prob_dir = parentdir / "ilastik_probtiffs_2D"
prob_df = utils.load_path_into_df(prob_dir, keywordregex=[r"Series\d{3}"], keyword="MAX_2D")

bin_dir = parentdir / "binary_volumes" / "PXN_binvol_2D"  
bin_dir.mkdir(parents=True, exist_ok=True)

#%% create binary masks

for row in prob_df.itertuples(index = False):
    probmap = tiff.imread(row.Filepath)
    
    binary = probmap > 0.9
    binary = binary.astype(np.uint8)
    binary_filled = ndi.binary_fill_holes(binary).astype(np.uint8)
    
    # save binary stacks
    tiff.imwrite(bin_dir / f"{row.Line}_{row.Condition}_t{row.Timepoint}_{row.Series}_PXN_binary.tiff", binary_filled)

#%% load in binary masks
   
binary_masks_df = utils.load_path_into_df(bin_dir / "watershed_2D", keywordregex=[r"Series\d{3}"], keyword="separated")
binary_masks_df = binary_masks_df.rename(columns={"Filepath": "Bin_filepath"})

FL_masks_df = utils.load_path_into_df(parentdir / "ilastik_input" / "PXN" / "Max_Intensity", keywordregex=[r"Series\d{3}"], keyword="PXN")
FL_masks_df = FL_masks_df.rename(columns={"Filepath": "FL_filepath"})

merge_cols = [col for col in binary_masks_df.columns if not col.endswith("filepath")]
full_df = binary_masks_df.merge(FL_masks_df, on=merge_cols, how="outer")
#%% preprocess binary images
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table, perimeter
import math
from skimage.morphology import remove_small_objects, convex_hull_image

plot_label = False

binary_imgs = []
for row in full_df.itertuples(index=False):
    
    img = tiff.imread(row.Bin_filepath)
    img_FL = tiff.imread(row.FL_filepath)
    labeled = label(img)
    regions = regionprops(labeled)

    mask_large = np.zeros_like(labeled, dtype=bool)
    for region in regions: 
        if 100 <= region.area:
            mask_large[labeled == region.label] = True
    
    labeled_large = label(mask_large)
    regions_large = regionprops(labeled_large)
    
    areas = [r.area for r in regions_large]
    max_area = np.percentile(areas, 90)
    
    mask_filtered = np.zeros_like(labeled, dtype=bool)
    area_um = []
    circularity = []
    solidity = []
    aspect_ratio = []
    
    for region in regions_large:
        if region.area <= max_area * 1.5:
            mask_filtered[labeled_large == region.label] = True
            mask = (labeled_large == region.label)
            
            perim = perimeter(mask, neighborhood=8)
            circ = 4.0 * math.pi * (region.area) / (perim * perim)
            hull = convex_hull_image(mask)             
            convex_area_pixels = float(hull.sum())
            sld = float(region.area) / convex_area_pixels
            ar = region.axis_major_length / region.axis_minor_length
            
            area_um.append(region.area * 0.481 * 0.481)
            circularity.append(circ)
            solidity.append(sld)
            aspect_ratio.append(ar)
    
    labeled_filtered = label(mask_filtered)
    labeled_filtered = clear_border(labeled_filtered)
    

    
    filtered_labeled = {
        "Bin_filepath": row.Bin_filepath,
        "Labeled_masks": labeled_filtered,
        "Original_FL": img_FL,
        "Cell_area_um": area_um,
        "Cell_circularity": circularity,
        "Cell_AR": aspect_ratio,
        "Cell_Solidity": solidity
        }
    
    
    if plot_label:
        plt.figure()
        plt.imshow(labeled_filtered)
        plt.title(f"{row.Series}_{row.Line}_{max_area*1.5}")
        plt.axis("off")
        
    binary_imgs.append(filtered_labeled)

binary_imgs_df = pd.DataFrame(binary_imgs)
df_masks = pd.merge(full_df, binary_imgs_df, on="Bin_filepath")    

#%% plot overlay 

for row in df_masks.itertuples(index=False):
    base_img = row.Original_FL
    outline = (row.Labeled_masks > 0).astype(np.uint8)
    fig, ax = utils.plot_overlays_masks_cell(base_img = row.Original_FL, outlines = {"Cell": outline}, legend=False)

#%% calculate cell area

df_indiv_cells["Cell_area_um"] = pd.to_numeric(
    df_indiv_cells["Cell_area_um"], errors="coerce"
)
#%% plot area comparison 
df_indiv_cells = df_masks.explode("Cell_circularity").reset_index(drop=True)

import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator

# params 
comparison_col = "Line" # x axis comparison col
condition_col = "Timepoint" # comparison within comparison col
measure = "Cell_circularity"
indiv_condition = "Condition"
indiv = "infected" # filter for only plotting this group

# prepare data 
df_filtered = df_indiv_cells[df_indiv_cells[indiv_condition] == indiv]
df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()
df_clean[measure] = pd.to_numeric(df_clean[measure], errors="coerce")
df_clean = df_clean.dropna(subset=[measure]).copy()

# detect labels for 0 and 4 (supports numeric or string)
unique_tps = pd.unique(df_clean[condition_col].astype(str))
tp_to_float = {}
for tp in unique_tps:
    try:
        tp_to_float[tp] = float(tp)
    except Exception:
        tp_to_float[tp] = None
time0_labels = [tp for tp, val in tp_to_float.items() if val is not None and val == 0.0] or [tp for tp in unique_tps if tp == "0"]
time4_labels = [tp for tp, val in tp_to_float.items() if val is not None and val == 4.0] or [tp for tp in unique_tps if tp == "4"]
if not time0_labels or not time4_labels:
    raise ValueError(f"Couldn't find Timepoint labels for 0 and 4. Detected: {list(unique_tps)}")
time0_label = time0_labels[0]
time4_label = time4_labels[0]

# restrict data to only those two timepoints (keeps plot clean)
df_plot = df_clean[df_clean[condition_col].astype(str).isin([time0_label, time4_label])].copy()
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
lines = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]
hue_order = [time0_label, time4_label]

# helper for p-values
def welch_p(a_vals, b_vals):
    if len(a_vals) < 2 or len(b_vals) < 2:
        return np.nan
    _, p = ttest_ind(a_vals, b_vals, equal_var=False, nan_policy="omit")
    return p

# Build pair lists and p-value lists for each family
within_pairs = []          # ((Line, TP0), (Line, TP4))
within_pvals = []

between0_pairs = []        # ((LineA, TP0), (LineB, TP0))
between0_pvals = []

between4_pairs = []        # ((LineA, TP4), (LineB, TP4))
between4_pvals = []

# Within-line (0 vs 4)
for L in lines:
    a = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    b = df_plot[(df_plot[comparison_col].astype(str) == L) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    p = welch_p(a, b)
    if not np.isnan(p):
        within_pairs.append(((L, time0_label), (L, time4_label)))
        within_pvals.append(p)
'''        
# Between-lines at time0 and time4 (all pairwise line combinations)
for aL, bL in itertools.combinations(lines, 2):
    a0 = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    b0 = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == time0_label)][measure].astype(float).values
    p0 = welch_p(a0, b0)
    if not np.isnan(p0):
        between0_pairs.append(((aL, time0_label), (bL, time0_label)))
        between0_pvals.append(p0)

    a4 = df_plot[(df_plot[comparison_col].astype(str) == aL) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    b4 = df_plot[(df_plot[comparison_col].astype(str) == bL) & (df_plot[condition_col].astype(str) == time4_label)][measure].astype(float).values
    p4 = welch_p(a4, b4)
    if not np.isnan(p4):
        between4_pairs.append(((aL, time4_label), (bL, time4_label)))
        between4_pvals.append(p4)
'''
# Apply FDR correction per family
within_corr = multipletests(within_pvals, method="fdr_bh")[1] if within_pvals else []
#between0_corr = multipletests(between0_pvals, method="fdr_bh")[1] if between0_pvals else []
#between4_corr = multipletests(between4_pvals, method="fdr_bh")[1] if between4_pvals else []

# Create single grouped violin plot: x=Line, hue=Timepoint (0 and 4)
plt.figure(figsize=(max(8, len(lines) * 1.2), 6))
ax = sns.violinplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                    order=lines, hue_order=hue_order, split=False, cut=0, palette=constants.CMAP)
sns.stripplot(data=df_plot, x=comparison_col, y=measure, hue=condition_col, order=lines,
              hue_order=hue_order, dodge=True, size=3, jitter=True, palette=["k","k"], alpha=0.6)

# The overlay above added a second legend; remove duplicate legend entries and keep single
handles, labels = ax.get_legend_handles_labels()
# keep only the first two (hue) and set the legend properly
if len(handles) >= 2:
    ax.legend(handles[:2], labels[:2], title=condition_col)

# Annotate using three separate Annotator instances (they accept pairs when both x and hue are used)
# 1) within-line
if within_pairs:
    annot_within = Annotator(ax, within_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                             order=lines, hue_order=hue_order)
    annot_within.configure(text_format="star", loc="inside")
    annot_within.set_pvalues_and_annotate(list(within_corr))
'''
# 2) between-lines at time0
if between0_pairs:
    annot_b0 = Annotator(ax, between0_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                         order=lines, hue_order=hue_order)
    annot_b0.configure(text_format="star", loc="outside", line_offset=6)  # put these outside so they don't overlap inside-line pairs
    annot_b0.set_pvalues_and_annotate(list(between0_corr))

# 3) between-lines at time4
if between4_pairs:
    annot_b4 = Annotator(ax, between4_pairs, data=df_plot, x=comparison_col, y=measure, hue=condition_col,
                         order=lines, hue_order=hue_order)
    annot_b4.configure(text_format="star", loc="outside", line_offset=12)  # offset further out so both timepoint families fit
    annot_b4.set_pvalues_and_annotate(list(between4_corr))
'''
ax.set_xlabel("")
ax.set_ylabel("")
#ax.set_title(f"{measure}: Time {time0_label} vs {time4_label} per Line; within-line and between-line comparisons")
plt.tight_layout()
plt.show()

#%% compare t0 to t0 between lines
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import matplotlib.patches as mpatches

# ---------------- settings ----------------
comparison_col = "Line"
condition_col = "Timepoint"
measure = "Cell_circularity"
indiv_condition = "Condition"
indiv = "uninfected" # filter for only plotting this group

plot_timepoint_value = 4
# ------------------------------------------

# Prepare data
df_filtered = df_indiv_cells[df_indiv_cells[indiv_condition] == indiv]
df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()


# Detect timepoint label corresponding to 0
unique_tps = pd.unique(df_clean[condition_col].astype(str))
tp_label = None
for tp in unique_tps:
    try:
        if float(tp) == float(plot_timepoint_value):
            tp_label = tp
            break
    except:
        continue

if tp_label is None:
    raise ValueError(f"Could not find Timepoint {plot_timepoint_value}")

# Filter to Time 0
df_plot = df_clean[df_clean[condition_col].astype(str) == tp_label].copy()
lines = sorted(pd.unique(df_plot[comparison_col].astype(str)))
lines = lines_ordered

# Generate color palette per line
palette = dict(zip(lines, sns.color_palette("colorblind", n_colors=len(lines))))
palette = constants.CMAP

# ---- Pairwise Welch tests between lines ----
pairs = []
pvals = []

for aL, bL in itertools.combinations(lines, 2):
    a_vals = df_plot[df_plot[comparison_col].astype(str) == aL][measure].astype(float)
    b_vals = df_plot[df_plot[comparison_col].astype(str) == bL][measure].astype(float)
    if len(a_vals) < 2 or len(b_vals) < 2:
        continue
    _, p = ttest_ind(a_vals, b_vals, equal_var=False)
    pairs.append((aL, bL))
    pvals.append(p)

if not pairs:
    raise ValueError("Not enough samples for pairwise comparisons.")

# FDR correction
_, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")

# ---- Plot ----
plt.figure(figsize=(max(8, len(lines)*1.2), 6))
ax = sns.violinplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    palette=palette,
    cut=0
)

sns.stripplot(
    data=df_plot,
    x=comparison_col,
    y=measure,
    order=lines,
    color="k",
    size=3,
    jitter=True,
    alpha=0.6
)

# Annotate comparisons
annot = Annotator(ax, pairs, data=df_plot, x=comparison_col, y=measure, order=lines)
annot.configure(text_format="star", loc="outside")
annot.set_pvalues_and_annotate(list(pvals_corr))

# ---- Custom legend showing Lines ----
legend_handles = [mpatches.Patch(color=palette[line], label=line) for line in lines]
ax.legend(handles=legend_handles, title="Line", bbox_to_anchor=(1.02, 1), loc="upper left")

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_ylim(top = 1)
#ax.set_title(f"{measure} at Time {tp_label} — Between-Line Comparisons")

plt.tight_layout()
plt.show()

#%% change in area + percent change per cell not possible (not same cell)

#%% change in area + percent change
comparison_col = "Line"
condition_col = "Timepoint"
measure = "Cell_area_um"
indiv_condition = "Condition"
indiv = "infected" # filter for only plotting this group

df_filtered = df_indiv_cells[df_indiv_cells[indiv_condition] == indiv]
df_clean = utils.remove_outliers_iqr(df_filtered, [condition_col, comparison_col], measure).copy()

measure_stats = (
    df_clean.pivot_table(
        index='Line',
        columns='Timepoint',
        values='Cell_area_um',
        aggfunc='mean'
    )
    .assign(diff=lambda x: x['4'] - x['0'])  # adjust category names
).reset_index()




#%%
wide = (
    df_clean.groupby([comparison_col, condition_col])[measure]
      .mean()
      .unstack()
).reset_index()

wide['\u0394 Cell Area (\u00B5m\u00B9)'] = wide['4'] - wide['0']
wide['\u0394% Cell Area'] = (wide['4'] - wide['0']) / wide['0'] * 100
wide['\u0394 Area (\u00B5m\u00B9/hr)'] = wide['\u0394 Cell Area (\u00B5m\u00B9)'] / 4
wide = wide.rename(columns={"0": "Area 0hrs (\u00B5m\u00B9)", "4": "Area 4hrs (\u00B5m\u00B9)"})



wide_display = wide.round(2)
wide_display = wide_display.set_index('Line').reindex(lines_ordered).reset_index()

utils.create_table(wide_display, body_fontsize=6, header_fontsize=6, fig_size=(6, 1.25))