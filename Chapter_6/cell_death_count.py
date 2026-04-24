from pathlib import Path
import tifffile as tiff
import numpy as np
import skimage as ski
from scipy import ndimage as ndi
import utils as utl
import matplotlib.pyplot as plt
import pandas as pd
import utils
import cv2
from roifile import ImagejRoi
from tqdm import tqdm
#%% IO
lines_uninfected = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]
lines_infected = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]


parentdir = Path(r"C:/Users/IvyWork/Documents/Ligoxygakis_data/0_stacks_renamed/")

# load in bacteria and macrophag emasks 
PXN_dir = parentdir / "PXN" /"3D_lowres" # use fluorescent signal - mask with cell label, count how many labels have signal (min) above threshold covering most of cell
count_dir = parentdir / "all_channels" / "3D_lowres" 
BF_dir = parentdir / "BF" / "3D_lowres" 

PXN_signal = utils.load_path_into_df(PXN_dir, keywordregex=[r"Series\d{3}"], keyword="PXN")
PXN_signal = PXN_signal.rename(columns={"Filepath": "PXN_filepath"})

cell_count_roi = utils.load_path_into_df(count_dir, keywordregex=[r"Series\d{3}"], keyword="cell_count")
cell_count_roi = cell_count_roi.rename(columns={"Filepath": "CellRoi_filepath"})

BF_stack = utils.load_path_into_df(BF_dir, keywordregex=[r"Series\d{3}"], keyword="BF")
BF_stack = BF_stack.rename(columns={"Filepath": "BF_filepath"})

merge_cols = [col for col in PXN_signal.columns if not col.endswith("filepath")]
filepath_df = PXN_signal.merge(cell_count_roi, on=merge_cols, how="outer")
filepath_df = filepath_df.merge(BF_stack, on=merge_cols, how="outer")
#%% BF cell binary 
plot = True

cell_count_list = []

for row in tqdm(filepath_df.itertuples(index=False)):
    
    # prepare FL
    PXN_img = tiff.imread(row.PXN_filepath)
    PXN_sum = PXN_img.sum(axis = 0)
    PXN_uint8 = utils.to_uint8_percentile(PXN_sum)
    PXN_binary, _ = utils.threshold_otsu(PXN_uint8) 
    
    # prepare roi
    cell_rois = ImagejRoi.fromfile(row.CellRoi_filepath)
    cells = utils.overlapping_to_layers(cell_rois, (512,512))
    # erode to cell core
    k = np.ones((3, 3), np.uint8)
    eroded_cells = [cv2.erode(img, k, iterations = 3) for img in cells]
    
    total_count_FL = 0
    total_count_noFL = 0
    total_mask_FL = []
    total_mask_noFL = []
    
    for cell_layer in eroded_cells:
        count_FL, count_noFL, FL_mask, noFL_count, _ = utils.cell_FL_count(cell_layer, PXN_binary, overlap_fraction = 0.3, plot = True)
        total_count_FL += count_FL
        total_count_noFL += count_noFL
        total_mask_FL.append(FL_mask)
        total_mask_noFL.append(noFL_count)
    
    total_cells = total_count_FL + total_count_noFL
    cell_count = {
        "Series": row.Series,
        "Condition": row.Condition,
        "Line": row.Line,
        "Timepoint": row.Timepoint,
        "PXN+_count": total_count_FL,
        "PXN-_count": total_count_noFL,
        "Cell_count": total_cells,
        "PXN+_proportion": total_count_noFL / total_cells,
        }
    
    cell_count_list.append(cell_count)
    
    BF_img = tiff.imread(row.BF_filepath)
    z,y,x = BF_img.shape
    BF_slice = BF_img[round(z/2), :,:]
    
    if plot:
        all_cells_FL = {
            f"cell_count_{i}" : layer for i,layer in enumerate(total_mask_FL)
            }

        fig, ax = utils.plot_overlays_masks_cell(base_img = BF_slice, outlines = all_cells_FL, masks = {"PXN": PXN_binary}, legend = False)
        
        all_cells_noFL = {
            f"cell_count_{i}" : layer for i,layer in enumerate(total_mask_noFL)
            }

        fig, ax = utils.plot_overlays_masks_cell(base_img = BF_slice, outlines = all_cells_noFL, masks = {"PXN": PXN_binary}, legend = False)
        

cell_count_df = pd.DataFrame(cell_count_list)

#%% check roi overlay

image_idx = 0

BF_img = tiff.imread(filepath_df.loc[image_idx].BF_filepath)
BF_slice = BF_img[round(z/2), :,:]

cell_rois = ImagejRoi.fromfile(filepath_df.loc[image_idx].CellRoi_filepath)
cells = utils.overlapping_to_layers(cell_rois, (512,512))
# erode to cell core
k = np.ones((3, 3), np.uint8)
eroded_cells = [cv2.erode(img, k, iterations = 3) for img in cells]

all_cells = {
    f"cell_count_{i}" : layer for i,layer in enumerate(eroded_cells)
    }

fig, ax = utils.plot_overlays_masks_cell(base_img = BF_slice, outlines = all_cells, legend = False)

#%% prepare condition stats and compare conditions/timepoints

condition_df = (
    cell_count_df
    .groupby(['Condition', 'Line', "Timepoint"], as_index=False)
    .agg({
        'PXN+_count': 'sum',
        'PXN-_count': 'sum',
        'Cell_count': 'sum' 
    })
)

condition_df["PXN-_proportion"] = condition_df["PXN-_count"] / condition_df["Cell_count"]

#%% plotting cell death - raw counts

import numpy as np

for condition in ["uninfected", "infected"]:
    df = condition_df[condition_df["Condition"]==condition]
    df = df.rename(columns={"PXN+_count": "Alive", "PXN-_count": "Dead"})
    # Optional: filter by condition if needed
    # df = df[df["condition"] == "A"]
    
    alive_colors = ["#1b9e77", "#66a61e"]
    dead_colors  = ["#d95f02", "#e7298a"]
    
    cell_lines = df["Line"].unique()
    cell_lines = lines_uninfected
    timepoints = df["Timepoint"].unique()
    
    x = np.arange(len(cell_lines))  # positions for cell lines
    width = 0.35  # width of bars
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    for i, tp in enumerate(timepoints):
        subset = (
            df[df["Timepoint"] == tp]
            .set_index("Line")
            .loc[cell_lines]
        )    
        alive = subset["Alive"]
        dead = subset["Dead"]
        
        ax.bar(x + i*width, alive, width, label=f"Timepoint {tp}hrs Alive", color = alive_colors[i])
        ax.bar(x + i*width, dead, width, bottom=alive, label=f"Timepoint {tp}hrs Dead", color = dead_colors[i])
    
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(cell_lines)
    ax.set_ylabel("Number of Cells")
    ax.set_xlabel(f"Cell Line ({condition})")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

#%% plotting cell death - proportions

import numpy as np
uninfected_df = condition_df[condition_df["Condition"]=="uninfected"]
uninfected_df = uninfected_df.rename(columns={"PXN+_count": "Alive", "PXN-_count": "Dead", "PXN-_proportion": "Dead_prop"})
# Optional: filter by condition if needed
# df = df[df["condition"] == "A"]

alive_colors = ["#1b9e77"]
dead_colors  = ["#d95f02"]

cell_lines = uninfected_df["Line"].unique()
cell_lines = lines_uninfected
timepoints = uninfected_df["Timepoint"].unique()

x = np.arange(len(cell_lines))  # positions for cell lines
width = 0.35

fig, ax = plt.subplots(figsize=(8,6))

for i, tp in enumerate(timepoints):
    subset = (
        uninfected_df[uninfected_df["Timepoint"] == tp]
        .set_index("Line")
        .reindex(cell_lines)
    )

    dead_prop = subset["Dead_prop"]*100 # percentage

    ax.bar(x + i*width, dead_prop, width, label=tp)

ax.set_xticks(x + width/2)
ax.set_xticklabels(cell_lines)
ax.set_ylim(0, 100)
ax.set_ylabel("Dead Cells (%)")
ax.set_xlabel("Cell Line")
ax.legend(title="Timepoint")

plt.tight_layout()
plt.show()

#%% plot cell death - proportions (infected vs uninfected)
df = condition_df.copy()
df = df.rename(columns={"PXN+_count": "Alive", "PXN-_count": "Dead", "PXN-_proportion": "Dead_prop"})
# Optional: filter by condition if needed
# df = df[df["condition"] == "A"]

colors = {
    "uninfected": "#0072B2",   # blue
    "infected": "#E69F00"    # orange
}

cell_lines = sorted(df["Line"].unique())
cell_lines = lines_uninfected
conditions = sorted(df["Condition"].unique())
conditions = ["uninfected", "infected"]
timepoints = sorted(df["Timepoint"].unique())

x = np.arange(len(cell_lines))
bar_width = 0.8 / (len(conditions) * len(timepoints))

fig, ax = plt.subplots(figsize=(8,6))

for c_idx, cond in enumerate(conditions):
    for t_idx, tp in enumerate(timepoints):

        subset = df[
            (df["Condition"] == cond) &
            (df["Timepoint"] == tp)
        ].set_index("Line").reindex(cell_lines)

        offset_index = c_idx * len(timepoints) + t_idx
        offsets = x + (offset_index - ((len(conditions)*len(timepoints)-1)/2)) * bar_width

        ax.bar(
            offsets,
            subset["Dead_prop"],
            bar_width,
            label=f"{cond} - {tp}h",
            color = colors[cond],
            alpha=1 if tp == "0" else 0.6,
        )

ax.set_xticks(x)
ax.set_xticklabels(cell_lines, rotation=45, ha="right")
ax.set_ylabel("Dead Cell Proportion")
ax.set_ylim(0, 0.5)
ax.set_xlabel("Cell Line")
ax.legend(title="Condition / Timepoint")

plt.tight_layout()
plt.show()

#%% show table of raw counts
utils.create_table(condition_df)

#%% statistical test

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

df = condition_df.copy()
df = df.rename(columns={"PXN+_count": "Alive", "PXN-_count": "Dead", "PXN-_proportion": "Dead_prop"})

# Make sure columns are categorical
df["Line"] = df["Line"].astype("category")
df["Timepoint"] = df["Timepoint"].astype("category")
df["Condition"] = df["Condition"].astype("category")

# -----------------------------
# Helper function for 2x2 tests
# -----------------------------
def run_2x2(table):
    chi2, p_chi, dof, expected = chi2_contingency(table, correction=False)
    if (expected < 5).any():
        _, p = fisher_exact(table)
        return p, "fisher"
    return p_chi, "chi2"

results = []
pvals = []

# 1️⃣ Infection effect within each cell line at each timepoint
for line in df["Line"].unique():
    for time in df["Timepoint"].unique():
        sub = df[(df.Line == line) & (df.Timepoint == time)]
        if len(sub) != 2:
            continue
        
        uninf = sub[sub.Condition == "uninfected"].iloc[0]
        inf = sub[sub.Condition == "infected"].iloc[0]
        
        table = np.array([
            [uninf.Dead, inf.Dead],
            [uninf.Alive, inf.Alive]
        ])
        
        p, test = run_2x2(table)
        
        results.append({
            "comparison": "Infection effect",
            "Line": line,
            "Timepoint": time,
            "p_raw": p,
            "test": test
        })
        pvals.append(p)

# Holm correction
rej, p_adj, _, _ = multipletests(pvals, method="holm")

for i in range(len(results)):
    results[i]["p_adj"] = p_adj[i]
    results[i]["significant"] = rej[i]

results_df = pd.DataFrame(results)
print(results_df)

df["prop_dead"] = df["Dead"] / df["Cell_count"]

ci = df.apply(
    lambda row: proportion_confint(row.Dead, row.Cell_count, method="wilson"),
    axis=1
)

df["ci_low"] = [x[0] for x in ci]
df["ci_high"] = [x[1] for x in ci]

import matplotlib.pyplot as plt

cell_lines = df["Line"].unique()
times = df["Timepoint"].unique()
infection_states = ["uninfected", "infected"]

fig, axes = plt.subplots(1, len(cell_lines), figsize=(5*len(cell_lines),5), sharey=True)

if len(cell_lines) == 1:
    axes = [axes]

for ax, line in zip(axes, cell_lines):
    sub = df[df.Line == line]
    x = np.arange(len(times))
    width = 0.35
    
    for i, inf_state in enumerate(infection_states):
        data = sub[sub.Condition == inf_state]
        means = data.sort_values("Timepoint")["prop_dead"].values
        lows = data.sort_values("Timepoint")["ci_low"].values
        highs = data.sort_values("Timepoint")["ci_high"].values
        
        ax.bar(x + i*width - width/2, means,
               width,
               yerr=[means-lows, highs-means],
               capsize=4,
               label=inf_state)
    
    ax.set_xticks(x)
    ax.set_xticklabels(times)
    ax.set_title(line)
    ax.set_ylim(0, df["prop_dead"].max() + 0.1)
    ax.set_ylabel("Proportion dead")

axes[0].legend()
plt.tight_layout()
plt.show()

utils.create_table(results_df)
#%% uninfected cell death over time
import statsmodels.api as sm
import statsmodels.formula.api as smf
uninfected_df = condition_df[condition_df["Condition"]=="uninfected"]
uninfected_df = uninfected_df.rename(columns={"PXN+_count": "Alive", "PXN-_count": "Dead"})

model = smf.glm(
    formula="Dead + Alive ~ Line * Timepoint",
    data=uninfected_df,
    family=sm.families.Binomial()
).fit()


print(model.summary())

#%% save data as xlsx
with pd.ExcelWriter(inputdir / "cell_death_count.xlsx") as writer:
    all_cell_stats.to_excel(writer, sheet_name = timepoint, index=False)


#%% plot differences between lines

