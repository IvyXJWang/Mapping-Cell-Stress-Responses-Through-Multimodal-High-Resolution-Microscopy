from pathlib import Path

PX_SIZE_LR = 481.47 #nm
PX_SIZE_HR = 96.20

LINES = ["W1118", "AnxB9rnai", "AnxB10rnai", "AnxB11rnai"]

CONDITION = ["uninfected", "infected"]

CMAP = {
        "W1118": "#009E73", # blueish green
        "AnxB9rnai":"#F0E442", # yellow
        "AnxB10rnai": "#D55E00", # vermillion
        "AnxB11rnai": "#CC79A7", # reddish purple
        "uninfected": "#56B4E9", # sky blue
        "infected": "#E69F00", # orange
        "cell_count": "#FF0000", #red 
        "0": "#008000", # green
        "4": "#4CBB17", # kelly green
        "Macrophage": "#FFFFFF", # white 
        "Cell": "#FF0000", # red
        "ecoli": "#FF0000", # red
        "PXN": "#FFFF00" # yellow
        }

FIG_DIR = Path("C:/Users/IvyWork/Documents/Thesis/figures_raw_results4")