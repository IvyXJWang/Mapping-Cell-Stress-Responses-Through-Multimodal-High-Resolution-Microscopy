from pathlib import Path

# paths
FIGDIR_THESIS = Path(r"C:/Users/IvyWork/Documents/Thesis/figures_draft")
DOWNLOADDIR = Path(r"C:/Users/IvyWork/Downloads")
PARENTDIR = Path(r"C:/Users/IvyWork/Desktop/projects/dataset")

# image info
PX_SIZE = 0.01
XRAY = "xray"
# organelles
LIPID_DROPLETS = "lipiddroplets"
MITOCHONDRIA = "mitochondria"
NUCLEUS = "nucleus"
CELL = "cell"
CYTOPLASM = "cytoplasm"
AU = "au"
ORGANELLE_CP_LIST = [CELL, CYTOPLASM, LIPID_DROPLETS, MITOCHONDRIA, NUCLEUS]
ORGANELLE_LIST = [
    "Cell",
    "Cytoplasm",
    "LipidDroplets",
    "Mitochondria",
    "Nucleus",
    ]

# features
OBJ_SHAPE = "AreaShape"
OBJ_INTENSITY = "Intensity"
OBJ_TEXTURE = "Texture"
OBJ_RADIALDISTRIBUTION = "RadialDistribution"
OBJ_STRUCTURE = "Structure"
FEATURE_TYPE_CP_LIST = [
    [OBJ_SHAPE],
    [OBJ_INTENSITY], [OBJ_RADIALDISTRIBUTION],
    [OBJ_TEXTURE],
    [OBJ_STRUCTURE],
]

FEATURE_TYPE_LIST = [
    "Geometric", 
    "Densitometric",
    "Textural",
    "Structural"
    ]

CELLID_COL = "Metadata_CellID"
TEATMENT_CTRL = "control"
TREATMENT_EXP = "inflammation"

# color schemes
ORGANELLE_CP_CMAP = {
    MITOCHONDRIA: "#EE3377" , #magenta
    NUCLEUS: "#EE7733", # orange
    LIPID_DROPLETS: "#33BBEE", #cyan
    CELL: (0,0,0), #black
    CYTOPLASM: "#0077BB", #blue
}

ORGANELLE_CMAP = {
    "Mitochondria": "#EE3377" , #magenta
    "Nucleus": "#EE7733", # orange
    "LipidDroplets": "#33BBEE", #cyan
    "Cell": "#000000", #black // #808080", #grey
    "Cytoplasm": "#0077BB", #blue
    
    MITOCHONDRIA: "#EE3377" , #magenta
    NUCLEUS: "#EE7733", # orange
    LIPID_DROPLETS: "#33BBEE", #cyan
    CELL: (0,0,0), #black
    CYTOPLASM: "#0077BB", #blue
}

FEATURE_TYPE_CP_CMAP = {
    OBJ_SHAPE: "#ca3146", #red
    OBJ_RADIALDISTRIBUTION: "#e35306", #orange
    OBJ_INTENSITY: "#e35306", #orange
    OBJ_TEXTURE: "#e7aa63", #gold
    OBJ_STRUCTURE: "#c3d690", #olive
}

FEATURE_TYPE_CMAP = {
    "Geometric": "#ca3146", #red
    "Densitometric": "#e35306", #orange
    "Textural": "#e7aa63", #gold
    "Structural": "#c3d690", #olive
}

# graph constants
FIG_SIZE = (6, 6)
LINE_WIDTH = 1
