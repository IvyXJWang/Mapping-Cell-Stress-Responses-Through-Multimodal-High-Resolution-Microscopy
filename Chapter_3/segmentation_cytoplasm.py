import numpy as np
from pathlib import Path
import tifffile as tiff
import constants
# from skimage.measure import label


def remove_object_from_base(base_img: np.ndarray, other_objects: list):
    # Combine all other binary masks into one
    allobjects = np.zeros_like(base_img, dtype=bool)
    for obj in other_objects:
        allobjects |= obj.astype(bool)

    # Keep only parts of main_binary that are not in any of the others
    background = base_img & (~allobjects)

    return background, allobjects


def postprocess_cytoplasm(
    allpaths_processed: dict, organelle_keywords: list, savedir=""
):
    cell_path_dict = allpaths_processed[constants.CELL]
    binary_cytoplasm_labelled = {}
    for cell, cell_path_list in cell_path_dict.items():
        cell_img = tiff.imread(cell_path_list)
        cell_binary = cell_img > 0

        other_img_list = []
        for organelle in organelle_keywords:
            # for idx in range(0, len(allpaths_processed[organelle][cell])):
            #print(organelle)
            img = tiff.imread(allpaths_processed[organelle][cell]) > 0
            other_img_list.append(img)

        binary_cytoplasm_labelled[cell], all_objects = remove_object_from_base(
            cell_binary, other_img_list
        )

    if savedir != "":
        for cell in binary_cytoplasm_labelled.keys():
            cyto_layer = f"{cell}_layer0"
            filename = cyto_layer + "_cytoplasm_labelled.tif"
            tiff.imwrite(
                savedir / filename,
                binary_cytoplasm_labelled[cell].astype(np.uint8),
            )
        print("Processed cytoplasm segmentation saved as tiff")

        return

    else:
        return binary_cytoplasm_labelled, all_objects


# %% test script
if __name__ == "__main__":
    treatment = "control"
    input_dir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset/input_{treatment}")
    results_dirs = input_dir / "results"
    binary_cytoplasm_labelled = postprocess_cytoplasm(results_dirs)

    for cell in binary_cytoplasm_labelled.keys():
        cyto_layer = f"{cell}_layer0"
        filename = cyto_layer + "_cytoplasm_labelled.tif"
        tiff.imwrite(
            results_dirs / filename,
            binary_cytoplasm_labelled[cell].astype(np.uint8),
        )
