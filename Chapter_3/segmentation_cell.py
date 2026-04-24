# pad images to prevent edge artifacts

from tqdm import tqdm
import tifffile as tiff
from skimage.measure import label
import numpy as np
import constants


def postprocess_cell(allpaths_dict: dict, padding=50, savedir=""):
    cell_final = {}
    for cell in tqdm(allpaths_dict[constants.CELL].keys()):
        cell_binary = tiff.imread(allpaths_dict[constants.CELL][cell])
        # turn binary image into labelled image (uint8)
        cell_labelled = label(cell_binary).astype(np.uint8)
        cell_final[cell] = np.pad(
            cell_labelled, pad_width=50, mode="constant", constant_values=0
        )

    if savedir != "":
        for cell, image in cell_final.items():
            num_labels = image.max()

            if num_labels > 1:
                print(f"Multiple cells found in {cell}")

        for cell in cell_final.keys():
            cell_layer = f"{cell}_layer0"
            filename = cell_layer + "_cell_labelled.tif"
            tiff.imwrite(
                savedir / filename,
                cell_final[cell].astype(np.uint8),
            )

        print("Processed cell segmentation saved as tiff")

        return

    else:
        return cell_final


# %% test script
import pathlib as Path

if __name__ == "__main__":
    treatment = "control"
    input_dir = Path(rf"C:/Users/IvyWork/Desktop/projects/dataset/input_{treatment}")
    img_dir = input_dir / "cell"
    cell_final = postprocess_cell(img_dir)
