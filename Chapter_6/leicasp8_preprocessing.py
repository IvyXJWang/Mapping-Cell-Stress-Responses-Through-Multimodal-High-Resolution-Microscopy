from pathlib import Path
import tifffile as tiff
import numpy as np
import pandas as pd

#%% Function definition 
import os
from pathlib import Path
import re
import shutil
import utils 

def is_tiff(file: Path) -> bool:
    return file.is_file() and (file.suffix == ".tif" or file.suffix == ".tiff")

def filelist_tiff(datadir: Path):
    """
    Return list of files in directory that are tiffs
    """
    return [str(f) for f in datadir.iterdir() if is_tiff(f)]



def detect_stack_type(filename):
    
    stack_type = ""
    
    if re.search(r"z\d{2}", filename):
        stack_type = stack_type + "_3D"
    else:
        stack_type = stack_type + "_2D"
    
    if re.search(r"t\d{2}", filename):
        stack_type = stack_type + "_timelapse"
    
    return stack_type
    
def sort_series_subfolders(SRC_DIR, regex = r"Series\d{3}", experiment_name = None):
    
    """
    Sorts all files into subfolders by series number
    Optional: prepend experiment name to each file - does not work yet
    """
    
    DEFAULT_DIR = "other"  # for files that match nothing 
    
    for file_path in SRC_DIR.iterdir():
        if not is_tiff(file_path):
            continue
    
        found = False
    
        if re.search(regex, file_path.name, re.IGNORECASE):
            stack_type = detect_stack_type(str(file_path.name))
            subdir = utils.extract_keyword(str(file_path.name), [regex]) + stack_type
            target_dir = SRC_DIR / subdir 
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(file_path), target_dir / file_path.name)
            found = True
    
        if not found:
            target_dir = SRC_DIR / DEFAULT_DIR
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(file_path), target_dir / file_path.name)
    


def rename_files(
    directory,
    prepend_text=None,
    replace_keyword=None,
    replace_with=None,
    recursive=False
):
    """
    Rename files in a directory by either:
    - Prepending text to filenames
    - Replacing a keyword in filenames

    Parameters:
        directory (str or Path): Target directory
        prepend_text (str): Text to prepend to filenames
        replace_keyword (str): Keyword in filename to replace
        replace_with (str): Replacement text
        recursive (bool): If True, process subdirectories
    """

    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        raise ValueError("Invalid directory path")

    if not prepend_text and not replace_keyword:
        raise ValueError("You must specify either prepend_text or replace_keyword")

    files = directory.rglob("*") if recursive else directory.glob("*")

    for file_path in files:
        if file_path.is_file():
            original_name = file_path.name
            new_name = original_name

            # Prepend text
            if prepend_text:
                new_name = prepend_text + original_name

            # Replace keyword
            if replace_keyword:
                if replace_with is None:
                    raise ValueError("replace_with must be specified when using replace_keyword")
                new_name = original_name.replace(replace_keyword, replace_with)

            # Skip if no change
            if new_name == original_name:
                continue

            new_path = file_path.with_name(new_name)

            # Prevent overwriting existing files
            if new_path.exists():
                print(f"Skipping (already exists): {new_path}")
                continue

            file_path.rename(new_path)
            print(f"Renamed: {original_name} → {new_name}")

#%% IO
condition = "AnxB10rnai_timelapse"
replicate = "nobac"

experiment_name = f"{condition}_{replicate}" 

parentdir = Path(r"C:/Users/IvyWork/Documents/Ligoxygakis_data/")
inputdir = parentdir / f"{condition}" / f"{replicate}"

sort_series_subfolders(inputdir, experiment_name)

#%% change file names - add condition to stack names
condition = "AnxB11rnai"
replicate = "nobac"
series = "Series024_3D"
timepoint = "t4"

experiment_name = f"{condition}_{replicate}"

parentdir = Path(r"C:/Users/IvyWork/Documents/Ligoxygakis_data/")
inputdir = parentdir / f"{condition}_timelapse" / f"{replicate}" / f"{series}" / "stacks"

# prepend
rename_files(inputdir, prepend_text=f"{condition}_{replicate}_{timepoint}_")

#%% replace
rename_files(inputdir, replace_keyword=f"{condition}_{replicate}_", replace_with=f"{condition}_{replicate}_{timepoint}_")

#%% copy file types to target directory
import shutil

def get_directory_names(parent_dir, keyword = None):
    parent = Path(parent_dir)
    
    if keyword is not None:
        subdir_list = [p.name for p in parent.iterdir() if p.is_dir() and keyword in p.name]
    else:
        subdir_list = [p.name for p in parent.iterdir() if p.is_dir()]

    return subdir_list

def get_all_tifs(directory):
    directory = Path(directory)
    return list(directory.glob("*.tif*"))

def copy_and_rename(source_path, target_directory, new_name):
    source = Path(source_path)
    target_dir = Path(target_directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    destination = target_dir / new_name
    shutil.copy2(source, destination)

    return destination

channel = "ecoli" # keyword of stack to copy

parentdir = Path(r"C:/Users/IvyWork/Documents/Ligoxygakis_data")

targetdir = parentdir / "0_stacks_renamed" / f"{channel}"
#targetdir = Path(r"C:/Users/IvyWork/Desktop/projects/dataset4D/3D_HighRes/ilastik_input/AnxB9rnai")
                 
targetdir.mkdir(parents=True, exist_ok=True)

subdir_list = get_directory_names(parentdir)[1:]

for directory in subdir_list:
    
    cell_line = directory.split("_")[0]
    
    sublist_halfbac = get_directory_names(parentdir / directory / "halfbac", keyword = "Series")    
    sublist_nobac = get_directory_names(parentdir / directory / "nobac", keyword = "Series")
    
    # find timelapse series
    timelapse_halfbac = [a for a in sublist_halfbac if "3D_timelapse" in a][0].split("_")[0]
    timelapse_nobac = [a for a in sublist_nobac if "3D_timelapse" in a][0].split("_")[0]
    
    for series in sublist_halfbac:
        condition = "infected"
        timepoint = "t0" if series.split("_")[0] < timelapse_halfbac else "t4"
        series_type = "_".join(series.split("_")[1:])
        
        seriesdir = parentdir / directory / "halfbac" / series
        tiff_files = sorted(seriesdir.glob("*.tif*"))
        img = tiff.imread(tiff_files[0])
        
        H, W = img.shape
        
        if H == 1024:
            FOV = "highres"
        elif H == 512:
            FOV = "lowres"
        else:
            FOV = "unknownres"
        
        stackdir = seriesdir / "stacks"
        tiff_list = get_all_tifs(stackdir)
        
        stackname_list = [str(tiff) for tiff in tiff_list if f"{channel}.tif" in str(tiff)]
        if len(stackname_list) == 0:
            continue
        
        stackname = stackname_list[0]
        stackpath = stackdir / stackname
    
        newfile_name = f"{cell_line}_{condition}_{timepoint}_{series}_{FOV}_{channel}.tif"
        targetdir_final = targetdir / f"{series_type}_{FOV}"
        targetdir_final.mkdir(parents=True, exist_ok=True)

        _ = copy_and_rename(stackpath, targetdir_final, newfile_name)
    
    for series in sublist_nobac:
        condition = "uninfected"
        timepoint = "t0" if series.split("_")[0] < timelapse_nobac else "t4"
        series_type = "_".join(series.split("_")[1:])

        seriesdir = parentdir / directory / "nobac" / series
        tiff_files = sorted(seriesdir.glob("*.tif*"))
        img = tiff.imread(tiff_files[0])
        
        H, W = img.shape
        
        if H == 1024:
            FOV = "highres"
        elif H == 512:
            FOV = "lowres"
        else:
            FOV = "unknownres"
        
        stackdir = seriesdir / "stacks"
        tiff_list = get_all_tifs(stackdir)
        
        stackname_list = [str(tiff) for tiff in tiff_list if f"{channel}.tif" in str(tiff)]
        if len(stackname_list) == 0:
            continue
        
        stackname = stackname_list[0]
        stackpath = stackdir / stackname
    
        newfile_name = f"{cell_line}_{condition}_{timepoint}_{series}_{FOV}_{channel}.tif"
        targetdir_final = targetdir / f"{series_type}_{FOV}"
        targetdir_final.mkdir(parents=True, exist_ok=True)

        _ = copy_and_rename(stackpath, targetdir_final, newfile_name)
#%%
inputdir = parentdir / "ilastik_input" / f"{channel}"

processed_dir = parentdir / "ilastik_input" / f"{channel}_processed"
processed_dir.mkdir(parents=True, exist_ok=True)

allfiles = utils.load_path_into_dict(inputdir, keywordregex=[r"Series\d{3}"], keyword="BF")
allfiles_df = utils.load_path_into_df(inputdir, keywordregex=[r"Series\d{3}"], keyword="BF")

root_directory = "/path/to/root"
target_directory = "/path/to/target"

results = process_tiffs_and_copy_using_tifffile(
    root_dir=root_directory,
    target_dir=target_directory,
    tiff_glob="*.tif",
    file_glob="*.txt",    # file you want to copy from deeper subdirectory
    search_depth=3,
    append_size_to_name=True,
    dry_run=True
)

for r in results:
    print(r)
