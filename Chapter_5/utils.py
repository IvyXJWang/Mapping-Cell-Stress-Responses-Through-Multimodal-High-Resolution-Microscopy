from pathlib import Path
import constants
import os
import re


def is_tiff(file: Path) -> bool:
    return file.is_file() and (file.suffix == ".tif" or file.suffix == ".tiff")


def is_xlsx(file: Path) -> bool:
    return file.is_file() and file.suffix == ".xlsx"

def filelist_type(datadir: Path, file_type: str):
    """
    Return list of files in directory that are of a specific type
    """
    if file_type == "tiff":
        return [str(f) for f in datadir.iterdir() if is_tiff(f)]
    elif file_type == "xlsx":
        return [str(f) for f in datadir.iterdir() if is_xlsx(f)]


def filter_dictionary_subset(full_dictionary, keys_subset):
    filtered_dictionary = {
        outer_k: {k: v for k, v in inner.items() if k in keys_subset}
        for outer_k, inner in full_dictionary.items()
    }

    return filtered_dictionary

def extract_keyword(phrase: str, keywordregex: list, keyword_index=0):
    if type(keywordregex) != list:
        keywordregex = list(keywordregex)

    matches = []
    for keyword in keywordregex:
        match = re.findall(keyword, phrase)  # find pattern

        if match[0] not in matches:
            matches.append(match[0])  # uses first instance of keyword match

        if len(matches) == len(keywordregex):  # all keywords found in phrase
            return matches[
                keyword_index
            ]  # return specified keyword in list matching pattern

def rename_files(directory, find_pattern, replace_with):
    for filename in os.listdir(directory):
        if find_pattern in filename:
            new_filename = filename.replace(find_pattern, replace_with)
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

def load_path_into_dict(datadir, keywordregex=[r"CELL\d{3}"], keyword="", filetype = "tiff"):
    """ """
    filelist = filelist_type(datadir, filetype)  # all files in directory

    path_dict = {
        extract_keyword(Path(f).stem, keywordregex): str(f)
        for f in filelist
        if keyword in Path(f).stem
    }
    return path_dict

def get_unique_filename(filepath):
    path = Path(filepath)
    
    if not path.exists():
        return path

    counter = 1
    while True:
        new_path = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not new_path.exists():
            return new_path
        counter += 1

def rename_df_columns_with_keyword(df, rename_dict):
    """
    Rename columns in a DataFrame based on a provided mapping keyword dictionary.
    
    Parameters:
    df (pd.DataFrame): The DataFrame whose columns are to be renamed.
    rename_dict (dict): A dictionary where keys are current column names and values are new column names.
    
    Returns:
    pd.DataFrame: A DataFrame with renamed columns.
    """

    renamed_df = df.copy()
    for keyword, replacement in rename_dict.items():
        renamed_df.columns = renamed_df.columns.str.replace(keyword, replacement, regex=False)
    
    return renamed_df

def rename_list_with_keyword(lst, rename_dict):
    """
    Rename items in a list based on a provided mapping keyword dictionary.
    
    Parameters:
    lst (list): The list whose items are to be renamed.
    rename_dict (dict): A dictionary where keys are current item names and values are new item names.
    
    Returns:
    list: A list with renamed items.
    """

    renamed_list = []
    for item in lst:
        renamed_item = item
        for keyword, replacement in rename_dict.items():
            if keyword in item:
                renamed_item = item.replace(keyword, replacement)
                break  # Assuming only one keyword match per item
        renamed_list.append(renamed_item)
    
    return renamed_list