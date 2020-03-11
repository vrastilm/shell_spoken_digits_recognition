import os
from keras.utils import to_categorical
import numpy as np
from typing import List, Tuple
import functools


@functools.lru_cache(maxsize=128)
def _get_all_dirs(data_folder: str, ignored_names: Tuple[str]) -> List[str]:
    dirs = []
    content = os.listdir(data_folder)
    for item in content:
        if os.path.isdir(os.path.join(data_folder, item)) and data_folder not in ignored_names:
            dirs.append(item)

    return dirs


def get_one_hot_vec(data_folder: str, dir_name: str, ignored_names: Tuple[str]) -> np.ndarray:
    dirs = _get_all_dirs(data_folder, ignored_names)
    vec = np.zeros(len(dirs))

    vec[dirs.index(dir_name)] = 1

    return vec


def get_number_of_classes(data_folder: str, ignored_names: Tuple[str]) -> int:
    return len(_get_all_dirs(data_folder, ignored_names))
