import numpy as np
from typing import Tuple


def map_prediction_to_dir(vec: np.ndarray) -> Tuple[str, float]:
    """
    Maps prdicted vector to directory and finds best prediction
    """
    # needs to have precise order, as during model learning
    dirs = ['nine', 'eight', 'zero', 'three', 'seven', 'five', 'six', 'one', 'two', 'four']

    return (dirs[np.argmax(vec)], vec.max())