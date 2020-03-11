from .classifier import Classifier
from tensorflow.keras.models import load_model
from ..utils.preprocessing import wave_to_mfcc_matrix
import numpy as np


class CnnWrapper(Classifier):
    """
    Wrapper class for convolutional neural network model
    """

    def __init__(self):
        self.model = load_model('shell_asr/src/classification/shell_asr_model')

    def classify(self, X: np.ndarray, fs: int, max_wave_len: int, window_len: int, max_mfcc_coefs: int) -> np.ndarray:
        """
        Method for classification of speech signal
        """
        features = np.array(np.reshape([wave_to_mfcc_matrix(
            X, fs, max_wave_len, window_len, max_mfcc_coefs)], (1, 20, 79, 1)))
        return self.model.predict(features)
