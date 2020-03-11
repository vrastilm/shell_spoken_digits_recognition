import numpy as np
import librosa
import scipy as scipy


def wave_to_mfcc_matrix(wave_file: str, max_wave_len: int, window_len: int, max_mfcc_coefs: int) -> np.array:
    data, fs = librosa.load(wave_file, sr=8000)
    if len(data) > max_wave_len:
        data = data[:max_wave_len]
    
    data = np.pad(data, (0, (max_wave_len - len(data))))

    mfcc_matrix = librosa.feature.mfcc(data, sr=fs, n_mfcc=max_mfcc_coefs)
    
    return fs, mfcc_matrix
