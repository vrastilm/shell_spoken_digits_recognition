import numpy as np
import librosa

def wave_to_mfcc_matrix(data: np.ndarray, fs: int, max_wave_len: int, window_len: int, max_mfcc_coefs: int) -> np.array:
    if len(data) > max_wave_len:
        data = data[:max_wave_len]
    
    data = np.pad(data, (0, (max_wave_len - len(data))))

    return librosa.feature.mfcc(data, sr=fs, n_mfcc=max_mfcc_coefs)