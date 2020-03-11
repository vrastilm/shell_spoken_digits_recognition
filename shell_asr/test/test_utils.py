import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from src.utils.mapping import map_prediction_to_dir
from src.utils.preprocessing import wave_to_mfcc_matrix


class TestUtils(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(map_prediction_to_dir(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1])), ('two', 0.9))

    @patch('librosa.feature.mfcc')
    def test_preprocessing(self, mfcc):
        wave_to_mfcc_matrix([], 0, 0, 0, 0)
        self.assertTrue(mfcc.called)
