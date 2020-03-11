import sounddevice as sd
from sounddevice import CallbackFlags
import numpy as np
from scipy.io.wavfile import write
from threading import Timer, Event
from .utils.loader import Loader
from uuid import uuid1
import click
from .classification.factory import ClassifierFactory
import librosa
from .utils.mapping import map_prediction_to_dir


class SingletonState:
    __instance = None

    @staticmethod
    def getInstance() -> object:
        """ 
        Static access method
        """
        if SingletonState.__instance == None:
            SingletonState()
        return SingletonState.__instance

    def __init__(self):
        """ 
        Virtually private constructor
        """
        if SingletonState.__instance != None:
            raise Exception("State is a singleton!")
        else:
            self.is_recording = False
            self.record = []
            self.fs = 8000
            self.max_wave_len = 40000
            self.win_len = 350
            self.max_mfcc_coefs = 20
            self.duration = int(self.max_wave_len/self.fs)
            self.stream = None
            self.record = np.array([], dtype=np.float32)
            self.timer = None
            self.loader = None
            self.stop_signal = Event()
            self.classifier = ClassifierFactory.get_classifier_wrapper()
            SingletonState.__instance = self

    def _recording_callback(self, outdata: np.ndarray, frames: int, time, status: CallbackFlags) -> None:
        if outdata.shape[1] == 1:
            if outdata.shape[0] != frames:
                self.record = np.append(self.record, np.zeros(frames))
            else:
                self.record = np.append(self.record, outdata)
        else:
            raise Exception('Not supported number of channels')

    def _classify_record(self) -> None:
        click.echo('You said:')
        predicts = self.classifier.classify(
            self.record, self.fs, self.max_wave_len, self.win_len, self.max_mfcc_coefs)
        click.echo(map_prediction_to_dir(predicts))
        click.echo()
        self.record = []

    def _stop_recording_callback(self) -> None:
        self.gracefully_terminate_workers()
        self.is_recording = False

    def start_stop_recording(self) -> None:
        if not self.is_recording:
            click.echo(f"Start recording, max duration: {self.duration} s")
            self.is_recording = True

            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=1,
                dtype=np.float32,
                callback=self._recording_callback,
                finished_callback=self._classify_record)

            self.timer = Timer(self.duration, self._stop_recording_callback)

            self.loader = Loader(self.stop_signal)

            self.loader.start()
            self.timer.start()
            self.stream.start()
        else:
            self._stop_recording_callback()

    def gracefully_terminate_workers(self) -> None:
        """
        Method for gracefuly terminating
        """
        # kill timer
        if self.timer:
            self.timer.cancel()
            self.timer = None
        # kill loader
        if self.loader:
            self.stop_signal.set()
            self.loader.join()
            self.loader = None
            self.stop_signal.clear()
        # close stream
        if self.stream:
            self.stream.close()
            self.stream = None


state = SingletonState()
