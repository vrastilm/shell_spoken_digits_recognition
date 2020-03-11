from pynput.keyboard import Key, Listener
import click
import os
import sys
from src.recording_state import state
import tensorflow as tf
import atexit

tf.keras.backend.set_learning_phase(0)
# set currrent file directory as working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _termination_handler():
    state.gracefully_terminate_workers()
    click.echo('Gracefully terminating')


def _on_press(key):
    if key == Key.alt_l:
        state.start_stop_recording()


@click.command()
def shell_asr():
    """
    Program for recording spoken digits and classification of them.
    """
    intro = 'Click on key left Alt for start recording your speech and afterwards click left Alt again for stop recording or wait for timeout'

    click.echo()
    click.echo('-'*len(intro))
    click.echo(intro)
    click.echo('-'*len(intro))
    click.echo()

    atexit.register(_termination_handler)

    with Listener(on_press=_on_press) as listener:
        listener.join()
