import threading
import sys
import time


class Loader(threading.Thread):
    """
    Class for async printing of loader during recording
    """

    def __init__(self, stop_signal):
        threading.Thread.__init__(self)
        self.stop_signal = stop_signal

    def run(self):
        """
        Runs loader that prints load bar during recording
        """

        sleep = 0.08
        sys.stdout.flush()

        while True:
            if self.stop_signal.wait(0):
                sys.stdout.write('\n')
                sys.stdout.flush()
                break

            sys.stdout.write('\r[-    ]')
            sys.stdout.flush()
            time.sleep(sleep)
            sys.stdout.write('\r[ -   ]')
            sys.stdout.flush()
            time.sleep(sleep)
            sys.stdout.write('\r[  -  ]')
            sys.stdout.flush()
            time.sleep(sleep)
            sys.stdout.write('\r[   - ]')
            sys.stdout.flush()
            time.sleep(sleep)
            sys.stdout.write('\r[    -]')
            sys.stdout.flush()
            time.sleep(sleep)
            sys.stdout.write('\r[   - ]')
            sys.stdout.flush()
            time.sleep(sleep)
            sys.stdout.write('\r[  -  ]')
            sys.stdout.flush()
            time.sleep(sleep)
            sys.stdout.write('\r[ -   ]')
            sys.stdout.flush()
            time.sleep(sleep)
