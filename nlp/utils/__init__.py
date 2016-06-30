import gzip
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys
import os
import errno
import datetime
from progressbar import ProgressBar, Percentage, UnknownLength, \
                        Bar, SimpleProgress, Timer, Counter, ETA


class PreciseTimer(Timer):
    """Widget which displays the elapsed seconds."""

    __slots__ = ('format_string',)
    TIME_SENSITIVE = True

    def __init__(self, format='Elapsed Time: %s'):
        self.format_string = format

    @staticmethod
    def format_time(seconds):
        """Formats time as the string "HH:MM:SS"."""

        return str(datetime.timedelta(seconds=seconds))


    def update(self, pbar):
        """Updates the widget to show the elapsed time."""

        return self.format_string % self.format_time(pbar.seconds_elapsed)


class PreciseETA(ETA):
    """Widget which attempts to estimate the time of arrival."""

    def update(self, pbar):
        """Updates the widget to show the ETA or total time when finished."""

        if pbar.currval == 0:
            return 'ETA:  --:--:--'
        elif pbar.finished:
            return 'Time: %s' % PreciseTimer.format_time(pbar.seconds_elapsed)
        else:
            elapsed = pbar.seconds_elapsed
            eta = elapsed * pbar.maxval / pbar.currval - elapsed
            return 'ETA:  %s' % PreciseTimer.format_time(eta)


class IterTimer(object):
    def __init__(self, name="", total=UnknownLength, period=1, verbose=3):
        """
        Args:
            verbose: 0: no output, do nothing
                     1: only output in start
                     2: only output in start and end
                     3: output in progresss
        """
        self.verbose = verbose
        self.period = period
        if verbose >= 2:
            widgets = ["...", name]
            if total is not UnknownLength:
                widgets.extend([" ", Bar()," ", SimpleProgress("/"), " (", Percentage(), ")  ", PreciseETA()])
            else:
                widgets.extend([" (", PreciseTimer(), ")"])
            self.pbar = ProgressBar(widgets=widgets, maxval=total)
        elif verbose >= 1:
            print("..." + name)

    def __enter__(self):
        if self.verbose >= 2:
            self.pbar.start()
            if self.verbose == 2:
                print("")
        return self

    def __exit__(self, type, value, traceback):
        if self.verbose >= 2:
            self.pbar.widgets.insert(2, " done")
            self.pbar.finish()

    def update(self, i, total=None):
        if i % self.period == 0:
            if self.verbose >= 2 and total is not None:
                self.pbar.maxval = total
            if self.verbose >=3:
                self.pbar.update(i)

def test_iter_timer():
    import time
    with IterTimer("Testing", 100) as timer:
        for i in xrange(100):
            timer.update(i)
            time.sleep(0.01)


def save_pklgz(obj, filename, verbose=0):
    with IterTimer("Pickling to %s" % (filename), verbose=verbose):
        pkl = pickle.dumps(obj, protocol = pickle.HIGHEST_PROTOCOL);
        with gzip.open(filename, "wb") as fp:
            fp.write(pkl);

def load_pklgz(filename, verbose=0):
    with gzip.open(filename, 'rb') as fp, IterTimer("Unpickling from %s" % (filename), verbose=verbose):
        obj = pickle.load(fp);
    return obj;

def save_pkl(obj, filename, verbose=0):
    with open(filename, "wb") as fp, IterTimer("Pickling to %s" % (filename), verbose=verbose):
        pickle.dump(obj, fp, protocol = pickle.HIGHEST_PROTOCOL);

def load_pkl(filename, verbose=0):
    with open(filename, "rb") as fp, IterTimer("Unpickling from %s" % (filename), verbose=verbose):
        obj = pickle.load(fp);
    return obj

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
