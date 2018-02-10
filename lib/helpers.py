"""
Misc. utilities
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from string import punctuation
from time import gmtime, strftime
import sys
from pathlib2 import Path

import numpy as np


def xywh_to_xyxy(box):
    """BBox transform"""
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return np.array([box[0], box[1], x2, y2])


def get_or_create_path(path_string, parents=True):
    """ Get path object from string, create if non-existing"""
    p = Path(path_string)
    if not p.exists():
        p.mkdir(parents=parents)
    return p


class Timer(object):
    """
    Simple timer tool
    """
    def __init__(self, history=500):
        self._time = 0.
        self._total_time = 0.
        self._total_calls = 0.

        self._history = history

    def tic(self):
        self._time = time.time()

    def toc(self):
        time_passed = time.time() - self._time
        self._total_time += time_passed
        self._total_calls += 1
        return time_passed

    def average(self):
        av = 0
        if self._total_calls > 0:
            av = self._total_time / self._total_calls
        if self._total_calls > self._history:
            self._total_calls = 0
            self._total_time = 0
        return av


class RunningAverages(object):
    def __init__(self, num_of_averages, max_length=100):
        self._averages = [RunningAverage(length=max(max_length, 100)) for _ in range(num_of_averages)]

    def __call__(self, *args, **kwargs):
        return self._averages

    def update(self, values):
        assert isinstance(values, (list, tuple)), 'values to average must be a list or a tuple'
        assert len(values) == len(self._averages)
        for l, av in zip(values, self._averages):
            av.update(l)


class RunningAverage(object):
    """
    Calculate a running average of a variable
    """

    def __init__(self, length=None):
        self._num_steps = 0
        self._sum_values = 0.

        self._length = length

    def __call__(self):
        if self._num_steps < 1:
            return 0.
        return self._sum_values / self._num_steps

    def update(self, x):

        if self._length and self._num_steps >= self._length:
            self._num_steps = 0
            self._sum_values = 0
        self._num_steps += 1
        self._sum_values += float(x)

    def reset(self):
        self.__init__()


class Logger(object):
    _log_file_name = 'log.txt'

    def __init__(self, log_dir):
        self._log_file = self._file_from_path(log_dir)
        self._orig_output = sys.stdout
        self._lines_written = 0
        return

    def __call__(self, print_string, show=True):
        sys.stdout = self._log_file
        log_string = '%s  %s' % (_log_time_stamp(), str(print_string))
        print (log_string)
        self._log_file.flush()
        sys.stdout = self._orig_output
        self._lines_written += 1

        if show:
            print(print_string)

    def _file_from_path(self, log_dir):
        log_file = Path(log_dir) / self._log_file_name
        file_obj = _check_and_get_file_obj(log_file)
        return file_obj

    def close(self):
        self.__call__('Finished Logging %d lines written' % self._lines_written)
        self._log_file.close()


def dense_time_stamp():
    return strftime("%d_%m_%Y_%H%M%S", gmtime())


def _log_time_stamp():
    return strftime("[%d-%m-%Y-%H:%M:%S]", gmtime())


def _check_and_get_wd_path_obj(wd):
    p = Path(wd)
    if not p.exists():
        p.mkdir(parents=True)
    return p


def _check_and_get_file_obj(fpath):
    p = Path(fpath)
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    if p.is_file():
        return p.open('ab')
    return p.open('wb')


def clean_word(word):
    """Remove punctuation marks and redundant spaces from a word"""
    exclude = set(punctuation)
    s = ''.join(ch for ch in word if ch not in exclude)
    s = s.strip().lower()
    return s


class TensorBoardFiles(object):

    LOGS = 'logs'

    def __init__(self, experiment_dir, prefix, sess):
        import tensorflow as tf
        if prefix:
            summary_writer = tf.summary.FileWriter(
                str(get_or_create_path(path_string=experiment_dir / self.LOGS / ('%s_%s' % (prefix, dense_time_stamp())))), sess.graph)
            fake_summary_writer = tf.summary.FileWriter(
                str(get_or_create_path(path_string=experiment_dir / self.LOGS / ('fake_%s_%s' % (prefix, dense_time_stamp())))), sess.graph)
        else:
            summary_writer = tf.summary.FileWriter(str(get_or_create_path(path_string=experiment_dir / 'logs' / dense_time_stamp())), sess.graph)
            fake_summary_writer = tf.summary.FileWriter(
                str(get_or_create_path(path_string=experiment_dir / self.LOGS / ('fake_%s' % dense_time_stamp()))), sess.graph)

        self.real = summary_writer
        self.fake = fake_summary_writer

