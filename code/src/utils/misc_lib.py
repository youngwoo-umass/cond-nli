import json
import os
import time
from typing import TypeVar
from typing import List, Iterable, Callable, Dict, Tuple, Set

A = TypeVar('A')
B = TypeVar('B')


def average(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


class TimeEstimator:
    def __init__(self, total_repeat, name="", sample_size=10):
        self.time_analyzed = None
        self.time_count = 0
        self.total_repeat = total_repeat
        if sample_size == 10:
            if self.total_repeat > 1000000:
                sample_size = 1000
            if self.total_repeat > 10000:
                sample_size = 100
        self.name = name
        self.base = 3
        self.sample_size = sample_size
        self.progress_tenth = 1

    def tick(self):
        self.time_count += 1
        if not self.time_analyzed:
            if self.time_count == self.base:
                self.time_begin = time.time()

            if self.time_count == self.base + self.sample_size:
                elapsed = time.time() - self.time_begin
                expected_sec = elapsed / self.sample_size * self.total_repeat
                expected_min = int(expected_sec / 60)
                print("Expected time for {} : {} min".format(self.name, expected_min))
                self.time_analyzed = True
        if self.total_repeat * self.progress_tenth / 10 < self.time_count:
            print("{}0% completed".format(self.progress_tenth))
            self.progress_tenth += 1


class TimeEstimatorOpt(TimeEstimator):
    def __init__(self, total_repeat, name="", sample_size=10):
        self.total_repeat = total_repeat
        if total_repeat is not None:
            super(TimeEstimatorOpt, self).__init__(total_repeat, name, sample_size)

    def tick(self):
        if self.total_repeat is not None:
            super().tick()


def exist_or_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


class SimpleMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get_average(self):
        if len(self.values) == 0:
            return None
        return sum(self.values) / len(self.values)

    def append(self, average, n_item):
        # Append the provided average value n_item times
        for _ in range(n_item):
            self.update(average)


# returns dictionary where key is the element in the iterable and the value is the func(key)


# returns dictionary where value is the func(value) of input dictionary


# Get direct child file (not directory)


class BinAverage:
    def __init__(self, bin_fn):
        self.list_dict = {}
        self.bin_fn = bin_fn

    def add(self, k, v):
        bin_id = self.bin_fn(k)
        if bin_id not in self.list_dict:
            self.list_dict[bin_id] = []

        self.list_dict[bin_id].append(v)

    def all_average(self):
        output = {}
        for k, v in self.list_dict.items():
            output[k] = average(v)
        return output


class IntBinAverage(BinAverage):
    def __init__(self):
        super(IntBinAverage, self).__init__(lambda x: int(x))


def ceil_divide(denom: int, nom: int) -> int:
    return int((denom + (nom - 1)) / nom)


# --- Dict related methods BEGIN ---


# --- Dict related methods END ---


bw_obj = None


class CountWarning:
    def __init__(self, warning_name="Warning case", interval=100):
        self.interval = interval
        self.cnt = 0
        self.warning_name = warning_name

    def add_warn(self):
        self.cnt += 1
        if self.cnt % self.interval == 0:
            print("{} reached {}".format(self.warning_name, self.cnt))


def group_by(interable: Iterable[A], key_fn: Callable[[A], B]) -> Dict[B, List[A]]:
    """

    :rtype: object
    """
    grouped = {}
    for elem in interable:
        key = key_fn(elem)
        if key not in grouped:
            grouped[key] = list()

        grouped[key].append(elem)
    return grouped


class RecentCounter:
    def __init__(self, interval, last_idx=None):
        self.last_idx = last_idx
        self.interval = interval

    def update_last(self, idx):
        self.last_idx = idx

    def is_over_interval(self, idx):
        if self.last_idx is None:
            self.update_last(idx)
            return True
        elapsed = idx - self.last_idx
        if elapsed < self.interval:
            return False
        else:
            self.update_last(idx)
            return True

def make_parent_exists(target_path):
    def make_dir_parent_exists(target_dir):
        parent_path = os.path.dirname(target_dir)
        if not os.path.exists(parent_path):
            make_dir_parent_exists(parent_path)
        exist_or_mkdir(target_dir)

    parent_path = os.path.dirname(target_path)
    make_dir_parent_exists(parent_path)


def save_as_jsonl(save_j_list, save_path):
    with open(save_path, "w") as f:
        for j in save_j_list:
            f.write(json.dumps(j) + "\n")


def load_jsonl(save_path):
    with open(save_path, "r") as f:
        return list(map(json.loads, f))
