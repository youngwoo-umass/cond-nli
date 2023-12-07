import csv
import os
from typing import List
from typing import NamedTuple

from cpath import data_path
from utils.download_util import download_file


class CondNLISentPair(NamedTuple):
    pair_id: str
    group_no: int
    text1: str
    text2: str


class CondNLISentPairWithLabel(NamedTuple):
    pair_id: str
    group_no: int
    text1: str
    text2: str
    sent1_contradiction: List[int]  # 0 or 1
    sent1_neutral: List[int]  # 0 or 1
    sent2_contradiction: List[int]  # 0 or 1
    sent2_neutral: List[int]  # 0 or 1

    def get_label_by_name(self, label_name):
        return {
            "sent1_contradiction": self.sent1_contradiction,
            "sent1_neutral": self.sent1_neutral,
            "sent2_contradiction": self.sent2_contradiction,
            "sent2_neutral": self.sent2_neutral
        }[label_name]


def get_url(split):
    return f"https://raw.githubusercontent.com/youngwoo-umass/cond-nli/main/BioClaim/{split}.csv"


def load_bioclaim_as_dict_list(split):
    csv_path = os.path.join(data_path, "BioClaim", split + ".csv")
    if not os.path.exists(csv_path):
        download_file(get_url(split), csv_path)

    reader = csv.reader(open(csv_path, "r"))

    rows = list(reader)
    column_names = rows[0]

    dict_list = []
    for row in rows[1:]:
        d = {}
        for idx, value in enumerate(row):
            column = column_names[idx]
            d[column] = value
        dict_list.append(d)
    return dict_list


def load_bioclaim_problems(split) -> List[CondNLISentPair]:
    dict_list = load_bioclaim_as_dict_list(split)

    data = []
    for d in dict_list:
        pair = CondNLISentPair(d['pair_id'], int(d['group_no']), d['sent1'], d['sent2'])
        data.append(pair)

    return data


def load_bioclaim_problems_w_labels(split) -> List[CondNLISentPairWithLabel]:
    dict_list = load_bioclaim_as_dict_list(split)

    def parse_label(s):
        return list(map(int, s.split()))

    data = []
    for d in dict_list:
        pair = CondNLISentPairWithLabel(
            d['pair_id'], int(d['group_no']), d['sent1'], d['sent2'],
            parse_label(d["sent1_contradiction"]),
            parse_label(d["sent1_neutral"]),
            parse_label(d["sent2_contradiction"]),
            parse_label(d["sent2_neutral"]),
            )
        data.append(pair)
    return data

