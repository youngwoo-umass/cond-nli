import os
from typing import NamedTuple, Iterator

from cpath import data_path
from data_generator.NLI.enlidef import nli_label_list
from utils.download_util import download_and_unpack_zip

corpus_dir = os.path.join(data_path, "multinli_1.0")

from data_generator.record_writer_wrap import write_records_w_encode_fn


class NLIPairData(NamedTuple):
    premise: str
    hypothesis: str
    label: str
    data_id: str

    def get_label_as_int(self):
        return nli_label_list.index(self.label)






class MNLIReader:
    def __init__(self):
        if not os.path.exists(corpus_dir):
            url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
            download_and_unpack_zip(url, corpus_dir)

        prefix = "multinli_1.0_"
        self.train_file = os.path.join(corpus_dir, prefix + "train.txt")
        self.dev_file = os.path.join(corpus_dir, prefix + "dev_matched.txt")
        self.split_file_path = {
            'train': self.train_file,
            'dev': self.dev_file
        }

    def get_train(self) -> Iterator[NLIPairData]:
        return self.load_split("train")

    def get_dev(self) -> Iterator[NLIPairData]:
        return self.load_split("dev")

    def load_split(self, split_name) -> Iterator[NLIPairData]:
        f = open(self.split_file_path[split_name], "r", encoding="utf-8")

        head = f.readline()
        columns = head.strip().split("\t")
        idx_label = columns.index("gold_label")
        idx_sent1 = columns.index("sentence1")
        idx_sent2 = columns.index("sentence2")
        idx_pair = columns.index("pairID")
        n_missing = 0
        for line in f:
            try:
                row = line.strip().split("\t")
                e = NLIPairData(
                    premise=row[idx_sent1],
                    hypothesis=row[idx_sent2],
                    label=row[idx_label],
                    data_id=row[idx_pair]
                )
                e.get_label_as_int()
                yield e
            except Exception as e:
                n_missing += 1

        print("Failed to parse {} items".format(n_missing))


    def get_data_size(self, split_name):
        data_size = 400 * 1000 if split_name == "train" else 9815
        return data_size


def mnli_encode_common(encode_fn, split, output_path):
    data_size = 400 * 1000 if split == "train" else 10000
    reader = MNLIReader()
    write_records_w_encode_fn(output_path, encode_fn, reader.load_split(split), data_size)
