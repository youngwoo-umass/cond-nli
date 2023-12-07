import random
from abc import ABC, abstractmethod
from typing import Tuple, List

from data_generator.bert_sequence_util import get_basic_input_feature_as_list, combine_with_sep_cls, concat_triplet_windows


def encode_single(tokenizer, tokens, max_seq_length):
    effective_length = max_seq_length - 2
    tokens = tokens[:effective_length]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                         tokens, segment_ids)

    return input_ids, input_mask, segment_ids


class PairEncoderInterface(ABC):
    @abstractmethod
    def encode_from_text(self, text1, text2) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        pass



def get_random_split_location(tokens) -> Tuple[int, int]:
    retry = True
    n_retry = 0
    while retry:
        st = random.randint(0, len(tokens) - 1)
        while 0 <= st < len(tokens) - 1 and tokens[st].startswith("##"):
            st += 1

        # st is located at end of the text
        if st + 1 > len(tokens) and n_retry < 4:
            n_retry += 1
            retry = True
            continue

        ed = random.randint(st+1, len(tokens))
        retry = False
        return st, ed


def random_token_split(tokens):
    st, ed = get_random_split_location(tokens)
    first_a = tokens[:st]
    first_b = tokens[ed:]
    first = first_a + ["[MASK]"] + first_b
    second = tokens[st:ed]
    return first, second


class TwoSegConcatEncoder(PairEncoderInterface):
    def __init__(self, tokenizer, total_max_seq_length):
        segment_len = int(total_max_seq_length / 2)
        self.segment_len = segment_len

        self.tokenizer = tokenizer
        if total_max_seq_length % 2:
            raise ValueError()

    def encode(self, tokens1, tokens2) -> Tuple[List, List, List]:
        tokens2_first, tokens2_second = random_token_split(tokens2)
        return self.two_seg_concat_core(tokens1, tokens2_first, tokens2_second)

    def two_seg_concat_core(self, tokens1, tokens2_first, tokens2_second) -> Tuple[List, List, List]:
        triplet_list = []
        for part_of_tokens2 in [tokens2_first, tokens2_second]:
            tokens, segment_ids = combine_with_sep_cls(self.segment_len, tokens1, part_of_tokens2)
            triplet = get_basic_input_feature_as_list(self.tokenizer, self.segment_len,
                                                      tokens, segment_ids)
            triplet_list.append(triplet)
        return concat_triplet_windows(triplet_list, self.segment_len)

    def encode_from_text(self, text1, text2) -> Tuple[List, List, List]:
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))


