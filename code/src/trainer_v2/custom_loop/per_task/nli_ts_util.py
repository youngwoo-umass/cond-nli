import functools
from abc import ABC, abstractmethod
from typing import List, Tuple

import tensorflow as tf

from data_generator.tokenizer_wo_tf import sb_tokenize_w_tokenizer
from utils.misc_lib import ceil_divide
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset


class EncodedSegmentIF(ABC):
    @abstractmethod
    def get_input(self):
        pass


class LocalDecisionNLICore:
    def __init__(self, model, strategy, encode_fn, batch_size=16):
        self.strategy = strategy
        self.model = model
        self.n_input = len(self.model.inputs)
        self.l_list: List[int] = [input_unit.shape[1] for input_unit in self.model.inputs]
        self.encode_fn = encode_fn
        self.batch_size = batch_size

        def get_spec(input_i: tf.keras.layers.Input):
            return tf.TensorSpec(input_i.type_spec.shape[1:], dtype=tf.int32)
        self.out_sig = [get_spec(input) for input in model.inputs]

    def predict(self, input_list: List[Tuple]):
        batch_size = self.batch_size

        while len(input_list) % batch_size:
            input_list.append(input_list[-1])

        def generator():
            yield from input_list

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tuple(self.out_sig))
        strategy = self.strategy

        def reform(*row):
            # return (*row),
            if self.n_input == 4:
                x = row[0], row[1], row[2], row[3]
            elif self.n_input == 2:
                x = row[0], row[1],
            else:
                raise ValueError
            return x,

        dataset = dataset.map(reform)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        maybe_step = ceil_divide(len(input_list), batch_size)
        dataset = distribute_dataset(strategy, dataset)
        model = self.model
        verbose = 1 if maybe_step > 15 else 0
        l_decision, g_decision = model.predict(dataset, steps=maybe_step, verbose=verbose)
        return l_decision, g_decision

    def predict_es(self, input_list: List[EncodedSegmentIF]):
        payload = [x.get_input() for x in input_list]
        l_decision_list, g_decision_list = self.predict(payload)
        real_input_len = len(input_list)
        l_decision_list = l_decision_list[:real_input_len]
        second_l_decision = [d[1] for d in l_decision_list]
        return second_l_decision










def enum_hypo_token_tuple_from_tokens(tokenizer, space_tokenized_tokens, window_size, offset=0) -> \
        List[Tuple[List[str], List[str], int, int]]:
    st = offset
    sb_tokenize = functools.partial(sb_tokenize_w_tokenizer, tokenizer)

    while st < len(space_tokenized_tokens):
        ed = st + window_size
        first_a = space_tokenized_tokens[:st]
        second = space_tokenized_tokens[st:ed]
        first_b = space_tokenized_tokens[ed:]

        first = sb_tokenize(first_a) + ["[MASK]"] + sb_tokenize(first_b)
        second = sb_tokenize(second)
        yield first, second, st, ed
        st += window_size


def enum_hypo_token_wmask(sb_tokenize, space_tokenized_tokens, window_size, offset=0) -> \
        List[Tuple[List[str], List[int], int, int]]:
    st = offset

    while st < len(space_tokenized_tokens):
        ed = st + window_size
        first_a = space_tokenized_tokens[:st]
        second = space_tokenized_tokens[st:ed]
        first_b = space_tokenized_tokens[ed:]

        first_a_sb = sb_tokenize(first_a)
        first_b_sb = sb_tokenize(first_b)
        second_sb = sb_tokenize(second)
        tokens = first_a_sb + second_sb + first_b_sb
        mask = [0] * len(first_a_sb) + [1] * len(second_sb) + [0] * len(first_b_sb)
        yield tokens, mask, st, ed
        st += window_size




