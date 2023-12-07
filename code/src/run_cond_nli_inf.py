import logging
import sys
from typing import List, Tuple, Callable

import tensorflow as tf

from contradiction.medical_claims.token_tagging.batch_solver_common import BatchSolver
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import NLITSAdapter, AvgReducer
from contradiction.medical_claims.token_tagging.cond_nli_apply_solver import solve_cond_nli
from data_generator.NLI.enlidef import CONTRADICTION, NEUTRAL
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator.seg_encoder_common import random_token_split
from data_generator.bert_sequence_util import combine_with_sep_cls, get_basic_input_feature_as_list, concat_triplet_windows
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.per_task.nli_ts_util import LocalDecisionNLICore
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser

EncoderType = Callable[[List, List, List], Tuple[List, List]]


def get_local_decision_layer_from_model_by_shape(model, n_label=3):
    for idx, layer in enumerate(model.layers):
        try:
            shape = layer.output.shape
            if shape[1] == 2 and shape[2] == n_label:
                c_log.debug("Maybe this is local decision layer: {}".format(layer.name))
                return layer
        except AttributeError:
            print("layer is actually : ", layer)
        except IndexError:
            pass

    c_log.error("Layer not found")
    for idx, layer in enumerate(model.layers):
        c_log.error(idx, layer, layer.output.shape)
    raise KeyError


def load_local_decision_model(model_path, n_label=3):
    model = tf.keras.models.load_model(model_path, compile=False)
    local_decision_layer = get_local_decision_layer_from_model_by_shape(model, n_label)
    new_outputs = [local_decision_layer.output, model.outputs]
    model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return model


class TwoSegConcatEncoder:
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


def get_two_seg_concat_encoder(max_seq_length):
    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, max_seq_length)

    begin = True

    def encode_two_seg_input(p_tokens, h_first, h_second):
        triplet = encoder.two_seg_concat_core(p_tokens, h_first, h_second)
        input_ids, input_mask, segment_ids = triplet
        x = input_ids, segment_ids
        nonlocal begin
        if begin:
            begin = False
        return x

    return encode_two_seg_input


def get_local_decision_nlits_core(run_config: RunConfig2):
    model_path = run_config.get_model_path()
    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        c_log.debug("Loading model from {} ...".format(model_path))
        model = load_local_decision_model(model_path)
        max_seg_length1 = model.inputs[0].shape[1]
        encode_fn: EncoderType = get_two_seg_concat_encoder(max_seg_length1)
        c_log.debug("Done")
        nlits: LocalDecisionNLICore \
            = LocalDecisionNLICore(model,
                                   strategy,
                                   encode_fn,
                                   run_config.common_run_config.batch_size)
    return nlits


def get_batch_solver(run_config: RunConfig2, target_label: int):
    nlits = get_local_decision_nlits_core(run_config)
    adapter = NLITSAdapter(nlits, AvgReducer(target_label))
    solver = BatchSolver(adapter)
    return solver


def main(args):
    c_log.info("Run Cond-NLI Inference")
    c_log.setLevel(logging.DEBUG)

    todo = [
        (NEUTRAL, "neutral"),
        (CONTRADICTION, "contradiction"),
    ]

    for target_label_idx, target_label in todo:
        def solver_factory(run_config):
            solver = get_batch_solver(run_config, target_label_idx)
            return solver

        solve_cond_nli(args, solver_factory, target_label, "val")
        solve_cond_nli(args, solver_factory, target_label, "test")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
