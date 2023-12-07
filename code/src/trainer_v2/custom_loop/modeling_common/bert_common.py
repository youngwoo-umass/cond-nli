import re
from typing import NamedTuple

import tensorflow as tf
from trainer_v2.bert_for_tf2.model import BertModelLayer
from trainer_v2.bert_for_tf2.loader import map_to_stock_variable_name, bert_prefix, map_stock_config_to_params, \
    StockBertConfig
from tensorflow import keras

from trainer_v2.chair_logging import c_log


class BERT_CLS(NamedTuple):
    l_bert: tf.keras.layers.Layer
    pooler: tf.keras.layers.Dense

    def apply(self, inputs):
        seq_out = self.l_bert(inputs)
        cls = self.pooler(seq_out[:, 0, :])
        return cls


def load_stock_weights(bert: BertModelLayer, ckpt_path,
                       map_to_stock_fn=map_to_stock_variable_name,
                       n_expected_restore=None,
                       ):

    assert len(bert.weights) > 0, "BertModelLayer weights have not been instantiated yet. " \
                                  "Please add the layer in a Keras model and call model.build() first!"

    skipped_weight_value_tuples = _load_stock_weights(bert, ckpt_path, map_to_stock_fn, n_expected_restore)

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)


def _load_stock_weights(bert, ckpt_path, map_to_stock_fn, n_expected_restore):
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)
    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())
    prefix = bert_prefix(bert)
    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []
    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_stock_fn(param.name, prefix)
        c_log.debug(param.name)

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)

            if param_value.shape != ckpt_value.shape:
                c_log.warn("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                           "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                      stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            ("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)
    if n_expected_restore is None or n_expected_restore == len(weight_value_tuples):
        pass
    else:
        c_log.warn("Done loading {} BERT weights from: {} into {} (prefix:{}). "
                   "Count of weights not found in the checkpoint was: [{}]. "
                   "Count of weights with mismatched shape: [{}]".format(
            len(weight_value_tuples), ckpt_path, bert, prefix, skip_count, len(skipped_weight_value_tuples)))

        c_log.warn("Unused weights from checkpoint: %s",
                   "\n\t" + "\n\t".join(sorted(stock_weights.difference(loaded_weights))))
        raise ValueError("Checkpoint load exception")
    return skipped_weight_value_tuples


def pooler_mapping(name, prefix="bert"):
    name = name.split(":")[0]
    ns = name.split("/")
    pns = prefix.split("/")

    if ns[:len(pns)] != pns:
        return None

    name = "/".join(["bert"] + ns[len(pns):])
    return name


def load_pooler(pooler: tf.keras.layers.Dense, ckpt_path):
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param pooler: a dense layer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)
    loaded_weights = set()

    re_bert = re.compile(r'(.*)/(pooler)/(.+):0')
    match = re_bert.match(pooler.weights[0].name)
    assert match, "Unexpected bert layer: {} weight:{}".format(pooler, pooler.weights[0].name)
    prefix = match.group(1)
    skip_count = 0
    weight_value_tuples = []
    bert_params = pooler.weights
    for ndx, (param) in enumerate(bert_params):
        name = pooler_mapping(param.name, prefix)
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            stock_name = m.group(1)
        else:
            stock_name = name

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)
            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("{} not found".format(stock_name))
            skip_count += 1

    assert len(loaded_weights) == 2
    keras.backend.batch_set_value(weight_value_tuples)


# cls/predictions/transform/dense/kernel
# cls/predictions/transform/dense/bias
# cls/predictions/transform/LayerNorm/gamma
# cls/predictions/transform/LayerNorm/beta
# cls/predictions/output_bias


def load_bert_config(bert_config_file):
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        is_brightmart_weights = bc["ln_type"] is not None
        bert_params.project_position_embeddings = not is_brightmart_weights  # ALBERT: False for brightmart/weights
        bert_params.project_embeddings_with_bias = not is_brightmart_weights  # ALBERT: False for brightmart/weights
    return bert_params


def load_bert_checkpoint(bert_cls, checkpoint_path):
    load_stock_weights(bert_cls.l_bert, checkpoint_path, n_expected_restore=197)
    load_pooler(bert_cls.pooler, checkpoint_path)


def define_bert_input(max_seq_len, post_fix=""):
    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids{}".format(post_fix))
    l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids{}".format(post_fix))
    return l_input_ids, l_token_type_ids
