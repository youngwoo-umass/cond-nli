import os.path

import tensorflow as tf
from tensorflow import keras

from utils.download_util import download_and_unpack_zip
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.modeling_common.bert_common import define_bert_input, BERT_CLS, load_bert_checkpoint
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF

from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input, get_shape_list2


def split_stack_flatten_encode_stack(encoder, input_list,
                                     total_seq_length, window_length):
    num_window = int(total_seq_length / window_length)
    assert total_seq_length % window_length == 0
    batch_size, _ = get_shape_list2(input_list[0])

    def r3to2(arr):
        return tf.reshape(arr, [batch_size * num_window, window_length])

    input_list_stacked = split_stack_input(input_list, total_seq_length, window_length)
    input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
    rep_flatten = encoder(input_list_flatten)  # [batch_size * num_window, dim]
    _, rep_dim = get_shape_list2(rep_flatten)

    def r2to3(arr):
        return tf.reshape(arr, [batch_size, num_window, rep_dim])

    rep_stacked = r2to3(rep_flatten)
    return rep_stacked


class TwoSegConcat2(BertBasedModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(TwoSegConcat2, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfigType):
        num_window = 2
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        bert_cls = BERT_CLS(l_bert, pooler)
        num_classes = config.num_classes
        max_seq_length = config.max_seq_length
        l_input_ids, l_token_type_ids = define_bert_input(max_seq_length, "")

        # [batch_size, dim]
        window_length = int(max_seq_length / num_window)
        inputs = [l_input_ids, l_token_type_ids]
        feature_rep = split_stack_flatten_encode_stack(
            bert_cls.apply, inputs,
            max_seq_length, window_length)

        B, _ = get_shape_list2(l_input_ids)
        # [batch_size, num_window, dim2 ]
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        comb_layer = self.combine_local_decisions_layer()
        output = comb_layer(local_decisions)
        inputs = (l_input_ids, l_token_type_ids)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls = bert_cls
        self.l_bert = l_bert
        self.pooler = pooler

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        checkpoint_dir = os.path.dirname(init_checkpoint)
        if not os.path.exists(checkpoint_dir):
            c_log.info("Checkpoint do not exists download BERT-base.")
            bert_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
            save_dir = os.path.dirname(checkpoint_dir)
            download_and_unpack_zip(bert_url, save_dir)
        load_bert_checkpoint(self.bert_cls, init_checkpoint)


