import tensorflow as tf
from collections import OrderedDict
from typing import Dict

from data_generator.NLI.mnli_common import NLIPairData
from data_generator.seg_encoder_common import PairEncoderInterface


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def get_encode_fn_from_encoder(encoder: PairEncoderInterface):
    def entry_encode(e: NLIPairData) -> Dict:
        features = OrderedDict()
        input_ids, input_mask, segment_ids = encoder.encode_from_text(e.premise, e.hypothesis)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features['label_ids'] = create_int_feature([e.get_label_as_int()])
        return features

    return entry_encode
