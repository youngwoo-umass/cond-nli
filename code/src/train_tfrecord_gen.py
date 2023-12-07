import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator.NLI.mnli_common import mnli_encode_common
from data_generator.seg_encoder_common import TwoSegConcatEncoder
from data_generator.segmented_tfrecord_gen import get_encode_fn_from_encoder
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig600_3
from utils.misc_lib import make_parent_exists


def gen_mnli_concat_two_seg(data_name, split):
    output_dir = at_output_dir("tfrecord", data_name)
    save_path = os.path.join(output_dir, split)
    make_parent_exists(save_path)
    tokenizer = get_tokenizer()

    model_config = ModelConfig600_3()
    encoder = TwoSegConcatEncoder(tokenizer, model_config.max_seq_length)
    encode_fn = get_encode_fn_from_encoder(encoder)
    mnli_encode_common(encode_fn, split, save_path)


def main():
    c_log.info("Generate Train data for MultiNLI training")
    data_name = "nli_concat"
    for split in ["dev", "train"]:
        gen_mnli_concat_two_seg(data_name, split)


if __name__ == "__main__":
    main()
