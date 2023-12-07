from typing import Callable, List, TypeVar

import tensorflow as tf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.run_config2 import RunConfig2


def parse_file_path(input_file):
    input_files = []
    for input_pattern in input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files


def create_dataset_common(decode_record: Callable,
                          run_config: RunConfig2,
                          file_path: str,
                          is_training_split: bool):
    do_shuffle = is_training_split and run_config.train_config.do_shuffle
    do_repeat = is_training_split
    config = run_config.dataset_config
    batch_size = run_config.common_run_config.batch_size
    if not is_training_split:
        if run_config.common_run_config.eval_batch_size is not None:
            batch_size = run_config.common_run_config.eval_batch_size
    input_files: List[str] = parse_file_path(file_path)
    if len(input_files) > 1:
        c_log.info("{} inputs files".format(len(input_files)))
    elif len(input_files) == 0:
        c_log.error("No input files found - Maybe you dont' want this ")
        raise FileNotFoundError(input_files)

    num_parallel_reads = min(len(input_files), 4)
    dataset = tf.data.TFRecordDataset(input_files, num_parallel_reads=num_parallel_reads)
    if do_shuffle:
        dataset = dataset.shuffle(config.shuffle_buffer_size)
    if do_repeat:
        dataset = dataset.repeat()
    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_classification_dataset(file_path,
                               run_config: RunConfig2,
                               model_config: ModelConfigType,
                               is_for_training,
                               ) -> tf.data.Dataset:
    seq_length = model_config.max_seq_length

    def select_data_from_record(record):
        for k, v in record.items():
            record[k] = tf.cast(v, tf.int32)
        entry = (record['input_ids'], record['segment_ids']), record['label_ids']
        return entry

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        record = tf.io.parse_single_example(record, name_to_features)
        return select_data_from_record(record)

    return create_dataset_common(decode_record, run_config,
                                 file_path, is_for_training)












