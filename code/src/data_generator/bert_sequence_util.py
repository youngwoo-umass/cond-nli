from typing import List


def get_basic_input_feature_as_list(tokenizer, max_seq_length, input_tokens, segment_ids):
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    return get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_length)


def get_basic_input_feature_as_list_all_ids(input_ids, segment_ids, max_seq_length):
    input_mask = [1] * len(input_ids)
    segment_ids = list(segment_ids)
    max_seq_length = max_seq_length
    assert len(input_ids) <= max_seq_length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids


def combine_with_sep_cls(max_seq_length, tokens1, tokens2):
    max_seg2_len = max_seq_length - 3 - len(tokens1)
    tokens2 = tokens2[:max_seg2_len]
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) \
                  + [1] * (len(tokens2) + 1)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    return tokens, segment_ids


def combine_with_sep_cls2(max_seq_length, tokens1, tokens2):
    max_seg1_len = max_seq_length - 3 - len(tokens2)
    tokens1 = tokens1[:max_seg1_len]
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids = [0] * (len(tokens1) + 2) \
                  + [1] * (len(tokens2) + 1)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    return tokens, segment_ids



def concat_triplet_windows(triplet_iterator, window_length=None):
    all_input_ids: List[int] = []
    all_input_mask: List[int] = []
    all_segment_ids: List[int] = []
    for input_ids, input_mask, segment_ids in triplet_iterator:
        all_input_ids.extend(input_ids)
        all_input_mask.extend(input_mask)
        all_segment_ids.extend(segment_ids)
        if window_length is not None:
            assert len(input_ids) == window_length
            assert len(input_mask) == window_length
            assert len(segment_ids) == window_length

    return all_input_ids, all_input_mask, all_segment_ids
