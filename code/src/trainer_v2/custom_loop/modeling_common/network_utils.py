import tensorflow as tf


def split_stack_input(input_ids_like_list,
                      total_seq_length: int,
                      window_length: int,
                      ):
    # e.g. input_id_like_list[0] shape is [8, 250 * 4],  it return [8 * 4, 250]
    num_window = int(total_seq_length / window_length)
    batch_size, _ = get_shape_list2(input_ids_like_list[0])

    def r2to3(arr):
        return tf.reshape(arr, [batch_size, num_window, window_length])

    return list(map(r2to3, input_ids_like_list))


def get_shape_list2(tensor):
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape