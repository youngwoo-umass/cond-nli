import numpy as np
import tensorflow as tf


# CPT: Conditional Probability Table
def cpt_combine2(local_decisions):
    cpt_discrete = tf.constant([[0, 1, 2],
                                [1, 1, 2],
                                [2, 2, 2]])

    local_decision_a = local_decisions[:, 0]
    local_decision_b = local_decisions[:, 1]  # [B, 3]

    cpt = tf.one_hot(cpt_discrete, 3)  # [3, 3, 3]   axis 2 is one hot

    left = tf.expand_dims(tf.expand_dims(local_decision_a, 2), 3)  # [B, 3, 1, 1]
    right = tf.expand_dims(cpt, axis=0)  # [1, 3, 3, 3]
    t = tf.multiply(left, right)
    res1 = tf.reduce_sum(t, axis=1)  # [B, 3, 3]

    left = tf.expand_dims(local_decision_b, axis=2)  # [B, 3, 1]
    right = res1
    t = tf.multiply(left, right)
    result = tf.reduce_sum(t, axis=1)  # [B, 3]
    return result


class MatrixCombine(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return cpt_combine2(inputs)


def numpy_one_hot(a, depth):
    output = (np.arange(depth) == a[..., None]).astype(int)
    return output


def numpy_test():
    init_val_dis = [[0, 1, 2],
                    [1, 1, 2],
                    [2, 2, 2]]
    a = np.array(init_val_dis)
    output = numpy_one_hot(a, 3)
    print(output)
    print(output.shape)

    for i in range(3):
        for j in range(3):
            c = init_val_dis[i][j]
            for k in range(3):
                if k == c:
                    assert output[i, j, k] == 1
                else:
                    assert output[i, j, k] == 0


if __name__ == "__main__":
    numpy_test()
