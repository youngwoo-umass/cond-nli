import tensorflow as tf

from trainer_v2.chair_logging import c_log
import atexit


def device_list_summary(device_list):
    if not device_list:
        return "No device found"
    n_gpu = 0
    name_set = set()
    for dev in device_list:
        if dev.device_type == 'GPU':
            if dev.name in name_set:
                c_log.warn("Duplicated name {}".format(dev.name))
            name_set.add(dev.name)
            n_gpu += 1
    if n_gpu == len(device_list):
        return "{} GPUs found".format(n_gpu)
    else:
        return str(device_list)


def get_strategy2():
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    gpu_devices = tf.config.list_logical_devices('GPU')
    c_log.info(device_list_summary(gpu_devices))
    try:
        atexit.register(strategy._extended._cross_device_ops._pool.close)  # type: ignore
        atexit.register(strategy._extended._host_cross_device_ops._pool.close)  # type: ignore
    except AttributeError:
        pass
    return strategy

