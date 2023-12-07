import abc
import dataclasses


@dataclasses.dataclass
class ModelConfigType:
    __metaclass__ = abc.ABCMeta
    max_seq_length = abc.abstractproperty()
    num_classes = abc.abstractproperty()


class ModelConfig600_3(ModelConfigType):
    max_seq_length = 600
    num_classes = 3

