from enum import Enum, unique

CONFIG_FILENAME: str = 'config.yaml'


@unique
class Mode(str, Enum):
    TRAIN = 'train'
    TEST = 'test'
    INFERENCE = 'inference'


@unique
class DataType(str, Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


@unique
class DataSets(str, Enum):
    DIV2K = 'div2k'


@unique
class OptimizerType(str, Enum):
    ADAM = 'adam'
    SGD = 'sgd'
