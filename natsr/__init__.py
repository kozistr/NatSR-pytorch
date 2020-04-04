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


@unique
class AdvLossType(str, Enum):
    GAN = 'gan'
    LSGAN = 'lsgan'
    RAGAN = 'ragan'
    WGAN = 'wgan'
    WGANGP = 'wgan-gp'


@unique
class ClsLossType(str, Enum):
    BCE = 'bce'
    CCE = 'cce'


@unique
class RecLossType(str, Enum):
    L1 = 'l1'
    L2 = 'l2'


@unique
class LRSchedulerType(str, Enum):
    EXPONENTIAL = 'exponential'
