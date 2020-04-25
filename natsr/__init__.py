from enum import Enum, unique

CONFIG_FILENAME: str = 'config.yaml'

THRESHOLD_ALPHA: float = 0.8
THRESHOLD_SIGMA: float = 0.0444
THRESHOLD_ACC: float = 0.95


@unique
class Mode(str, Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    INFERENCE = 'inference'


@unique
class ModelType(str, Enum):
    FRSR = 'frsr'
    NMD = 'nmd'


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
    DRAGAN = 'drgan'
    LSGAN = 'lsgan'
    WGAN = 'wgan'
    WGANGP = 'wgan-gp'
    WGANLP = 'wgan-lp'
    HINGE = 'hinge'


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


@unique
class DeviceType(str, Enum):
    CPU = 'cpu'
    GPU = 'cuda'
