from .toy_3d import *
from .datasets import *
from .uci import *
from .mnist import MNIST
from .fmnist import FMNIST 

root = f"{os.path.dirname(os.path.abspath(__file__))}/../../data/"

_DATASETS = {
    "HELIX"     :   HELIX,
    "KNOTTED"   :   KNOTTED,
    "BentLISSAJOUS":BentLISSAJOUS,
    "DisjointCIRCLES":DisjointCIRCLES,
    "TwistedEIGHT":TwistedEIGHT,
    "InterlockedCIRCLES":InterlockedCIRCLES,
    "POWER"     :   POWER,
    "HEPMASS"   :   HEPMASS,
    "MINIBOONE" :   MINIBOONE,
    "GAS"       :   GAS,
    "MNIST"     :   MNIST,
    "FMNIST"    :   FMNIST,
}