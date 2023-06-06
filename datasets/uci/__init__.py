import os 
root = f"{os.path.dirname(os.path.abspath(__file__))}/../../data/uci/data/"
from .power import POWER
from .gas import GAS
from .hepmass import HEPMASS
from .miniboone import MINIBOONE
# from .moons import MOONS