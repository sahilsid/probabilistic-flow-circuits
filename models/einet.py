from models.base import TractableModel
from components.spn.ExponentialFamilyArray import NormalArray 

class EinsumNet(TractableModel):
    def __init__(self, config):
        self.leaf_distribution = NormalArray
        super().__init__(config)