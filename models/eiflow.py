from models.base import TractableModel
from components.spn.FlowArray import *
    
class LinearSplineEinsumFlow(TractableModel):
    def __init__(self, config):
        self.leaf_distribution = LinearRationalSpline
        super().__init__(config)
        
class QuadraticSplineEinsumFlow(TractableModel):
    def __init__(self, config):
        self.leaf_distribution = QuadraticRationalSpline
        super().__init__(config)

    
    