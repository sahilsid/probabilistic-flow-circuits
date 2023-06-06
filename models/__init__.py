
from models.eiflow import *
from models.einet  import EinsumNet

_MODELS = {
    "EinsumNet"                         :   EinsumNet,
    "LinearSplineEinsumFlow"            :   LinearSplineEinsumFlow,
    "QuadraticSplineEinsumFlow"         :   QuadraticSplineEinsumFlow,
    "AffineEinsumFlow"                  :   AffineEinsumFlow,
}