
import os 
import json
from components.spn.EinsumNetwork import Args
from components.spn.Graph import random_binary_trees, poon_domingos_structure
import torch 
import json 

root = f"{os.path.dirname(os.path.abspath(__file__))}"
test_dir = os.path.join(root, "tests")
experiment_dir = os.path.join(root, "experiments")
data_dir = os.path.join(root, "data")

class ExperimentConfig():
    experiment_name         = "3D"
    dataset_name            = "KNOTTED"
    model_name              = "LinearSplineEinsumFlow"
    num_input_distributions = 10
    num_repetition          = 10
    num_sums                = 10
    log_freq                = 10
    visualize_freq          = 5
    epochs                  = 100
    lr                      = 1e-3
    batch_size              = 200
    leaf_config             = {}
    use_em                  = 0
    online_em_frequency     = 1
    online_em_stepsize      = 5e-2
    graph_type              = random_binary_trees
    depth                   = -1
    num_classes             = 1
    device                  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pd_num_pieces           = [7, 28]
    num_samples_to_save     = 64
    resume                  = True 
    eval_batch_size         = 2000
    
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
           self.set_value(key, value)
        self.leaf_config["device"] = self.device
  
    def get_value(self,key):
        return getattr(self,key) if hasattr(self,key) else self.leaf_config[key] if key in self.leaf_config else None
    
    def set_value(self, key, value):
        if hasattr(self,key):
            setattr(self,key,value)
        else: 
            self.leaf_config[key] = value