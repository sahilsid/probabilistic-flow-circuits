import sys 
import unittest
sys.path.append("./../")
from components.spn.Graph import random_binary_trees
from components.spn.EinsumNetwork import EinsumNetwork, Args, NormalArray
import torch 

class TestSPN(unittest.TestCase):
    num_vars                = 8
    depth                   = 3
    num_repetition          = 5
    num_sums                = 6
    exponential_family      = NormalArray
    num_input_distributions = 10
    graph = random_binary_trees(num_vars, depth, num_repetition)
    device = torch.device('cuda')
    args = Args(
            num_classes=1,
            num_input_distributions=num_input_distributions,
            exponential_family=NormalArray,
            num_sums=num_sums,
            num_var=num_vars
    )
    einet = EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    
    def test_layers(self):
        self.assertEqual(len(self.einet.einet_layers),self.depth+2)
        self.assertEqual(type(self.einet.einet_layers[0].ef_array),self.exponential_family)

if __name__ == '__main__':
    unittest.main()