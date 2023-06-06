import sys 
import unittest
sys.path.append("./../")

from datasets.toy_3d import *
from datasets.uci import *
from datasets.mnist import MNIST

class TestToy3D(unittest.TestCase):
    num_dim  = 3
    datasets_to_check = [HELIX, KNOTTED]
    
    def test_dimensions(self):
        for dname in self.datasets_to_check:
            dataset = dname()
            self.assertEqual(len(dataset.trn.x.shape), 2)
            self.assertEqual(dataset.trn.x.shape[1], self.num_dim)
            self.assertEqual(dataset.trn.x.shape[0]+dataset.val.x.shape[0]+dataset.tst.x.shape[0], 20000)

class TestImageDatasets(unittest.TestCase):
    dataset = None
    def test_dimensions(self):
        if self.dataset is None:
            return
        self.assertEqual(len(self.dataset.trn.x.shape), 2)
        self.assertEqual(self.dataset.trn.x.shape[1], self.num_dim)
        self.assertEqual(self.dataset.trn.x.shape[0], self.n_train)
        self.assertEqual(self.dataset.val.x.shape[0], self.n_val)
        self.assertEqual(self.dataset.tst.x.shape[0], self.n_test)

class TestMNIST(TestImageDatasets):
    num_dim = 784
    n_train = 50000
    n_val   = 10000
    n_test  = 10000
    dataset = MNIST()

class TestUCI(unittest.TestCase):
    datasets_to_check = {
        'POWER'       : [POWER, 6, (1659917,184435,204928)],
        'HEPMASS'     : [HEPMASS, 21, (315123, 35013, 174987)],
        'MINIBOONE'   : [MINIBOONE, 43, (29556, 3284, 3648)],
        'GAS'         : [GAS, 8, (852174, 94685, 105206)],
    }
  
    def test_dimensions(self):
        for key in self.datasets_to_check:
            dname, n_f, n_d = self.datasets_to_check[key]
            dataset = dname()
            n_train, n_val, n_test = n_d
            self.assertEqual(len(dataset.trn.x.shape), 2)
            self.assertEqual(dataset.trn.x.shape[1], n_f)
            self.assertEqual(dataset.trn.x.shape[0], n_train)
            self.assertEqual(dataset.val.x.shape[0], n_val)
            self.assertEqual(dataset.tst.x.shape[0], n_test)
            
if __name__ == '__main__':
    unittest.main()
    