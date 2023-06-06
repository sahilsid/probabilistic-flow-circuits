import sys 
import unittest
sys.path.append("./../")
from components.spn.FlowArray import *
import torch

class TestFlowArray(unittest.TestCase):
    num_vars                = 2
    num_dims                = 1
    num_repetition          = 10
    num_input_distributions = 10
    batch_size              = 100
    device                  = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    
    def __init__(self,*args,**kwargs):
        super(TestFlowArray, self).__init__(*args,**kwargs)
        self.flow_array = self.define_flow_array()

    def define_flow_array(self):
        return FlowArray(self.num_vars, self.num_dims, (self.num_input_distributions, self.num_repetition), device=self.device)
        
    def test_likelihood_computation(self):
        x = torch.randn(self.batch_size,self.num_vars,self.num_dims).to(self.device)
        context = self.flow_array.context.repeat(self.batch_size,1)
        ll = self.flow_array(x)
        self.assertEqual(ll.shape,(self.batch_size,self.num_vars,self.num_input_distributions, self.num_repetition)) 
    
    def test_sampling(self):
        samples = self.flow_array.sample(self.batch_size)
        self.assertEqual(samples.shape,(self.batch_size,self.num_vars,self.num_dims,self.num_input_distributions, self.num_repetition)) 
        
    def test_inveribility(self):
        flow_array = self.flow_array
        x = torch.randn(self.batch_size,self.num_vars,self.num_dims).to(self.device).to(torch.float64)
        z = flow_array.forward_transform(x)[:,:,0,0,:]
        x_recon = flow_array.inverse_transform(z)[:,:,0,0,:]
        self.assertLess((x-x_recon).abs().mean().item(),1e-5)
        
class TestLinearRationalSpline(TestFlowArray):
    def define_flow_array(self):
        return LinearRationalSpline(self.num_vars, self.num_dims, (self.num_input_distributions, self.num_repetition), device=self.device).to(self.device)
    
class TestQuadraticRationalSpline(TestFlowArray):
    def define_flow_array(self):
        return QuadraticRationalSpline(self.num_vars, self.num_dims, (self.num_input_distributions, self.num_repetition), device=self.device).to(self.device)

class TestAffineGaussian(TestFlowArray):
    def define_flow_array(self):
        return AffineGaussian(self.num_vars, self.num_dims, (self.num_input_distributions, self.num_repetition), device=self.device).to(self.device)

class TestHouseholderGaussian(TestFlowArray):
    def define_flow_array(self):
        return HouseholderGaussian(self.num_vars, self.num_dims, (self.num_input_distributions, self.num_repetition), device=self.device).to(self.device)

class TestScaledGaussian(TestFlowArray):
    def define_flow_array(self):
        return ScaledGaussian(self.num_vars, self.num_dims, (self.num_input_distributions, self.num_repetition), device=self.device).to(self.device)

class TestTranslatedGaussian(TestFlowArray):
    def define_flow_array(self):
        return TranslatedGaussian(self.num_vars, self.num_dims, (self.num_input_distributions, self.num_repetition), device=self.device).to(self.device)

if __name__ == '__main__':
    unittest.main()