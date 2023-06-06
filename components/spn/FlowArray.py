import torch
import os 
import pyro.distributions as dist
import numpy as np  
import itertools
from components.flows.Splines import *
from components.flows.Householder import *
from components.flows.Affine import *
from components.spn.ExponentialFamilyArray import NormalArray, shift_last_axis_to

class FlowArray(NormalArray):
    """Implementation of Leaf distributions using Normalizing Flows."""

    def __init__(self, num_var, num_dims, array_shape, min_var=0.0001, max_var=10., use_em=False, device=torch.device('cuda')):
        super(FlowArray, self).__init__(num_var, num_dims, array_shape, 2 * num_dims, use_em=use_em)
        self.device  = device
        self.min_var = min_var
        self.max_var = max_var
        self.num_var = num_var
        self.num_dims = num_dims
        self.array_shape = array_shape
        
        independednt_dims = np.product([num_var,*array_shape])
        self.independednt_dims = independednt_dims
        self.context    = torch.arange(independednt_dims).view(1,-1).to(device)
        
        self.mu, self.sigma = torch.zeros(self.num_dims).to(self.device), torch.ones(self.num_dims).to(self.device)
        self.base_dist_type = dist.Normal
        self.base_dist_params = [torch.zeros(self.num_dims).to(self.device), torch.ones(self.num_dims).to(self.device)]
        self.base_dist  = self.base_dist_type(*self.base_dist_params)
        self.transforms =  torch.nn.ModuleList([])
        self.initialize_flow()
    
    def update_flow_dist(self):
        self.base_dist  = self.base_dist_type(*self.base_dist_params)
        self.flow_dist = dist.ConditionalTransformedDistribution(self.base_dist,self.transforms)
        
    def initialize_flow(self):
        self.base_dist  = self.base_dist_type(*self.base_dist_params)
        self.num_stats = 0
        for p in self.transforms.parameters():
            self.num_stats += np.prod(list(p.size()))
        self.generative_flows = list(itertools.chain(*zip(self.transforms)))
        self.normalizing_flows = self.generative_flows[::-1] # normalizing direction (x-->z)
        self.flow_dist = dist.ConditionalTransformedDistribution(self.base_dist,self.transforms)
        self.flow_dist.clear_cache()
    
    def forward_transform(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size,self.num_var,1,1,self.num_dims).to(torch.float64)
        x = x.repeat(1,1,self.array_shape[0],self.array_shape[1],1)
        context = self.context.repeat(batch_size,1)
        x = x.view(-1,self.num_dims)
        context = context.view(-1)
        for transform in self.normalizing_flows:
            if(hasattr(transform,'condition')):
                x = transform.condition(context)(x)
            else:
                x = transform(x)
        return x.view(batch_size,self.num_var,self.array_shape[0],self.array_shape[1],self.num_dims)[:,:,:,:,:]
    
    def inverse_transform(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size,self.num_var,1,1,self.num_dims)
        x = x.repeat(1,1,self.array_shape[0],self.array_shape[1],1)
        context = self.context.repeat(batch_size,1)
        x = x.view(-1,self.num_dims)
        context = context.view(-1)
        for transform in self.generative_flows:
            if(hasattr(transform,'condition')):
                x = transform.condition(context).inv(x)
            else:
                x = transform.inv(x)
        return x.view(batch_size,self.num_var,self.array_shape[0],self.array_shape[1],self.num_dims)[:,:,:,:,:]
    
    def forward(self, x):
        """
        :return: log-densities of implemented exponential family (Tensor).
                 Will be of shape (batch_size, self.num_var, *self.array_shape)
        """
        batch_size = x.shape[0]
        self.device = x.device
        x = x.view(batch_size,self.num_var,1,1,self.num_dims)
        x = x.repeat(1,1,self.array_shape[0],self.array_shape[1],1)
        context = self.context.repeat(batch_size,1)
        x = x.view(-1,self.num_dims)
        context = context.view(-1)
        flow_dist = self.flow_dist.condition(context)
        ll = flow_dist.log_prob(x)
        
        if(len(ll.shape)>1):
            ll.view(batch_size,self.num_var,self.array_shape[0],self.array_shape[1],self.num_dims)
            ll = ll.sum(axis=-1)
            
        self.ll = ll.view(batch_size,self.num_var,self.array_shape[0],self.array_shape[1])
        
        # Marginalization in PCs works by simply setting leaves corresponding to marginalized variables to 1 (0 in
        # (log-domain). We achieve this by a simple multiplicative 0-1 mask, generated here.
        # TODO: the marginalization mask doesn't need to be computed every time; only when marginalization_idx changes.
        if self.marginalization_idx is not None:
            with torch.no_grad():
                self.marginalization_mask = torch.ones(self.num_var, dtype=self.ll.dtype, device=self.ll.device)
                self.marginalization_mask.data[self.marginalization_idx] = 0.0
                shape = (1, self.num_var) + (1,) * len(self.array_shape)
                self.marginalization_mask = self.marginalization_mask.reshape(shape)
                self.marginalization_mask.requires_grad_(False)
        else:
            self.marginalization_mask = None

        if self.marginalization_mask is not None:
            output = self.ll * self.marginalization_mask
        else:
            output = self.ll
        
        return output
    
    def default_initializer(self):
        phi = torch.empty(self.num_var, *self.array_shape, 2*self.num_dims)
        with torch.no_grad():
            phi[..., 0:self.num_dims] = torch.randn(self.num_var, *self.array_shape, self.num_dims)
            phi[..., self.num_dims:] = 1. + phi[..., 0:self.num_dims]**2       
        return phi

    def _sample(self, num_samples, params, std_correction=1.0):
        with torch.no_grad():
            context = self.context.repeat(num_samples,1)
            context = context.view(-1)
            flow_dist = self.flow_dist.condition(context)
            samples = flow_dist.sample(sample_shape=context.size())
            self.flow_dist.clear_cache()
            samples = samples.view(num_samples,self.num_var,self.array_shape[0],self.array_shape[1],self.num_dims)
            return shift_last_axis_to(samples, 2)

    def _argmax(self, params, **kwargs):
        with torch.no_grad():
            mu = params[..., 0:self.num_dims]
            return shift_last_axis_to(mu, 1)

    def em_set_hyperparams(self, update_frequency, learning_rate=1e-3, purge=True):
        """Set new setting for online EM."""
        self.leaf_optimizer = torch.optim.Adam(self.transforms.parameters(), learning_rate, maximize=True)
        
        if purge:
            self.em_purge()
            self._online_em_counter = 0
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate

    def em_purge(self):
        """ Discard em statistics."""
        if(hasattr(self, "leaf_optimizer")):
            self.leaf_optimizer.zero_grad()
        if self.ll is not None and self.ll.grad is not None:
            self.ll.grad.zero_()
        
    def em_process_batch(self):
        """
        Accumulate EM statistics of current batch. This should typically be called via EinsumNetwork.em_process_batch().
        """
        if not self._use_em:
            raise AssertionError("em_process_batch called while _use_em==False.")
        if self.params is None:
            return
        with torch.no_grad():
            # if self.ll is not None and self.ll.grad is not None:
            #     self.ll.grad.zero_()
            if self._online_em_frequency is not None:
                self._online_em_counter += 1
                if self._online_em_counter == self._online_em_frequency:
                    self.em_update(True)
                    self._online_em_counter = 0

    def em_update(self, _triggered=False):
        """
        Do an EM update. If the setting is online EM (online_em_stepsize is not None), then this function does nothing,
        since updates are triggered automatically. (Thus, leave the private parameter _triggered alone)

        :param _triggered: for internal use, don't set
        :return: None
        """
        if not self._use_em:
            raise AssertionError("em_update called while _use_em==False.")
        if self._online_em_stepsize is not None and not _triggered:
            return
        self.leaf_optimizer.step()
        self.leaf_optimizer.zero_grad()

class TranslatedGaussian(FlowArray):
    """Implementation of Leaf distributions using Linear Rational Spline Based Normalizing Flows."""

    def __init__(self, num_var, num_dims, array_shape, n_flows=1, use_em=False, device=torch.device('cuda')):
        
        super(TranslatedGaussian, self).__init__(num_var, num_dims, array_shape, use_em=use_em, device=device)
        transforms = []
        for _ in range(n_flows):
            transforms += [IndexedTranslate(self.independednt_dims, self.num_dims)]
        self.transforms =  torch.nn.ModuleList(transforms)
        self.initialize_flow()
       
class ScaledGaussian(FlowArray):
    """Implementation of Leaf distributions using Linear Rational Spline Based Normalizing Flows."""

    def __init__(self, num_var, num_dims, array_shape, n_flows=1, use_em=False, device=torch.device('cuda')):
        
        super(ScaledGaussian, self).__init__(num_var, num_dims, array_shape, use_em=use_em, device=device)
        transforms = []
        for _ in range(n_flows):
            transforms += [IndexedScale(self.independednt_dims, self.num_dims)]
        self.transforms =  torch.nn.ModuleList(transforms)
        self.initialize_flow()
 
class HouseholderGaussian(FlowArray):
    """Implementation of Leaf distributions using Linear Rational Spline Based Normalizing Flows."""

    def __init__(self, num_var, num_dims, array_shape, n_flows=1, use_em=False, device=torch.device('cuda')):
        super(HouseholderGaussian, self).__init__(num_var, num_dims, array_shape, use_em=use_em, device=device)
        transforms = []
        for _ in range(n_flows):
            transforms += [
                MixtureHouseholder(self.independednt_dims,self.num_dims)
            ]
        self.transforms =  torch.nn.ModuleList(transforms)
        self.initialize_flow()      
        
class LinearRationalSpline(FlowArray):
    """Implementation of Leaf distributions using Linear Rational Spline Based Normalizing Flows."""
    def __init__(self, num_var, num_dims, array_shape, n_flows=1, count_bins=32, bound=8, use_em=False, use_student=False, device=torch.device('cuda')):
        super(LinearRationalSpline, self).__init__(num_var, num_dims, array_shape, use_em=use_em, device=device)
        transforms = []
        if(use_student):
            self.base_dist_type   = dist.StudentT
            self.base_dist_params = [3*torch.ones(num_dims).to(device),  torch.zeros(num_dims).to(device),  torch.ones(num_dims).to(device)]
            self.base_dist   = self.base_dist_type(*self.base_dist_params)
        for _ in range(n_flows):
            transforms += [MixtureSpline(self.independednt_dims, self.num_dims, count_bins=count_bins, order='linear',bound=bound)]
        self.transforms =  torch.nn.ModuleList(transforms)
        self.initialize_flow()
        
        
class QuadraticRationalSpline(FlowArray):
    """Implementation of Leaf distributions using Quadratic Rational Spline Based Normalizing Flows."""

    def __init__(self, num_var, num_dims, array_shape, n_flows=1, count_bins=16, bound=16, use_em=False, device=torch.device('cuda')):
        super(QuadraticRationalSpline, self).__init__(num_var, num_dims, array_shape, use_em=use_em, device=device)
        transforms = []
        self.base_dist_type   = dist.StudentT
        self.base_dist_params = [3*torch.ones(num_dims).to(device),  torch.zeros(num_dims).to(device),  torch.ones(num_dims).to(device)]
        for _ in range(n_flows):
            transforms += [
                MixtureSpline(self.independednt_dims, self.num_dims, count_bins=count_bins, order='quadratic',bound=bound)
            ]
        self.transforms =  torch.nn.ModuleList(transforms)
        self.initialize_flow()
        
       