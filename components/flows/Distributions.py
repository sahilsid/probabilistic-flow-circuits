
from abc import ABC, abstractmethod
import torch
import torch.nn
from pyro.distributions.conditional import ConditionalDistribution
import pyro

class ConditionalNormalDistribution(ConditionalDistribution):
    def __init__(self, num_mixture, num_dim):
        self.num_mixture = num_mixture
        self.num_dim = num_dim
        self.mus    = torch.nn.Parameter(torch.randn(self.num_mixture, self.num_dim))
        self.sigmas = torch.nn.Parameter(torch.randn(self.num_mixture, self.num_dim))

    def condition(self, context):
        mu    = torch.index_select(self.mus,0,context)
        sigma = torch.nn.functional.softplus(torch.index_select(self.sigma,0,context))
        return pyro.distributions.Normal(mu, sigma)
