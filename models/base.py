import torch 
from components.spn.EinsumNetwork import EinsumNetwork, Args
import copy 

class TractableModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.args   = Args(
                num_var                    =   self.config.num_vars,
                num_dims                   =   self.config.num_dims,
                num_input_distributions    =   self.config.num_input_distributions,
                num_sums                   =   self.config.num_sums,
                num_classes                =   self.config.num_classes,
                exponential_family         =   self.leaf_distribution,
                exponential_family_args    =   self.config.leaf_config,
                use_em                     =   self.config.use_em,
        )
        self.model  = EinsumNetwork(
        graph = copy.deepcopy(self.config.graph),
        args  = self.args
        )
        self.model.initialize() 
         
    def forward(self, x):
        return self.model(x)
    
    def em_process_batch(self):
        self.model.em_process_batch()
    
    def em_update(self):
        self.model.em_update()
        
    def sample(self, batch_size):
        with torch.no_grad():
            if(batch_size <= self.config.batch_size):
                samples = self.model.sample(batch_size)
            else:
                samples = torch.cat([self.model.sample(self.config.batch_size).detach() for _ in range(batch_size//self.config.batch_size)],axis=0) 
        return samples
    
    def sample_conditional(self, x, marginalization_idx):
        self.model.set_marginalization_idx(marginalization_idx)
        with torch.no_grad():
            samples = self.model.sample(x=x)
        self.model.set_marginalization_idx([])
        return samples
    
    def marginal_inference(self, x, marginalization_idx):
        self.model.set_marginalization_idx(marginalization_idx)
        marginal_ll = self.model(x)
        self.model.set_marginalization_idx([])
        return marginal_ll
    
    def mpe(self, x=None, marginalization_idx=[]):
        self.model.set_marginalization_idx(marginalization_idx)
        mpe = self.model.mpe(x)
        self.model.set_marginalization_idx([])
        return mpe
    