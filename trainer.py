import os 
import torch 
import pickle 
from config import experiment_dir 
from datasets import _DATASETS
from models import _MODELS
from components.spn.EinsumNetwork import eval_loglikelihood_batched
from config import ExperimentConfig 
import numpy as np  
from components.spn.Graph import random_binary_trees, poon_domingos_structure
from utils import *
import argparse 
import tqdm 
import json 
from torch.utils.data.dataloader import default_collate
import torch 
import random 
import numpy as np  


class Experiment():
    def __init__(self, config, dataset=None):
        self.config = config
        self.load_dataset(dataset)
        self.experiment_dir  = os.path.join(experiment_dir, self.config.experiment_name, self.config.dataset_name, self.config.model_name)
        self.experiment_dir  = os.path.join(self.experiment_dir, "main")
        self.visualization_dir = os.path.join(self.experiment_dir,"plots")
        self.config_filepath   = os.path.join(self.experiment_dir,"config.pkl")
        self.log_filepath      = os.path.join(self.experiment_dir,"log.pkl")
        self.model_filepath    = os.path.join(self.experiment_dir,f"{self.config.model_name}.ckpt")
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        self.epoch = 0
        self.log = {
            "train_ll"  :   [],
            "test_ll"   :   [],
            "valid_ll"  :   [],
        }
        self.config.graph  = config.graph if hasattr(config, "graph") else self.configure_pc_structure()
        self.initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.config.lr) if not self.config.use_em else None
        if(self.config.resume):
            self.load_exp_status()
        
    def configure_pc_structure(self):
        if(self.config.graph_type == poon_domingos_structure):
            height, width = self.dataset.trn.H, self.dataset.trn.W 
            pd_delta = [[height / d, width / d] for d in self.config.pd_num_pieces]
            return poon_domingos_structure(shape=(height, width), delta=pd_delta)
        else:
            self.config.depth = self.config.depth if self.config.depth>0 else int(max(1,np.log2(self.config.num_vars)))
            return random_binary_trees(self.config.num_vars, self.config.depth, self.config.num_repetition)
            
    def load_dataset(self, dataset=None):
        assert self.config.dataset_name in _DATASETS
        self.dataset           = _DATASETS[self.config.dataset_name]() if dataset is None else dataset
        self.config.num_vars   = self.dataset.num_vars if(hasattr(self.dataset,"num_vars")) else self.dataset.trn.x.shape[1]
        self.config.num_dims   = 1 if(len(self.dataset.trn.x.shape)==2) else self.dataset.trn.x.shape[-1]
        if(hasattr(self.dataset,"train_dataloader")):
            self.train_dataloader = self.dataset.train_dataloader
        else:
            self.train_dataloader  = torch.utils.data.DataLoader(
                torch.from_numpy(self.dataset.trn.x).to(torch.float32),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=lambda x: default_collate(x).to(self.config.device)
            )
        if(hasattr(self.dataset,"valid_dataloader")):
            self.valid_dataloader = self.dataset.valid_dataloader
        else:
            self.valid_dataloader  = torch.utils.data.DataLoader(
                torch.from_numpy(self.dataset.val.x).to(torch.float32),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=lambda x: default_collate(x).to(self.config.device)
            )
        if(hasattr(self.dataset,"test_dataloader")):
            self.test_dataloader = self.dataset.test_dataloader
        else:
            self.test_dataloader  = torch.utils.data.DataLoader(
                torch.from_numpy(self.dataset.tst.x).to(torch.float32),
                batch_size=self.config.batch_size,
                shuffle=True,
                collate_fn=lambda x: default_collate(x).to(self.config.device)
            )
        self.visualize_fn = get_visualization_fn(self.config.num_vars, self.config.dataset_name)
        
    def initialize_model(self):
        assert self.config.model_name in _MODELS
        self.model = _MODELS[self.config.model_name](self.config)
        self.model.to(self.config.device)
        
    def save_exp_status(self):
        with open(self.config_filepath, 'wb') as file:
            pickle.dump(self.config, file)
        
        with open(self.log_filepath, 'wb') as file:
            pickle.dump(self.log, file)
            
        last_ll, best_ll = self.log["valid_ll"][-1][1], max([a[1] for a in self.log["valid_ll"]])
        if(last_ll>=best_ll):
            torch.save({
                "model": self.model.state_dict(),
                "graph": self.config.graph,
                "args" : self.model.args,
                "epoch": self.epoch,
                "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") and self.optimizer is not None else None,
                },
                self.model_filepath
            ) 
    
    def load_exp_status(self):
        if(os.path.exists(self.config_filepath)):
            with open(self.config_filepath, 'rb') as file:
                self.config = pickle.load(file)
            self.configure_pc_structure()
            self.initialize_model()
            
        if(os.path.exists(self.log_filepath)):
            with open(self.log_filepath, 'rb') as file:
                self.log = pickle.load(file)
        
        if(os.path.exists(self.model_filepath)):
            ckpt = torch.load(self.model_filepath)
            self.config.graph = ckpt["graph"]
            self.config.args  = ckpt["args"]
            self.initialize_model()
            self.model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt and not ckpt["optimizer"] is None:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epoch = ckpt["epoch"]
    
    def grad_check(self):
        if("Flow" in self.config.model_name):
            for name, param in self.model.named_parameters():
                if param is None or not hasattr(param,"grad") or hasattr(param,"grad") and param.grad is None:
                    continue
                if torch.isnan(param.grad).any():
                    param.grad = torch.nan_to_num(param.grad)   
                      
    def run_training_loop(self):
        # self.run_validation_loop()
        for self.epoch in range(self.epoch+1, self.epoch+self.config.epochs+1):
            for batch in tqdm.tqdm(self.train_dataloader):
            # for batch in self.train_dataloader:
                if(type(batch)==list):
                    batch = batch[0]
                batch = batch.to(self.config.device)
                log_likelihood = self.model(batch).mean()
                if(not self.config.use_em):
                    self.optimizer.zero_grad()
                    nll = -log_likelihood
                    nll.backward()
                    self.grad_check()
                    self.optimizer.step()
                        
                else:
                    objective = log_likelihood
                    objective.backward()
                    self.model.em_process_batch()
                    
            if(self.config.use_em):
                self.model.em_update()
                
            if(self.epoch%self.config.log_freq==0):
                self.run_validation_loop()
        self.run_testing_loop()
        
    def run_validation_loop(self):
        n_samples, train_ll = 0, 0
        for i,batch in enumerate(self.train_dataloader):
            if(type(batch)==list):
                batch = batch[0]
            batch = batch.to(self.config.device)
            train_ll += eval_loglikelihood_batched(
                            self.model,
                            batch.to(self.config.device)
                        )
            n_samples += len(batch)
            if(n_samples>=self.config.eval_batch_size):
                break
        self.log["train_ll"].append((self.epoch, train_ll/n_samples))
        
        n_samples, valid_ll = 0, 0
        for i,batch in enumerate(self.valid_dataloader):
            if(type(batch)==list):
                batch = batch[0]
            batch = batch.to(self.config.device)
            valid_ll += eval_loglikelihood_batched(
                            self.model,
                            batch.to(self.config.device)
                        )
            n_samples += len(batch)
            if(n_samples>=self.config.eval_batch_size):
                break
        self.log["valid_ll"].append((self.epoch, valid_ll/n_samples))
        
        print(f'EP-[{self.epoch:4d}] \t {self.config.model_name} \t {self.config.dataset_name} \t Train LL: {self.log["train_ll"][-1][1]: .4f} \t Valid LL: {self.log["valid_ll"][-1][1]: .4f}')
        if(self.epoch%self.config.visualize_freq ==0 and self.visualize_fn is not None):
            self.visualize_fn(self.model, self.config, self.dataset, self.visualization_dir, self.epoch)   
        self.save_exp_status()
        
    def run_testing_loop(self):
        self.load_exp_status()
        n_samples, test_ll = 0, 0
        for i,batch in enumerate(self.test_dataloader):
            if(type(batch)==list):
                batch = batch[0]
            batch = batch.to(self.config.device)
            test_ll += eval_loglikelihood_batched(
                            self.model,
                            batch.to(self.config.device)
                        )
            n_samples += len(batch)
        self.log["test_ll"].append((self.epoch, test_ll/n_samples))
        print(f'----------\nEP-[{self.epoch:4d}] \t {self.config.model_name} \t {self.config.dataset_name} \t Test LL: {self.log["test_ll"][-1][1]: .4f} ')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--dataset')
    parser.add_argument('-g', '--graph')
    parser.add_argument('-c', '--config')

    args = parser.parse_args()
    graph_type = random_binary_trees if "binary" in args.graph else poon_domingos_structure
    config     = ExperimentConfig(
        dataset_name    =   args.dataset,
        graph_type      =   graph_type,
        model_name      =   args.model,
        **json.loads(args.config)
    )   
    experiment = Experiment(config)
    experiment.run_training_loop()