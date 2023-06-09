import numpy as np
import os
import torch
import errno
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt
from math import sqrt, ceil
import torchvision 

def mkdir_p(path):
    """Linux mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def one_hot(x, K, dtype=torch.float):
    """One hot encoding"""
    with torch.no_grad():
        ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
        ind.scatter_(-1, x.unsqueeze(-1), 1)
        return ind

def remove_if_exists(file_name):
    if(os.path.exists(file_name)):
        os.remove(file_name)
   
def factor_int_2(n):
    val = ceil(sqrt(n))
    while True:
        if not n%val:
            val2 = n//val
            break
        val -= 1
    return val, val2
    

def sample_matrix_categorical(p):
    """Sample many Categorical distributions represented as rows in a matrix."""
    with torch.no_grad():
        cp = torch.cumsum(p[:, 0:-1], -1)
        rand = torch.rand((cp.shape[0], 1), device=cp.device)
        rand_idx = torch.sum(rand > cp, -1).long()
        return rand_idx


def get_visualization_fn(num_vars, dataset_name):
    if(num_vars == 3):
        return visualize_3d 
    elif dataset_name in ["MNIST",  "FMNIST"]:
        return visualize_image
    else:
        return None
    

def visualize_2d(model, config, dataset, save_dir, epoch=0):
    real_data = dataset.val.x
    gen_data  = model.sample(len(real_data)).cpu().numpy()
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    plot2d(data=real_data, ax=ax1, alpha=0.25)
    ax1.set_title("Real Data",fontsize=12,fontweight='bold')

    plot2d(gen_data,ax2,alpha=0.25)
    ax2.set_title(f"Generated Data \n {config.model_name} \n Epoch: {epoch}",fontsize=12,fontweight='bold')
    plt.savefig(os.path.join(save_dir,f"{epoch}.png"), bbox_inches="tight")
    plt.close()
                
def visualize_3d(model, config, dataset, save_dir, epoch=0):
    real_data = dataset.val.x[np.random.choice(np.arange(0, len(dataset.val.x)),1000)]
    gen_data  = model.sample(len(real_data)).cpu().numpy()
    
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    (xmax,xmin),(ymax,ymin),(zmax,zmin) = dataset.dim_range()
    plot3d(real_data, ax1, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
    ax1.set_title("Real Data",fontsize=12,fontweight='bold')

    plot3d(gen_data,ax2,0.25, ymin, ymax, xmin, xmax, zmin, zmax)
    ax2.set_title(f"Generated Data \n {config.model_name} \n Epoch: {epoch}",fontsize=12,fontweight='bold')
    plt.savefig(os.path.join(save_dir,f"{epoch}.png"), bbox_inches="tight")
    plt.show()  
    # plt.close()  


def visualize_image(model, config, dataset, save_dir, epoch=0):
    nr, nc = factor_int_2(config.num_samples_to_save)
    height, width, channels = dataset.val.H, dataset.val.W, dataset.val.C 
    
    samples_dir = os.path.join(save_dir, "samples")
    conditional_dir = os.path.join(save_dir, "conditional")
    mpe_dir = os.path.join(save_dir, "mpe")
    for dirname in [samples_dir, conditional_dir, mpe_dir]:
        os.makedirs(dirname, exist_ok=True)
    
    real_data = dataset.val.x[np.random.choice(np.arange(0, len(dataset.val.x)), config.num_samples_to_save)]
    if(isinstance(real_data, np.ndarray)):
        real_data = torch.from_numpy(real_data).to(config.device).to(torch.float32)
    gen_data  = model.sample(config.num_samples_to_save).detach().cpu().numpy()
    
    if(hasattr(dataset,"logit") and dataset.logit):
        gen_data = dataset.val._inv_logit_transform(gen_data)
    gen_data = np.reshape(gen_data, (-1, channels, height, width))
    remove_if_exists(os.path.join(samples_dir, f"sample_{epoch}.png"))
    torchvision.utils.save_image(torch.from_numpy(gen_data),os.path.join(samples_dir, f"sample_{epoch}.png"))
   
    # ground truth
    ground_truth = real_data.reshape((-1, channels, height, width)).detach().cpu().numpy()
    if(hasattr(dataset,"logit") and dataset.logit):
        ground_truth = dataset.val._inv_logit_transform(ground_truth)
    remove_if_exists(os.path.join(samples_dir, f"ground_truth.png"))
    torchvision.utils.save_image(torch.from_numpy(ground_truth),os.path.join(save_dir, "ground_truth.png"))
             
def plot2d(data, ax, alpha=1):
    x, y = data[:, 0], data[:, 1]
    xy = np.vstack([x, y])
    density = stats.gaussian_kde(xy)(xy)
    idx = density.argsort()
    x, y, density = x[idx], y[idx], density[idx]
    ax.scatter(x, y, c=density, alpha=alpha,s=10)
    for ax_ in [ax.xaxis,ax.yaxis]:
        ax_.set_ticklabels([])
        for line in ax_.get_ticklines():
            line.set_visible(False)
        
    ax.set_ylim(-2, 2.)
    ax.set_xlim(-2, 2.)

def plot3d(data, ax, alpha=0.25,ymin=-1,ymax=1,xmin=-1,xmax=1,zmin=-1,zmax=1, color = None):
    """
    Function to plot datapoints in 3D space. Takes as input a numpy array of size (N,3) 
    and a matplotlib axis object on which to plot.
    """
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    xyz = np.vstack([x, y, z])
    z[z < zmin] = np.nan
    z[z > zmax] = np.nan
    y[y < ymin] = np.nan
    y[y > ymax] = np.nan
    x[x < xmin] = np.nan
    x[x > xmax] = np.nan
    
    if(color is None):
        density = stats.gaussian_kde(xyz)(xyz)
        idx = density.argsort()
        x, y, z, density = x[idx], y[idx], z[idx], density[idx]
        ax.scatter(x, y, z, c=density, alpha=alpha,s=25, cmap="rainbow")
    else:
        ax.scatter(x, y, z, c=color, alpha=alpha,s=25, cmap="rainbow")
    
    for ax_ in [ax.xaxis,ax.yaxis,ax.zaxis]:
        ax_.pane.set_edgecolor('r')
        ax_.pane.fill = False
        ax_.set_ticklabels([])
        for line in ax_.get_ticklines():
            line.set_visible(False)
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)
    ax.set_zlim(zmin,zmax)