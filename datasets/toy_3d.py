import numpy as np
from sklearn.model_selection import train_test_split

class Base:
    """
    Provides an abstract skeleton for the dataset.
    """
    npoints = 20000
    ndim = 3
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        data = self.sample()
        trn, val, tst = self.__split__(data)
        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)
        self.n_dims = self.trn.x.shape[1]

    def __split__(self, data):
        data = data[np.random.choice(data.shape[0],size=self.npoints, replace=True)]
        train, test = train_test_split(data,test_size=0.5)
        test, val = train_test_split(test,test_size=0.5)
        return train, val, test
    
    def sample(self):
        raise NotImplementedError
    
    def dim_range(self):
        return (3.,-3.), (3.,-3.), (1.5,-1.5)
      
class KNOTTED(Base):
    """
    Implements a concrete 3D manifold dataset.
    """
    def __init__(self):
        self.std = 0.1
        self.scale = 4
        super(KNOTTED, self).__init__()
        
    def sample(self):
        n = self.npoints
        theta = np.linspace(-np.pi, np.pi, n)
        x =  np.sin(theta) + 2*np.sin(2*theta)
        y =  np.cos(theta) - 2*np.cos(2*theta)
        z =  np.sin(3*theta) 
        noise = self.std*np.random.normal(size=(x.shape[0],3))
        data  = self.scale*np.vstack([x,y,z]).T + noise
        return data

    def dim_range(self):
        return (8,-8), (6.,-6.), (4.5,-4.5)

class HELIX(Base):
    """
    Implements a concrete 3D manifold dataset.
    """
    def __init__(self):
        self.std = 0.05
        self.scale = 1
        super(HELIX, self).__init__()
        
    def sample(self):
        n = self.npoints
        theta_max = 6 * np.pi
        theta = np.linspace(0, theta_max, n)
        x = theta
        z =  np.sin(theta)
        y =  np.cos(theta)
        self.manifold = np.vstack([x,y,z]).T
        noise = self.std*np.random.normal(size=(x.shape[0],3))
        data  = self.scale*np.vstack([x,y,z]).T + noise
        return data
    
    def dim_range(self):
        return (20.,-1.), (1.5,-1.5), (1.5,-1.5)

       
class BentLISSAJOUS(Base):
    def __init__(self):
        self.std = 0.1
        self.scale = 4
        super(BentLISSAJOUS, self).__init__()
        
    def sample(self):
        n = self.npoints
        theta = np.linspace(-np.pi, np.pi, n)
        x =  np.sin(2*theta)
        y =  np.cos(theta)
        z =  np.cos(2*theta)
        noise = self.std*np.random.normal(size=(x.shape[0],3))
        data  = self.scale*np.vstack([x,y,z]).T + noise
        return data

    def dim_range(self):
        return (8,-8), (6.,-6.), (4,-4)

    
class DisjointCIRCLES(Base):
    def __init__(self):
        self.std = 0.05
        self.scale = 2
        super(DisjointCIRCLES, self).__init__()
        
    def sample(self):
        n = self.npoints
        theta = np.linspace(-np.pi, np.pi, n//2)
        x =  np.hstack([-2 + np.sin(theta), 2 + np.sin(theta)])
        y =  np.hstack([-1 + np.sin(theta), 1 + 2*np.cos(theta)])
        z =  np.hstack([-1 + np.sin(theta), 1 + 2*np.cos(theta)])
        noise = self.std*np.random.normal(size=(x.shape[0],3))
        data  = self.scale*np.vstack([x,y,z]).T + noise
        return data

    def dim_range(self):
        return (10,-10), (8.,-8.), (6,-6)

class TwistedEIGHT(Base):
    def __init__(self):
        self.std = 0.1
        self.scale = 4
        super(TwistedEIGHT, self).__init__()
        
    def sample(self):
        n = self.npoints
        theta = np.linspace(-np.pi, np.pi, n//2)
        x =  np.hstack([np.sin(theta), 2 + np.sin(theta)])
        y =  np.hstack([np.cos(theta), np.zeros_like(theta)])
        z =  np.hstack([np.zeros_like(theta), np.cos(theta)])
        noise = self.std*np.random.normal(size=(x.shape[0],3))
        data  = self.scale*np.vstack([x,y,z]).T + noise
        return data

    def dim_range(self):
        return (12,-12), (8.,-8.), (6,-6)
    
class InterlockedCIRCLES(Base):
    def __init__(self):
        self.std = 0.1
        self.scale = 4
        super(InterlockedCIRCLES, self).__init__()
        
    def sample(self):
        n = self.npoints
        theta = np.linspace(-np.pi, np.pi, n//2)
        x =  np.hstack([np.sin(theta), 1 + np.sin(theta)])
        y =  np.hstack([np.cos(theta), np.zeros_like(theta)])
        z =  np.hstack([np.zeros_like(theta), np.cos(theta)])
        noise = self.std*np.random.normal(size=(x.shape[0],3))
        data  = self.scale*np.vstack([x,y,z]).T + noise
        return data

    def dim_range(self):
        return (6,-6), (4.,-4.), (3,-3)
    