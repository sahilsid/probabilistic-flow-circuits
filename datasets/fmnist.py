import numpy as np
import os
import tempfile
import urllib.request
import utils
import shutil
import gzip
import datasets 
from datasets.uci import util

def maybe_download(directory, url_base, filename):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        return False

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base + filename
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading {} to {}'.format(url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return True

def maybe_download_fashion_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download( datasets.root + 'fashion-mnist', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file):
            continue
        filepath = os.path.join( datasets.root + 'fashion-mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_fashion_mnist():
    """Load fashion-MNIST"""

    maybe_download_fashion_mnist()

    data_dir = datasets.root + 'fashion-mnist'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels

class FMNIST:
    alpha = 0.05
    class Data:
        """
        Constructs the dataset.
        """
        def __init__(self, data, logit, dequantize, rng):
            x = self._dequantize(data[0], rng) if dequantize else data[0]  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x              # logit
            self.labels = data[1]                                          # numeric labels
            self.y = self.labels                                           # 1-hot encoded labels
            self.N = self.x.shape[0]                                       # number of datapoints
            self.H = 28
            self.W = 28
            self.C = 1
            
        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return util.logit(FMNIST.alpha + (1 - 2*FMNIST.alpha) * x)
        
        @staticmethod
        def _inv_logit_transform(x):
            return (util.logistic(x) - FMNIST.alpha)/(1 - 2*FMNIST.alpha)


    def __init__(self, logit=True, flip=False, dequantize=True):
        rng = np.random.RandomState(42)
        self.logit = logit
        train_x, train_l , test_x, test_l= load_fashion_mnist()
        train_x, test_x = train_x.astype(np.float32)/255, test_x.astype(np.float32)/255
        split = int(0.9 *train_x.shape[0])
        self.trn = self.Data((train_x[:split], train_l[:split]), logit, dequantize, rng)
        self.val = self.Data((train_x[split:], train_l[split:]), logit, dequantize, rng)
        self.tst = self.Data((test_x, test_l), logit, dequantize, rng)
        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims / 3))] * 2 + [3]
