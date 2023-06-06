import numpy as np
import os
import tempfile
import urllib.request
import utils
import shutil
import gzip

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


def maybe_download_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']
    for file in mnist_files:
        if not maybe_download('../data/mnist', 'http://yann.lecun.com/exdb/mnist/', file):
            continue
        print('unzip ../data/mnist/{}'.format(file))
        filepath = os.path.join('../data/mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_mnist():
    """Load MNIST"""
    maybe_download_mnist()
    data_dir = '../data/mnist'
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

def maybe_download_fashion_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download('../data/fashion-mnist', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file):
            continue
        print('unzip ../data/fashion-mnist/{}'.format(file))
        filepath = os.path.join('../data/fashion-mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_fashion_mnist():
    """Load fashion-MNIST"""
    maybe_download_fashion_mnist()
    data_dir = '../data/fashion-mnist'
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

if __name__ == '__main__':
    print('Downloading dataset -- this might take a while')

    print()
    print('MNIST')
    maybe_download_mnist()

    print()
    print('fashion MNIST')
    maybe_download_fashion_mnist()