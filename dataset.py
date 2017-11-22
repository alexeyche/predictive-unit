
import numpy as np
from sklearn.datasets import make_classification
import os

def get_toy_sparse_data(dest_dim, size, n_classes=2, seed=2):
    x_values, y_values = make_classification(
        n_samples=size,
        n_features=2, 
        n_informative=2, 
        n_redundant=0, 
        n_repeated=0,
        n_clusters_per_class=2,
        n_classes=n_classes,
        scale=0.1,
        shift=5.0,
        random_state=seed
    )
    return quantize_data(x_values, dest_dim), y_values.astype(np.int32)

def quantize_data(x, dest_size):
    x_out = np.zeros((x.shape[0], dest_size))
    
    dim_size = x.shape[1]
    size_per_dim = dest_size/dim_size
    
    min_vals = np.min(x, 0)
    max_vals = np.max(x, 0)
    for xi in xrange(x.shape[0]):
        for di in xrange(dim_size):
            v01 = (x[xi, di] - min_vals[di]) / (max_vals[di] - min_vals[di])
            x_out[xi, int(di * size_per_dim + v01 * (size_per_dim-1))] = 1.0
    return x_out


def get_toy_data_baseline():
    return get_toy_data(4, 200, 2, 2)

def get_toy_data(dest_dim, size, n_classes=2, seed=2):
    x_values, y_values = make_classification(
        n_samples=size,
        n_features=dest_dim, 
        n_informative=dest_dim/2, 
        n_redundant=0, 
        n_repeated=0,
        n_clusters_per_class=2,
        n_classes=n_classes,
        scale=0.1,
        shift=5.0,
        random_state=seed
    )
    return x_values, y_values.astype(np.int32)

def one_hot_encode(target_v):
    y_v = np.zeros((target_v.shape[0], len(np.unique(target_v))))
    for cl_id, cl_v in enumerate(np.unique(target_v)):
        y_v[np.where(target_v==cl_v)[0], cl_id] = 1.0

    return y_v

class Dataset(object):
    @property
    def train_shape(self):
        raise NotImplementedError

    @property
    def test_shape(self):
        raise NotImplementedError

    @property
    def batch_size(self):        
        raise NotImplementedError

    def next_train_batch(self):
        raise NotImplementedError

    def next_test_batch(self):
        raise NotImplementedError

    @property
    def train_batch_size(self):
        raise NotImplementedError

    @property
    def test_batch_size(self):
        raise NotImplementedError

    @property
    def train_batches_num(self):
        return self.train_shape[0][0]/self.train_batch_size

    @property
    def test_batches_num(self):
        return self.test_shape[0][0]/self.test_batch_size


class MNISTDataset(Dataset):
    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data

        self._data = input_data.read_data_sets(
            "{}/tmp/MNIST_data/".format(os.environ["HOME"]),
            one_hot=True
        )
        self._batch_size = 100
        self._test_batch_size = 100

    @property
    def train_shape(self):
        return self._data.train.images.shape, self._data.train.labels.shape

    @property
    def test_shape(self):
        return self._data.test.images.shape, self._data.test.labels.shape

    def next_train_batch(self):
        return self._data.train.next_batch(self._batch_size)

    def next_test_batch(self):
        return self._data.test.next_batch(self._test_batch_size)

    @property
    def train_batch_size(self):
        return self._batch_size

    @property
    def test_batch_size(self):
        return self._batch_size

    @property
    def train_batches_num(self):
        return 1

    @property
    def test_batches_num(self):
        return 1 # TODO


class ToyDataset(Dataset):
    def __init__(self):
        x_v, target_v = get_toy_data_baseline()
        y_v = one_hot_encode(target_v)

        test_prop = x_v.shape[0]/5

        self._xt_v = x_v[:test_prop]
        self._yt_v = y_v[:test_prop]

        self._x_v = x_v[test_prop:]
        self._y_v = y_v[test_prop:]
        
    @property
    def train_shape(self):
        return self._x_v.shape, self._y_v.shape

    @property
    def test_shape(self):
        return self._xt_v.shape, self._yt_v.shape

    def next_train_batch(self):
        return self._x_v, self._y_v

    def next_test_batch(self):
        return self._xt_v, self._yt_v

    @property
    def train_batch_size(self):
        return self._x_v.shape[0]

    @property
    def test_batch_size(self):
        return self._xt_v.shape[0]
