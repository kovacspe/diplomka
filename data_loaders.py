import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime
from utils import reshape_input_to_NDN
import math

class DataLoader:
    def __init__(self,data_file):
        pass
    def val(self,NDN_reshape=False):
        pass

    def train(self,NDN_reshape=False):
        pass

    def test(self, averages=True,NDN_reshape=False):
        pass

    @property
    def width(self):
        pass

    @property
    def height(self):
        pass

    @property
    def num_neurons(self):
        pass

class Dataset:
    def __init__(self,
                 images_train,
                 responses_train,
                 images_val,
                 responses_val,
                 images_test,
                 responses_test):

        # normalize images (mean=0, SD=1)
        m = images_train.mean()
        sd = images_train.std()
        zscore = lambda img: (img - m) / sd
        self.images_train = zscore(images_train)[...,None]
        self.images_val = zscore(images_val)[...,None]
        self.images_test = zscore(images_test)[...,None]
        
        # normalize responses (SD=1)
        sd = responses_train.std(axis=0)
        sd[sd < (sd.mean() / 100)] = 1
        def rectify_and_normalize(x):
            x[x < 0] = 0    # responses are non-negative; this gets rid
                            # of small negative numbers due to numerics
            return x / sd
        self.responses_train = rectify_and_normalize(responses_train)
        self.responses_val = rectify_and_normalize(responses_val)
        self.responses_test = rectify_and_normalize(responses_test)
        
        self.num_neurons = responses_train.shape[1]
        self.num_train_samples = images_train.shape[0]
        self.px_x = images_train.shape[2]
        self.px_y = images_train.shape[1]
        self.input_shape = [None, self.px_y, self.px_x, 1]
        self.minibatch_idx = 1e10
        self.train_perm = []

    def val(self,NDN_reshape=False):
        if NDN_reshape:
            return reshape_input_to_NDN(self.images_val),reshape_input_to_NDN(self.responses_val)
        else:
            return self.images_val, self.responses_val

    def train(self,NDN_reshape=False):
        if NDN_reshape:
            return reshape_input_to_NDN(self.images_train),reshape_input_to_NDN(self.responses_train)
        else:
            return self.images_train, self.responses_train

    def test(self, averages=True,NDN_reshape=False):
        responses = self.responses_test.mean(axis=0) if averages else self.responses_test
        if NDN_reshape:
            return reshape_input_to_NDN(self.images_test),reshape_input_to_NDN(responses)
        else:
            return self.images_test, responses

    def minibatch(self, batch_size):
        if self.minibatch_idx + batch_size > self.num_train_samples:
            self.next_epoch()
        idx = self.train_perm[self.minibatch_idx + np.arange(0, batch_size)]
        self.minibatch_idx += batch_size
        return self.images_train[idx, :, :], self.responses_train[idx, :]

    def next_epoch(self):
        self.minibatch_idx = 0
        self.train_perm = np.random.permutation(self.num_train_samples)

class ICLRDataLoader(DataLoader):
    def __init__(self,data_file):
        with open(data_file,'rb') as file:
            self.data = pickle.load(file)
    def val(self,NDN_reshape=False):
        return self.data.val(NDN_reshape)

    def train(self,NDN_reshape=False):
        return self.data.train(NDN_reshape)

    def test(self, averages=True,NDN_reshape=False):
        return self.data.train(NDN_reshape)

    @property
    def width(self):
        return self.data.train[0].shape[3]

    @property
    def height(self):
        return self.data.train[0].shape[2]

    @property
    def num_neurons(self):
        return self.data.train[1].shape[1]
    

class AntolikDataLoader(DataLoader):
    def __init__(self, data_folder,region):
        self.data_folder=data_folder
        self.region=region

        self.train_x, self.train_y = self.load(region,'training')
        self.val_x, self.val_y = self.load(region,'validation')

    def train(self,NDN_reshape=False):
        return self.train_x, self.train_y

    def val(self,NDN_reshape=False):
        return self.val_x, self.val_y
    
    def test(self,NDN_reshape=False):
        return None,None
    
    @property
    def width(self):
        return int(math.sqrt(self.train_x.shape[1]))

    @property
    def height(self):
        return int(math.sqrt(self.train_x.shape[1]))

    @property
    def num_neurons(self):
        return self.train_y.shape[1]

    def load(self,region,data_type):
        x = np.load(f'{self.data_folder}/region{region}/{data_type}_inputs.npy')
        y = np.load(f'{self.data_folder}/region{region}/{data_type}_set.npy')
        return x,y

def get_data_loader(data_type):
    data_path = 'data'
    if data_type=='antolik1':
        return AntolikDataLoader(data_path,1)
    if data_type=='antolik2':
        return AntolikDataLoader(data_path,2)
    if data_type=='antolik3':
        return AntolikDataLoader(data_path,3)
    if data_type=='iclr':
        return ICLRDataLoader(f'{data_path}/data.pkl')