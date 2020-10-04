import numpy as np


data_in = np.load('antolik_data/training_inputs.npy')
data_out = np.load('antolik_data/training_set.npy')
val_in = np.load('antolik_data/training_inputs.npy')
val_out = np.load('antolik_data/training_set.npy')
print(data_in.shape)
print(data_out.shape)
print(data_out)