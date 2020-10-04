import NDN3
import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np


# Define the Keras TensorBoard callback.
logdir="output/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


def get_hsm_params(n_neurons, output_shape, hls=10):
    

    hsm_params = NDNutils.ffnetwork_params(
        input_dims=[3, 3, 2], 
        xstim_n=[0,1],
        layer_sizes=[n_neurons],
        layer_types=['grid_sample']
        )
    
    return hsm_params


def classify(inp,grid):
    inp = np.tile(inp,(1,grid.shape[1]//2))
    grid = np.tile(grid,(inp.shape[0],1))
    return inp+grid



def eval(out,gold):
    return np.mean(np.square(out-gold))

def reshape_to_NDN(inp):
    return np.reshape(inp,[-1,np.prod(inp.shape[1:])])





n_neurons = 3
n_data = 2000
n_test = 100

grid = np.array([[[[0,1,2],[3,4,5],[6,7,8]],[[10,11,12],[13,14,15],[16,17,18]]],
[[[100,101,102],[103,104,105],[106,107,108]],[[110,111,112],[113,114,115],[116,117,118]]]],dtype=np.float32)
coord = np.array([[[0.5,0.5],[-1.0,-1.0]],[[0,-1.0],[0.5,1.0]]])
print(coord.shape)
#print(inp)
#out = classify(inp,grid_points)
print('grid',grid.shape)
#grid = reshape_to_NDN(grid)
#coord = reshape_to_NDN(coord)
#inp = np.concatenate([grid,coord],axis=1)
#out = reshape_to_NDN(out)
#print(f'INP Reshaped{inp.shape}, OUT:{0}')


hsm_params = get_hsm_params(n_neurons,10)
hsm = NDN.NDN(hsm_params,
    noise_dist='gaussian',
    input_dim_list=[[1,3,3,2],[1,2,2]])

with tf.Session() as sess:
    pred = hsm.generate_prediction([grid,coord])
file_writer = tf.summary.FileWriter(logdir, hsm.graph)


