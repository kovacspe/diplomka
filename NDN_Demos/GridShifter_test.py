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
        input_dims=[1, 2], 
        layer_sizes=[n_neurons],
        layer_types=['grid_shift']
        )
    hsm_params['bias_initializers']=['normal']
    
    return hsm_params


def classify(inp,grid):
    inp = np.tile(inp,(1,grid.shape[1]//2))
    grid = np.tile(grid,(inp.shape[0],1))
    return inp+grid



def eval(out,gold):
    return np.mean(np.square(out-gold))

def reshape_to_NDN(inp):
    return np.reshape(inp,[-1,np.prod(inp.shape[1:])])

n_neurons = 20
n_data = 2000
n_test = 100

grid_points = np.random.random(size=(1,n_neurons*2))*2 - 1 
inp = np.random.random(size=(n_data,2))*2-1
print(inp)
out = classify(inp,grid_points)

inp = reshape_to_NDN(inp)
out = reshape_to_NDN(out)
print(f'INP Reshaped{inp.shape}, OUT:{out.shape}')


hsm_params = get_hsm_params(n_neurons,10)
hsm = NDN.NDN(hsm_params,noise_dist='gaussian')
print('biases')

b = hsm.networks[0].layers[0].biases
#print(b)
opt_params = {'batch_size':5}
hsm.train(inp,out,test_indxs=np.arange(n_test),opt_params=opt_params,learning_alg="adam")

inp = np.random.rand(100,2)
out = classify(inp,grid_points)
inp = reshape_to_NDN(inp)
out = reshape_to_NDN(out)
with tf.Session() as sess:
    pred = hsm.generate_prediction(inp)
file_writer = tf.summary.FileWriter(logdir, hsm.graph)
print(out.shape)
print(pred.shape)

print(f'MSE:{eval(pred,out)}')
print(grid_points)
print('biases')

b = hsm.networks[0].layers[0].biases
print(b)
print(eval(grid_points,b))

