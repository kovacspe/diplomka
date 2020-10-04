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


def get_hsm_params(in_w,in_h, output_shape, hls=10):
    

    hsm_params = NDNutils.ffnetwork_params(
        input_dims=[1, in_w, in_h], 
        layer_sizes=[hls, output_shape],
        conv_filter_widths=[3,None],
        normalization=[0], 
        layer_types=['conv','normal'], 
        act_funcs=['relu','sigmoid']
        )
    hsm_params['weights_initializers']=['normal','normal']
    hsm_params['normalize_weights']=[1,0]
    
    return hsm_params

def get_hsm_params2(in_neuro, output, hls=10):
    
    _,output_shape = output.shape

    hsm_params = NDNutils.ffnetwork_params(
        
        layer_sizes=[hls, output_shape],
        normalization=[0], 
        layer_types=['normal','normal'], 
        act_funcs=['relu','sigmoid'],
        
        )
    hsm_params['ffnet_n'] = [0]
    hsm_params['xstim_n'] = None
    hsm_params['weights_initializers']=['normal','normal']
    
    return hsm_params

def classify(arr):
    m = np.mean(arr,axis=(1,2))
    m[m>0.5]=1
    m[m<=0.5]=0
    return np.expand_dims(m,axis=1)

def eval(out,gold):
    return np.mean(np.square(out-gold))

inp = np.random.rand(100,20,20)
out = classify(inp)
print(inp.shape)
inp = np.reshape(inp,[-1,np.prod(inp.shape[1:])])
print('Reshaped',inp.shape)
print(out.shape)

hsm_params = [get_hsm_params(20,20,10)]
hsm_params.append(get_hsm_params2(10,out))
hsm = NDN.NDN(hsm_params, noise_dist='poisson')
hsm.train(inp,out,test_indxs=np.arange(2),learning_alg="lbfgs")

inp = np.random.rand(10,20,20)
out = classify(inp)
inp = np.reshape(inp,[-1,np.prod(inp.shape[1:])])

with tf.Session() as sess:
    pred = hsm.generate_prediction(inp)
file_writer = tf.summary.FileWriter(logdir, hsm.graph)
print(f'MSE:{eval(pred,out)}')

