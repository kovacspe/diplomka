import tensorflow as tf
import numpy as np
from datetime import datetime
# Import NDN
from NDN3.NDN import NDN
from NDN3.NDNutils import ffnetwork_params
from utils import filter_dict 
from data_loaders import reshape_input_to_NDN
import matplotlib.pyplot as plt
from MEIutils import find_MEI, plot_filter

single_input = np.random.random((1,5,5))
train_inputs = np.random.random((2000,5,5))#np.ones((2000,5,5))
filt = np.tile(filter_dict['gauss5x5'],(2000,1,1))
train_outputs = np.array([[x,y] for x,y in zip(np.sum(train_inputs*filt,axis=(1,2)),-np.sum(train_inputs,axis=(1,2)))])
print(train_outputs)

def get_simple_net(input_shape,out):
    hsm_params = ffnetwork_params(
        input_dims=input_shape,
        layer_sizes=[out,out],
        layer_types=['var','normal'], 
        act_funcs=['lin','lin'],
        reg_list={
            'l2':[0.1,None]
            }
        )
    hsm_params['xstim_n'] = [0]
    hsm_params['weights_initializers']=['normal','normal']
    net = NDN(hsm_params,noise_dist='gaussian')
    return net

def get_simple_gan_ffnetwork_params(noise_vector_len, image_w, image_h):
    gan_params = ffnetwork_params(
        input_dims=[noise_vector_len],
        layer_sizes=[image_w*image_h],
        layer_types=['normal'], 
        act_funcs=['tanh']
        )
    hsm_params['xstim_n'] = [0]
    hsm_params['weights_initializers']=['normal']
    return gan_params

def construct_gan_net(ffnetwork,noise_vector_len,image_w,image_h):
    gan_params = get_simple_gan_ffnetwork_params(noise_vector_len, image_w, image_h)
    net = NDN(network_list=[ffnetwork,gan_params],noise_dist='poisson')
    return net

def find_MEI_old(net,neurons,output_size):
    input_data, output_data, _ = net._data_format(single_input, None, None)
    input_data = reshape_input_to_NDN(single_input)
    net.batch_size=1
    data_filter = np.zeros(output_size)
    data_filter[neurons] = 1
    net.network_list[0]['as_var'] = True
    net._build_graph(batch_size=net.batch_size, use_dropout=False)
    x = []
    logdir="output/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    with tf.Session(graph=net.graph) as sess:

        net._restore_params(sess,input_data,output_data)
        file_writer = tf.summary.FileWriter(logdir, net.graph)
        SGD = tf.train.GradientDescentOptimizer(0.001)
        sgd_step = SGD.minimize(
            -(net.networks[net.ffnet_out[0]].layers[-1].outputs * data_filter) + net.cost_reg,
            global_step=None,
            var_list=[net.networks[0].layers[0].weights_var]
        )
        feed_dict = {net.indices: [0]}
        for step in range(100000):
            sess.run(sgd_step, feed_dict=feed_dict)
            if step%1000==0:
                x.append(np.sum(net.networks[0].layers[0].weights_var.eval()))
        print(net.networks[0].layers[0].weights_var.eval())
        #print(unreal_penalty.eval())
    
    return

net = get_simple_net([5,5],2)
net.batch_size=1

train_inputs = reshape_input_to_NDN(train_inputs)
train_outputs = train_outputs
net.train(train_inputs,train_outputs,learning_alg='lbfgs')

for neuron in range(2):
    net = find_MEI(net,neuron)
    plot_filter(net)



