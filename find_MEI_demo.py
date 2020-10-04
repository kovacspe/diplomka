import tensorflow as tf
import numpy as np
from datetime import datetime
# Import NDN
from NDN3.NDN import NDN
from NDN3.NDNutils import ffnetwork_params


single_input = np.random.random((1,50))
train_inputs = np.random.random((2000,50))
print(train_inputs)
train_outputs = np.sum(train_inputs,axis=1)
print(train_outputs)

def get_simple_net(input_neurons,out):
    

    hsm_params = ffnetwork_params(
        input_dims=[input_neurons],
        layer_sizes=[out],
        layer_types=['normal'], 
        act_funcs=['relu']
        )
    hsm_params['xstim_n'] = [0]
    hsm_params['weights_initializers']=['normal']
    net = NDN(hsm_params,noise_dist='gaussian')
    return net

net = get_simple_net(50,1)
net.train(train_inputs,train_outputs,learning_alg='lbfgs')



input_data, output_data, _ = net._data_format(single_input, None, None)
net.batch_size=1
#net.data_in_var[0] = tf.Variable(single_input,constraint=lambda x: tf.clip_by_value(x, -1., 1.),)
net._build_graph(batch_size=net.batch_size, use_dropout=False)
x= []
logdir="output/" + datetime.now().strftime("%Y%m%d-%H%M%S")
with tf.Session(graph=net.graph) as sess:
    #net._restore_params(sess, input_data, output_data)
    #optimizer
    
    net._restore_params(sess,input_data,output_data)
    
    SGD = tf.train.GradientDescentOptimizer(0.001)
    # now find most exciting input
    #unreal_penalty = -1000*tf.reduce_sum(tf.minimum(0.0,net.data_in_var[0])) + 1000*tf.reduce_sum(tf.maximum(1.0,net.data_in_var[0])-1.0)

    feed_dict = {net.indices: [0]}
    #net.data_in_batch[0] = tf.clip_by_value(net.data_in_var[0], 0., 1.)
    
    #net.data_in_var[0].constraint = lambda x: tf.clip_by_value(x, -1., 1.)
    sgd_step = SGD.minimize(
        -net.networks[net.ffnet_out[0]].layers[-1].outputs,
        global_step=None,
        var_list=[net.data_in_var[0]]
    )
    for step in range(10000):
        sess.run(sgd_step, feed_dict=feed_dict)
        if step%1000==0:
            x.append(np.sum(net.data_in_var[0].eval()))
    print(net.data_in_var[0].eval())
    #print(unreal_penalty.eval())
file_writer = tf.summary.FileWriter(logdir, net.graph)



