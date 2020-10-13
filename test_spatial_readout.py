import NDN3
import NDN3.NDNutils as NDNutils
from NDN3.NDNutils import ffnetwork_params
import NDN3.NDN as NDN
import numpy as np
import math

def define_readout_net(input_channels,height,width,channels,neurons,batch_size):
    params = ffnetwork_params(
        input_dims=[input_channels,height,width], 
        layer_sizes=[channels,channels,channels,neurons],
        layer_types=['conv','conv','conv','spatialxfeature'], 
        act_funcs=['softplus','softplus','softplus','softplus'],
        shift_spacing=[2,2,2,None],
        reg_list={
            'd2x':[0.02, 0.01, 0.01, None],
            'l1': [None, None, None, 0.01]
            })
    params['conv_filter_widths'] = [13,5,5,None]
    params['weights_initializers'] = ['normal','normal','normal','normal']
    network = NDN.NDN(params,
        input_dim_list=[[1,height,width]],
        batch_size=batch_size,
        noise_dist='poisson')
    network.log_correlation='filter-low-std-gold'
    #network.networks[-1].layers[-1].biases = means/2
    fit_vars = network.fit_variables(layers_to_skip=[[]],fit_biases=True)
    return network,fit_vars

net = define_readout_net(1,64,32,16,6005,8)