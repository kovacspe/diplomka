import NDN3
import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN
import numpy as np
import math
from utils import create_filter



# Core - number 0,1,2
def get_core_params(height,width,input_channels,output_channels,layer_num):

    d2x = 0.0005
    l1 = 0.000001
    # channels 12
    # input kernel 7
    # hidden kernel 3 .. rec kernel
    output_shape = 1
    params = NDNutils.ffnetwork_params(
        input_dims=[input_channels,height,width], 
        layer_sizes=[output_channels],
        normalization=[0], 
        layer_types=['conv'], 
        act_funcs=['elu'],
        reg_list={
            'd2x':[d2x],
            'l1':[l1]},
        verbose=True)
    params['weights_initializers']=['normal']
    params['normalize_weights']=[1]
    if layer_num == 0:
        params['xstim_n']=[0]
        params['conv_filter_widths']=[7]
    else:
        params['xstim_n'] = None
        params['ffnet_n']=range(layer_num)
        params['conv_filter_widths']=[3]
    return params

# Filters #5 #6 #7
def get_lowpass_filter_params(height,width,input_channels,input_num):
    params = NDNutils.ffnetwork_params(
        input_dims=[input_channels,height,width], 
        layer_sizes=[input_channels],
        shift_spacing=2,
        conv_filter_widths=[5],
        normalization=[0], 
        layer_types=['conv'], 
        act_funcs=['lin']
        )
    params['weights_initializers']=['normal']
    params['normalize_weights']=[1]
    params['ffnet_n']=input_num
    params['xstim_n']=None
    return params

# Shifter #4
def get_shifter_params():
    params = NDNutils.ffnetwork_params(
        input_dims=[1, 2], 
        layer_sizes=[2],
        normalization=[0], # Check
        layer_types=['normal'], 
        act_funcs=['tanh']
        )
    params['weights_initializers']=['normal'] # Check
    params['normalize_weights']=[0] # Check
    params['xstim_n']=[2]
    return params

# Gridpoints #8
def get_grid_shifter_params(num_neurons):
    params = NDNutils.ffnetwork_params(
        input_dims=[2], 
        layer_sizes=[num_neurons],
        normalization=[0], # Check
        layer_types=['grid_shift'], 
        verbose=True
        )
    params['weights_initializers']=['normal'] # Check
    params['normalize_weights']=[0] # Check
    params['ffnet_n']=[4]
    params['xstim_n']=None

    return params

# Modulator #3
def get_modulator_params(number_of_neurons,hidden=50):
    l1 = 0.000001
    params = NDNutils.ffnetwork_params(
        input_dims=[1, 3], 
        layer_sizes=[hidden,number_of_neurons],
        normalization=[0], # Check
        layer_types=['normal','normal'], 
        act_funcs=['relu','exp'],
        reg_list={
            'l1':[None,l1]
            }
        )
    params['weights_initializers']=['normal','normal']# Check
    params['normalize_weights']=[0,0]# Check
    params['xstim_n']=[1]
    return params

# Channel combination #12
def get_combine_chan_params(number_of_neurons):
    params = NDNutils.ffnetwork_params(
            input_dims=[3,number_of_neurons,1], 
            layer_sizes=[number_of_neurons],
            normalization=[0], # Check
            layer_types=['add'], 
            act_funcs=['lin']
            )
    #params['output_dims'] = [1,number_of_neurons,1]
    params['weights_initializers'] = ['zeros']# Check
    params['normalize_weights'] = [0,0]# Check
    params['ffnet_n'] = [9,10,11]
    params['xstim_n'] = None
    return params

# Modulator multiplier #13
def get_modulator_mult_params(number_of_neurons):
    params = NDNutils.ffnetwork_params(
            input_dims=[2*number_of_neurons,1], 
            layer_sizes=[number_of_neurons],
            normalization=[0], # Check
            layer_types=['mult'], 
            act_funcs=['lin']
            )
    params['weights_initializers'] = ['zeros']# Check
    params['normalize_weights'] = [0,0]# Check
    params['ffnet_n'] = [3,12]
    params['xstim_n'] = None
    return params

# Sampler #9 #10 #11
def get_sampler_network(channels,height,width,n_neurons,img_net,location_net):
    d2x = 0.0005
    l1 = 0.000001
    params = NDNutils.ffnetwork_params(
        input_dims=[channels,height,width],
        layer_sizes=[n_neurons],
        normalization=[0], 
        layer_types=['grid_sample'], 
        act_funcs=['elu'],
        reg_list={
            'd2x':[d2x],
            'l1':[l1]},
        verbose=True)
    params['ffnet_n'] = [img_net]
    params['xstim_n'] = None
    params['network_type'] = 'sampler'
    params ['locationnet_n'] = [location_net]
    params['num_locations'] = n_neurons
    params['computed_locations'] = True
    return params


def define_MEI_network(height,width,input_channels,neurons):
    ff_networks = []
    chan = input_channels
    out_chans = []
    # Core 0-2
    for i in range(3):
        ff_networks.append(get_core_params(height,width,chan,2*chan,i))
        out_chans.append(2*chan)
        chan = sum(out_chans)

    # Modulator 3
    ff_networks.append(get_modulator_params(neurons))

    # Shifter 4
    ff_networks.append(get_shifter_params())

    curr_h = height
    curr_w = width
    # Lowpass filter 5-7
    for i in range(3):
        if i==0:
            ff_networks.append(get_lowpass_filter_params(curr_h,curr_w,chan,range(3)))
        else:
            ff_networks.append(get_lowpass_filter_params(curr_h,curr_w,chan,[4+i]))
        curr_h=math.ceil(curr_h/2)
        curr_w=math.ceil(curr_w/2)

    # GridShift #8
    ff_networks.append(get_grid_shifter_params(neurons))

    curr_h = height//2
    curr_w = width//2
    # Sampler #9-11
    for i in range(3):
        ff_networks.append(get_sampler_network(chan,curr_h,curr_w,neurons,img_net=i+5,location_net=8))
        curr_h=math.ceil(curr_h/2)
        curr_w=math.ceil(curr_w/2)

    # Combine channels #12
    ff_networks.append(get_combine_chan_params(neurons))

    # Mult #13
    ff_networks.append(get_modulator_mult_params(neurons))
    network = NDN(ff_networks,
        input_dim_list=[[1,64,36],[1,3,1],[1,2,1]])
    fit_vars = network.fit_variables()
    # Foreach lowpass filter
    for i in range(5,7):
        #initialize weights with gaussian filter and remove from fit variables 
        #network.networks[i].layers[0].weights[:,0] = filter_dict['gauss5x5']
        pass
    return network,fit_vars


