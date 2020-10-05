import NDN3
import NDN3.NDNutils as NDNutils
from NDN3.NDNutils import ffnetwork_params
import NDN3.NDN as NDN
import numpy as np
import math



filter_dict = {
        'gauss5x5': np.float32([
            [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
            [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
            [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
            [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]),
        'gauss3x3': np.float32([
            [1 / 16, 1 / 8, 1 / 16],
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16]]
        ),
        'laplace5x5': np.outer(np.float32([1, 4, 6, 4, 1]), np.float32([1, 4, 6, 4, 1])) / 256,

    }

def create_filter(filter_name,in_channels,out_channels):
    if filter_name not in filter_dict:
        raise AttributeError(f'filter {filter_name} doesnt exist')
    conv_filter = filter_dict[filter_name]
    conv_filter = np.tile(conv_filter,(in_channels,out_channels,1,1))
    conv_filter = np.transpose(conv_filter,(3,2,0,1))
    return conv_filter

# Core - number 0,1,2
def get_core_params(height,width,input_channels,output_channels,layer_num):
    # channels 12
    # input kernel 7
    # hidden kernel 3 .. rec kernel
    output_shape = 1
    params = ffnetwork_params(
        input_dims=[input_channels,height,width], 
        layer_sizes=[output_channels],
        layer_types=['conv'], 
        act_funcs=['softplus'],
        reg_list={
            'd2x':[0.02 if layer_num==0 else 0.01]})
    params['weights_initializers'] = ['normal']
    params['pos_constraint'] = True
    if layer_num == 0:
        params['xstim_n']=[0]
        params['conv_filter_widths']=[13]
    else:
        params['xstim_n'] = None
        params['ffnet_n'] = range(layer_num)
        params['conv_filter_widths']=[5]
    return params

# Filters #3 #4 #5
def get_lowpass_filter_params(height,width,input_channels,input_num):
    params = NDNutils.ffnetwork_params(
        input_dims=[input_channels,height,width], 
        layer_sizes=[input_channels],
        shift_spacing=2,
        conv_filter_widths=[5],
        layer_types=['conv'], 
        act_funcs=['lin']
        )
    params['ffnet_n']=input_num
    params['xstim_n']=None
    return params

# Channel combination #9
def get_combine_chan_params(number_of_neurons):
    params = NDNutils.ffnetwork_params(
            input_dims=[3,number_of_neurons,1], 
            layer_sizes=[number_of_neurons],
            #normalization=[0], # Check
            layer_types=['add'], 
            act_funcs=['relu'],
            log_activations=True
            )
    #params['output_dims'] = [1,number_of_neurons,1]
    params['weights_initializers'] = ['normal']# Check
    params['ffnet_n'] = [6]
    params['xstim_n'] = None
    return params



# Sampler #6
def get_readout_network(channels,height,width,n_neurons,img_net):
    l1 = 0.1
    params = ffnetwork_params(
        input_dims=[channels,height,width],
        layer_sizes=[n_neurons],
        layer_types=['normal'], 
        act_funcs=['lin'],
        log_activations=True,
        reg_list={
            'l1':[l1]})
    params['ffnet_n'] = [img_net]
    params['xstim_n'] = None
    params['network_type'] = 'sampler'
    return params

def define_MEI_network(batch_size,height,width,input_channels,neurons,channels,means):
    ff_networks = []
    layers_to_skip = []
    chan = input_channels
    out_chans = []

    # Core 0-2
    for i in range(3):
        if i==0:
            ff_networks.append(get_core_params(height,width,1,channels,i))
            out_chans.append(channels)
        elif i==2:
            ff_networks.append(get_core_params(height,width,chan,8,i))
            out_chans.append(8)
        else:
            ff_networks.append(get_core_params(height,width,chan,chan,i))
            out_chans.append(chan)
        chan = sum(out_chans)
        layers_to_skip.append([])

    curr_h = height
    curr_w = width
    # Lowpass filter 3-5
    for i in range(3):
        if i==0:
            ff_networks.append(get_lowpass_filter_params(curr_h,curr_w,chan,range(3)))
        else:
            ff_networks.append(get_lowpass_filter_params(curr_h,curr_w,chan,[2+i]))
        layers_to_skip.append([0])
        curr_h=math.ceil(curr_h/2)
        curr_w=math.ceil(curr_w/2)

    curr_h = height//2
    curr_w = width//2
    # Readout layer
    ff_networks.append(get_readout_network(chan,curr_h,curr_w,neurons,img_net=i+3))
    
    network = NDN.NDN(ff_networks,
        input_dim_list=[[1,64,36]],
        batch_size=batch_size,
        noise_dist='poisson')
    network.log_correlation='filter-low-std-gold'
    fit_vars = network.fit_variables(layers_to_skip=layers_to_skip,fit_biases=True)
    
    return network,fit_vars




