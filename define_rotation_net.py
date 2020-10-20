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


def create_filter(filter_name, in_channels, out_channels):
    if filter_name not in filter_dict:
        raise AttributeError(f'filter {filter_name} doesnt exist')
    conv_filter = filter_dict[filter_name]
    conv_filter = np.tile(conv_filter, (in_channels, out_channels, 1, 1))
    conv_filter = np.transpose(conv_filter, (3, 2, 0, 1))
    return conv_filter


def define_spatial_feature_readout_network(batch_size, height, width, input_channels, neurons, channels, means):
    params = ffnetwork_params(
        input_dims=[input_channels, height, width],
        layer_sizes=[channels, channels, channels, neurons],
        layer_types=['conv', 'conv', 'conv', 'spatialxfeature'],
        act_funcs=['softplus', 'softplus', 'lin', 'softplus'],
        shift_spacing=[1, 1, 1, None],
        reg_list={
            # 'd2x': [0.03, 0.015, 0.015, None],
            'l1': [None, None, None, 0.04]
        })
    params['conv_filter_widths'] = [13, 5, 5, None]
    params['weights_initializers'] = ['trunc_normal',
                                      'trunc_normal', 'trunc_normal', 'trunc_normal']
    params['bias_initializers'] = ['zeros', 'zeros', 'zeros', 'trunc_normal']
    params['pos_constraint'] = [False, False, False, True]
    network = NDN.NDN(params,
                      input_dim_list=[[1, height, width]],
                      batch_size=batch_size,
                      noise_dist='poisson')
    network.log_correlation = 'filter-low-std-gold'
    network.networks[-1].layers[-1].biases = 0.5 * np.log(np.exp(means) - 1)
    return network
