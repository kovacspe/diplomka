import NDN3
import NDN3.NDNutils as NDNutils
from NDN3.NDNutils import ffnetwork_params
import NDN3.NDN as NDN
import numpy as np
import math


def define_spatial_feature_readout_network(batch_size, height, width, input_channels, neurons, channels, means):
    params = ffnetwork_params(
        input_dims=[input_channels, height, width],
        layer_sizes=[channels, channels, channels, neurons],
        layer_types=['conv', 'conv', 'conv', 'spatialxfeature'],
        act_funcs=['softplus', 'softplus', 'lin', 'softplus'],
        shift_spacing=[1, 1, 1, None],
        reg_list={
            # 'd2x': [0.03, 0.015, 0.015, None],
            'l1': [None, None, None, 0.02]
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
