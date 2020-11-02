import NDN3
import NDN3.NDNutils as NDNutils
from NDN3.NDNutils import ffnetwork_params
import NDN3.NDN as NDN
import numpy as np
import math
from model import Model

class ILCRModel(Model):
    def get_params(self,input_channels, neurons, channels, means):
        params = ffnetwork_params(
            input_dims=[input_channels, self.height, self.width],
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
        return params

    def get_net(self, params):
        net = super().get_net(params)
        net.log_correlation = 'filter-low-std-gold'
        net.networks[-1].layers[-1].biases = 0.5 * np.log(np.exp(self.data_loader.means) - 1)
        return net

    def get_opt_params(self):
        epochs = 2000
        return {'batch_size': 256, 'use_gpu': False, 'epochs_summary': 25, 'epochs_training': epochs, 'learning_rate': 0.002}
