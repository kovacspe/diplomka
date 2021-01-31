import NDN3.NDN as NDN
from NDN3 import NDNutils
import numpy as np

class Model:
    def __init__(self,data_loader,args):
        self.data_loader = data_loader
        self.width = data_loader.width
        self.height = data_loader.height
        self.out_num = data_loader.num_neurons
        self.args = args
        self.net_name='generic'
        self.opt_params = {}

    def get_params(self):
        return {}

    def get_net(self, seed):
        params = self.get_params()
        print(self.opt_params)
        bs = self.get_opt_params()['batch_size']
        seed = self.get_opt_params()['seed'] if 'seed' in self.get_opt_params() else seed
        net = NDN.NDN(params,
                      input_dim_list=[[1, self.data_loader.width, self.data_loader.height]],
                      batch_size=bs,
                      noise_dist='poisson',
                      tf_seed=seed)
        return net

    def get_opt_params(self):
        return self.opt_params

    def get_name(self):
        name = self.net_name
        for key,value in self.args.items():
            name+=f'-{key}{value}'
        return name


class SimpleConvModel(Model):
    def __init__(self,data_loader,args):
        super().__init__(data_loader,args)
        epochs = 5000
        self.opt_params = {'display': 1,'batch_size': 16, 'use_gpu': False, 'epochs_summary': epochs//50, 'epochs_training': epochs, 'learning_rate': 0.001}
        self.opt_params.update(self.args)
        self.net_name = 'conv'
        
        
    def get_params(self):
        params = NDNutils.ffnetwork_params(
                    input_dims=[1, self.width, self.height], 
                    layer_sizes=[self.args['channels'],self.args['channels'], int(0.2*self.out_num), self.out_num], # paper: 9, 0.2*output_shape
                    ei_layers=[None,None, None, None],
                    normalization=[0,0, 0, 0], 
                    layer_types=['var','conv', self.args['hidden_lt'], 'normal'],
                    act_funcs=['lin','softplus', 'softplus','softplus'],
                    shift_spacing=[1,(self.args['c_size']+1)//2, 1, 1],
                    conv_filter_widths=[0,self.args['c_size'], 0, 0],
                    reg_list={
                        'd2x': [None,self.args['cd2x'], None , None],
                        self.args['hidden_t']:[None,None, self.args['hidden_s'], None],
                        'l2':[0.1,None, None, 0.1],
                        })
        params['weights_initializers']=['normal','normal','normal','normal']
        params['biases_initializers']=['normal','trunc_normal','trunc_normal','trunc_normal']
        return params
        
class SimpleConvGANModel(Model):
    def __init__(self,data_loader,args):
        super().__init__(data_loader,args)
        epochs = 5000
        self.opt_params = {'display': 1,'batch_size': 16, 'use_gpu': False, 'epochs_summary': epochs//50, 'epochs_training': epochs, 'learning_rate': 0.001}
        self.opt_params.update(self.args)
        self.net_name = 'conv'
        
        
    def get_params(self):
        params = NDNutils.ffnetwork_params(
                    input_dims=[10], 
                    layer_sizes=[[31,31],30, int(0.2*self.out_num), self.out_num], # paper: 9, 0.2*output_shape
                    ei_layers=[None,None, None, None],
                    normalization=[0,0, 0, 0], 
                    layer_types=['normal','conv', 'conv', 'normal'],
                    act_funcs=['lin','softplus', 'softplus','softplus'],
                    shift_spacing=[1,(7+1)//2, 1, 1],
                    conv_filter_widths=[0,7, 0, 0],
                    reg_list={
                        'd2x': [None,0.2, None , None],
                        'l2':[0.1,None, None, 0.1],
                        })
        params['weights_initializers']=['normal','normal','normal','normal']
        params['biases_initializers']=['normal','trunc_normal','trunc_normal','trunc_normal']
        return params




class FCModel(Model):
    def __init__(self,data_loader,args):
        super().__init__(data_loader,args)
        epochs = 5000
        self.opt_params = {'batch_size': 16, 'use_gpu': False, 'epochs_summary': epochs//50, 'epochs_training': epochs, 'learning_rate': 0.001}
        self.opt_params.update(self.args)
        self.net_name = 'basicFC'

    def get_params(self):
        hsm_params = NDNutils.ffnetwork_params(
            input_dims=[1, self.width, self.height], 
            layer_sizes=[int(self.args['hidden']*self.out_num),int(self.args['hidden']*self.out_num), self.out_num], # paper: 9, 0.2*output_shape
            ei_layers=[None, None, None],
            normalization=[0,0, 0], 
            layer_types=['var','normal','normal'],
            act_funcs=['lin','softplus','softplus'],
            reg_list={
                'l2':[0.1,None, self.args['reg_l']],
                'd2x':[None,self.args['reg_h'], None],
                })
        hsm_params['weights_initializers']=['normal','normal','normal']
        hsm_params['biases_initializers']=['trunc_normal','trunc_normal','trunc_normal']

        return hsm_params


class DoGModel(Model):
    def get_paramas(self, channels, filt_size, neurons):
        pass

class ConvDoGModel(Model):
    pass

class ICLRModel(Model):
    def get_params(self):
        params = NDNutils.ffnetwork_params(
            input_dims=[1, self.width, self.height],
            layer_sizes=[self.args['channels'],self.args['channels'],self.args['channels'], self.out_num],
            layer_types=['conv', 'conv', 'conv', 'sep'],
            act_funcs=['softplus', 'softplus', 'lin', 'softplus'],
            shift_spacing=[1, 1, 1, None],
            reg_list={
                #'d2x': [0.03, 0.015, 0.015, None],
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

