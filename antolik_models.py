import NDN3.NDNutils as NDNutils
import NDN3.NDN as NDN
from model import Model

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
                    layer_sizes=[self.args['channels'], int(0.2*self.out_num), self.out_num], # paper: 9, 0.2*output_shape
                    ei_layers=[None, None, None],
                    normalization=[0, 0, 0], 
                    layer_types=['conv', self.args['hidden_lt'], 'normal'],
                    act_funcs=['softplus', 'softplus','softplus'],
                    
                    shift_spacing=[(self.args['c_size']+1)//2, 0],
                    conv_filter_widths=[self.args['c_size'], 0, 0],

                    reg_list={
                        'd2x': [self.args['cd2x'], None , None],
                        self.args['hidden_t']:[None, self.args['hidden_s'], None],
                        'l2':[None, None, 0.1],
                        })
        params['weights_initializers']=['normal','normal','normal']
        params['biases_initializers']=['trunc_normal','trunc_normal','trunc_normal']

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
            layer_sizes=[int(self.args['hidden']*self.out_num), self.out_num], # paper: 9, 0.2*output_shape
            ei_layers=[None, None],
            normalization=[0, 0], 
            layer_types=['normal','normal'],
            act_funcs=['softplus','softplus'],
            reg_list={
                'l2':[None, self.args['reg_l']],
                'd2x':[self.args['reg_h'], None],
                })
        hsm_params['weights_initializers']=['normal','normal']
        hsm_params['biases_initializers']=['trunc_normal','trunc_normal']

        return hsm_params


class DoGModel(Model):
    def get_paramas(self, channels, filt_size, neurons):
        pass

class ConvDoGModel(Model):
    pass
