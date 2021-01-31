from NDN3.layer import VariableLayer
from NDN3.NDN import NDN
from NDN3 import NDNutils

class GAN:
    def __init__(self,original_net,input_noise_size,loss):
        networks = original_net.network_list[:]
        # Find one and only one input ffnetwork
        input_net = []
        for inet, net_param in enumerate(networks):
            if net_param['xstim_n'] is not None:
                input_net.append((inet, net_param['xstim_n']))

        gan_net_shift = len(networks)
        gan_nets=[]
        # Add first layer into ff_net params
        for inp_size in original_net.input_sizes:
            gan_nets.append(generate_gan_net(input_noise_size, inp_size))
        num_gan_nets = len(gan_nets)

        networks = gan_nets+networks
        # Rewire nets
        for i,net in enumerate(networks):
            if i>=num_gan_nets:
                if networks[i]['xstim_n'] is not None:
                    stims = [inp for inp in networks[i]['xstim_n']] 
                else:
                    stims = []
                if networks[i]['ffnet_n'] is not None:
                    ffnets = [inp+num_gan_nets for inp in networks[i]['ffnet_n']]
                else:
                    ffnets = []
                networks[i]['ffnet_n'] = stims+ffnets
                print(f'FFNETS for {i}: ',networks[i]['ffnet_n'])
                if len(networks[i]['ffnet_n'])==0:
                    networks[i]['ffnet_n'] = None
                networks[i]['xstim_n'] = None
        print(networks)

        # Define new NDN
        new_net = NDN(networks,
                        input_dim_list=[[1, input_noise_size]],
                        batch_size=original_net.batch_size if original_net.batch_size is not None else 265,
                        noise_dist=loss,
                        tf_seed=250)

        # Copy weight from original net
        for i in range(num_gan_nets,len(networks)):
            _copy_net_params(original_net, new_net, i-num_gan_nets, i)

        # Construct fit vars
        layers_to_skip = []
        for i, net in enumerate(new_net.networks):
            if i >= num_gan_nets:
                layers_to_skip.append([x for x in range(len(net.layers))])
            else:
                layers_to_skip.append([])

        self.fit_vars = new_net.fit_variables(
            layers_to_skip=layers_to_skip, fit_biases=False)
        self.net = new_net


    def auto_train(self):
        pass

    def manual_train(self,input,output):
        pass

    def extract_generator(self):
        # Extracts generator as a simple 1-layer net with biases
        if self.net.network_list[gan_subnet_id]['xstim_n'] is None:
            raise AttributeError(f'Network {gan_subnet_id} is not connected to input')
        gan_only = NDN([self.net.network_list[gan_subnet_id]],
        noise_dist='max',input_dim_list=[self.net.network_list[gan_subnet_id]['input_dims']])

        # Copy weights
        _copy_net_params(self.net,gan_only,gan_subnet_id,0)

        return gan_only

    @staticmethod
    def _copy_net_params(original_NDN_net, target_NDN_net, net_num_original, net_num_target):
        for layer_source, layer_target in zip(original_NDN_net.networks[net_num_original].layers, target_NDN_net.networks[net_num_target].layers):
            layer_target.copy_layer_params(layer_source)

    @staticmethod
    def generate_gan_subnet(input_noise_size, output_shape):
        generator_params = NDNutils.ffnetwork_params(
            input_dims=[1, input_noise_size],
            layer_sizes=[output_shape],
            normalization=[0],
            layer_types=['normal'],
            act_funcs=['tanh'],
            shift_spacing=[1],
            conv_filter_widths=[0],
            reg_list={
                'l2': [0.1]
            },
            verbose=False)
        params['xstim_n'] = [0]

        bias_params = NDNutils.ffnetwork_params(
            input_dims=[1, output_shape],
            layer_sizes=[output_shape],
            normalization=[0],
            layer_types=['var'],
            act_funcs=['lin'],
            shift_spacing=[1],
            conv_filter_widths=[0],
            reg_list={
                'l2': [0.1]
            },
            verbose=False)
        bias_params['xstim_n'] = [0]
        bias_params['as_var'] = True

        add_params = NDNutils.ffnetwork_params(
            input_dims=[1, input_noise_size],
            layer_sizes=[output_shape],
            normalization=[0],
            layer_types=['add'],
            act_funcs=['tanh'],
            shift_spacing=[1],
            conv_filter_widths=[0],
            reg_list={
                'l2': [0.1]
            },
            verbose=False)
        add_params['xstim_n'] = None
        add_params['ffnet_n']=[0,1]
        add_params['weights_initializer'] = 'ones'
        return generator_params,bias_params,add_params
