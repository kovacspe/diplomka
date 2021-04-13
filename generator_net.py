import copy
import numpy as np

from NDN3.layer import VariableLayer
from NDN3.NDN import NDN
from NDN3 import NDNutils

class GeneratorNet:
    def __init__(self,original_netx,input_noise_size,loss='oneside-gaussian',is_aegan=False):
        # Save parameters
        self.original_net = copy.deepcopy(original_netx)
        self.noise_size = input_noise_size
        self.is_aegan = is_aegan
        # Copy network list
        networks = self.original_net.network_list.copy()
        
        # Find ffnetworks, which are connected to input
        input_net = []
        for inet, net_param in enumerate(networks):
            if net_param['xstim_n'] is not None:
                if len(net_param['xstim_n'])>1 or net_param['xstim_n'][0]>0:
                    raise NotImplementedError('GAN does not support multiple inputs. Original net should have only one visual input. Current input: '+str(net_param['xstim_n']))
                input_net.append((inet, net_param['xstim_n']))

        #
        gan_net_shift = len(networks)
        gan_nets = []
        encoders = []
        
        # Add first layer into ff_net params
        for generator_id,inp_size in enumerate(self.original_net.input_sizes):
            gan_nets.append(self.get_gan_subnet(input_noise_size, inp_size))
            if is_aegan:
                encoders.append(self.get_encoder(input_noise_size,inp_size,generator_id))
        num_generator_nets = len(gan_nets)
        self.generator_subnet_ids = list(range(num_generator_nets))

        networks = gan_nets+networks
        self.input_nets = [len(gan_nets) + i for i,_ in input_net]

        # Rewire nets
        for i,net in enumerate(networks):
            if i>=num_generator_nets:
                if networks[i]['xstim_n'] is not None:
                    stims = [inp for inp in networks[i]['xstim_n']] 
                else:
                    stims = []
                if networks[i]['ffnet_n'] is not None:
                    ffnets = [inp+num_generator_nets for inp in networks[i]['ffnet_n']]
                else:
                    ffnets = []
                networks[i]['ffnet_n'] = stims+ffnets
                if len(networks[i]['ffnet_n'])==0:
                    networks[i]['ffnet_n'] = None
                networks[i]['xstim_n'] = None

        # Add encoders
        

        # Infer output networks
        ffnet_out=len(networks)-1
        self.encoder_subnet_ids = list(range(len(networks),len(networks)+len(encoders)))
        networks = networks + encoders
        # print(networks)
        # quit()
        if is_aegan:
            ffnet_out = [ffnet_out] + self.encoder_subnet_ids
            loss = [loss] + ['gaussian']*len(encoders)
        # Define new NDN
        self.net_with_generator = NDN(networks,
                        input_dim_list=[[1, input_noise_size]],
                        batch_size=self.original_net.batch_size if self.original_net.batch_size is not None else 265,
                        ffnet_out=ffnet_out,
                        noise_dist=loss,
                        tf_seed=250)

        # Copy weight from original net
        for i in range(num_generator_nets,len(networks)-len(encoders)):
            GeneratorNet._copy_net_params(self.original_net, self.net_with_generator, i-num_generator_nets, i)

        # Construct fit vars
        layers_to_skip = []
        for i, net in enumerate(self.net_with_generator.networks):
            if i in self.generator_subnet_ids or i in self.encoder_subnet_ids:
                layers_to_skip.append([])
            else:
                layers_to_skip.append([x for x in range(len(net.layers))])
        
        self.generator_fit_vars = self.net_with_generator.fit_variables(
            layers_to_skip=layers_to_skip, fit_biases=False)

    def train_generator_on_neuron(self,optimize_neuron,data_len,l2_norm=None,max_activation=None,perc=0.9,epochs=5,noise_input=None,output=None):
        if output is not None and noise_input is None:
            raise ValueError('Output specified, but no input provided')

        # Create input if not specified
        if noise_input is None:
            input_shape = (data_len, self.noise_size)
            # Create input
            noise_input = np.random.normal(size=input_shape,scale=1)
            # TODO: Is normalizing necesary?
            input_norm = np.sqrt(np.sum(noise_input**2,axis=1))/np.sqrt(self.noise_size)
        else:
            noise_input = noise_input

        # Create output if not specified
        if output is None:
            output_shape = (data_len, self.net_with_generator.output_sizes[0])
            #Create_output
            output = np.zeros(output_shape)
            if max_activation is not None:
                output[:,optimize_neuron] = output[:,optimize_neuron]+(perc*np.ones(output_shape[0]) *max_activation)

        # Setup data filter to filter only desired neuron
        tmp_filters = np.zeros((data_len, self.net_with_generator.output_sizes[0]))
        tmp_filters[:, optimize_neuron] = 1

        output = [output]
        tmp_filters = [tmp_filters]
        for encoder_id in self.encoder_subnet_ids:
            output.append(noise_input)
            tmp_filters.append(
                np.ones((data_len, self.noise_size))
            )
        # Set L2-norm on output
        if l2_norm is not None:
            if not isinstance(l2_norm,list):
                l2_norm = [l2_norm]
            if not self.is_aegan:
                for subnet_id, subnet_l2_norm in zip(self.generator_subnet_ids,l2_norm):
                    self.net_with_generator.networks[subnet_id].layers[-1].normalize_output = subnet_l2_norm
            else:
                for inp_net, subnet_l2_norm in zip(self.input_nets,l2_norm):
                    self.net_with_generator.networks[inp_net].layers[0].normalize_output = subnet_l2_norm
        self.current_l2_norm = l2_norm

        # Generator training
        self.net_with_generator.train(
            noise_input, 
            output, 
            fit_variables=self.generator_fit_vars,
            data_filters=tmp_filters, 
            learning_alg='adam',
            train_indxs=np.arange(data_len*0.9),
            test_indxs=np.arange(data_len*0.9,data_len),
            opt_params={
                'display': 1,
                'batch_size': 256, 
                'use_gpu': False, 
                'epochs_training': epochs , 
                'learning_rate': 0.001
            }
        )


    def extract_generator(self,generator_subnet_id=0):
        # Check if subnet on id `generator_subnet_id` is a generator
        if generator_subnet_id not in self.generator_subnet_ids:
            raise IndexError(f'Subnet with id {generator_subnet_id} is not a generator')   
        if self.net_with_generator.network_list[generator_subnet_id]['xstim_n'] is None:
            raise AttributeError(f'Network {generator_subnet_id} is not connected to input')
        # Extracts generator as a simple 1-layer net with biases
        generator_subnet = NDN(
            [self.net_with_generator.network_list[generator_subnet_id]],
            noise_dist='max',
            input_dim_list=[self.net_with_generator.network_list[generator_subnet_id]['input_dims']]
        )
        print('l2 norm: '+str(self.current_l2_norm))
        if self.current_l2_norm is not None:
            generator_subnet.networks[-1].layers[-1].normalize_output = self.current_l2_norm

        # Copy weights
        GeneratorNet._copy_net_params(
            self.net_with_generator,
            generator_subnet,
            generator_subnet_id,
            0
        )
        print('l2 norm: '+str(self.current_l2_norm))
        if self.current_l2_norm is not None:
            generator_subnet.networks[-1].layers[-1].normalize_output = self.current_l2_norm

        return generator_subnet
    
    def generate_stimulus(
        self,
        generator_subnet_id=0,
        num_samples=1000,
        noise_input=None
    ):
        # Generate noise_input if not specified
        if noise_input is None:
            noise_input = np.random.uniform(-2,2,size=(num_samples,self.noise_size))
        generator = self.extract_generator(generator_subnet_id)
        image_out = generator.generate_prediction(noise_input)
        return image_out


    @staticmethod
    def _copy_net_params(original_NDN_net, target_NDN_net, net_num_original, net_num_target):
        for layer_source, layer_target in zip(original_NDN_net.networks[net_num_original].layers, target_NDN_net.networks[net_num_target].layers):
            layer_target.copy_layer_params(layer_source)


    def get_gan_subnet(self,input_noise_size, output_shape):
        output_shape = output_shape[1:]
        out = [16,8,8]
        params = NDNutils.ffnetwork_params(
            input_dims=[1, input_noise_size],
            layer_sizes=[out,8,8,1], 
            layer_types=['normal','deconv','deconv','deconv'],
            act_funcs=['relu','relu','relu','tanh'],
            conv_filter_widths=[None,5,5,5],
            shift_spacing=[None,2,2,1],
            reg_list={
                #'l2':[0.001,None,None,None],
                'd2x': [None,None,0.1,0.1]
            },
            verbose=False
        )
            
        params['xstim_n'] = [0]
        params['normalize_output'] =  [None,None,None,None]

        params['output_shape'] = [None,None,output_shape,output_shape]
        return params

    def get_encoder(self,noise_size, input_shape,ffnet_in):
        params = NDNutils.ffnetwork_params(
            input_dims= input_shape,
            layer_sizes=[8,8,16,noise_size], 
            layer_types=['conv','conv','conv', 'normal'],
            act_funcs=['relu','relu','relu','lin'],
            conv_filter_widths=[5,5,7,None],
            shift_spacing=[1,2,2,None],
            reg_list={
                #'l2':[0.001,None,None,None],
                'd2x': [0.1,0.1,None,None]
            },
            verbose=False
        )
        params['xstim_n'] = None
        params['ffnet_n'] = [ffnet_in]

        return params
