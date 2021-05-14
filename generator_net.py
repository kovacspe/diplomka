import copy
import numpy as np

from NDN3.layer import VariableLayer
from NDN3.NDN import NDN
from NDN3 import NDNutils


class GeneratorNet:
    def __init__(self, original_nets, input_noise_size, loss='oneside-gaussian', norm='online', is_aegan=False, mask=None, gen_type='conv'):
        # Save parameters
        if not isinstance(original_nets, list):
            original_nets = [original_nets]

        self.original_nets = [copy.deepcopy(net) for net in original_nets]
        self.noise_size = input_noise_size
        self.is_aegan = is_aegan
        if norm not in ['online', 'post', 'none']:
            raise ValueError(
                f'Incorrect norm \'{norm}\'. Norm should be one of online/post/none')
        self.current_norm = norm

        # Assert all networks has only one input of a same shape
        input_stimuli_size = self.original_nets[-1].input_sizes[0]
        for i, network in enumerate(self.original_nets):
            assert len(
                network.input_sizes) == 1, f'Network {i} has more than one input. Input sizes: {network.input_sizes}.'
            assert input_stimuli_size == network.input_sizes[
                0], f'Network {i} has different input shape. Expected {input_stimuli_size}, given {network.input_sizes}'

        # Create generator
        generator = self.get_gan_subnet(
            input_noise_size, input_stimuli_size, gen_type)
        self.generator_subnet_id = 0
        merged_networks = [generator]
        net_prefix_num = 1

        # Merge networks and rewire inputs
        ffnet_out = []
        losses = []
        network_mapping = []
        for i_net, network in enumerate(self.original_nets):
            for i_subnet, subnetwork in enumerate(network.network_list):
                network_mapping.append(
                    ((i_net, i_subnet), len(merged_networks)))
                if subnetwork['ffnet_n'] is not None:
                    subnetwork['ffnet_n'] = [
                        f+net_prefix_num for f in subnetwork['ffnet_n']]
                if subnetwork['xstim_n'] is not None:
                    subnetwork['xstim_n'] = None
                    subnetwork['ffnet_n'] = [0]
                merged_networks.append(subnetwork)
            out_nets = [len(network.network_list)-1 if x == -
                        1 else x for x in network.ffnet_out]
            ffnet_out += [f+net_prefix_num for f in out_nets]
            losses.append(loss)
            net_prefix_num += len(network.network_list)

        # Optionally create decoder
        self.encoder_subnet_id = None
        if is_aegan:
            ffnet_out += [len(merged_networks)]
            merged_networks.append(self.get_encoder(
                input_noise_size, input_stimuli_size, 0))
            self.encoder_subnet_id = len(merged_networks)-1
            losses.append('gaussian')

        # Define new NDN
        self.net_with_generator = NDN(merged_networks,
                                      input_dim_list=[[1, input_noise_size]],
                                      batch_size=self.original_nets[0].batch_size if self.original_nets[
                                          0].batch_size is not None else 265,
                                      ffnet_out=ffnet_out,
                                      noise_dist=losses,
                                      tf_seed=250)

        # Copy weight from original net
        for (i_net, i_subnet), target_i_net in network_mapping:
            GeneratorNet._copy_net_params(
                self.original_nets[i_net], self.net_with_generator, i_subnet, target_i_net)

        # Set mask
        if mask is None:
            # Set all ones mask = no change of input
            mask = np.ones(input_stimuli_size, dtype=np.float32)
        self.net_with_generator.networks[0].layers[-1].weights = mask.astype(
            np.float32)

        # Construct fit vars
        layers_to_skip = []
        for i, net in enumerate(self.net_with_generator.networks):
            if i == self.generator_subnet_id:
                # Fit all except mask
                layers_to_skip.append([len(net.layers)-1])
            elif i == self.encoder_subnet_id:
                # Fit all
                layers_to_skip.append([])
            else:
                # Freeze weights
                layers_to_skip.append([x for x in range(len(net.layers))])

        self.generator_fit_vars = self.net_with_generator.fit_variables(
            layers_to_skip=layers_to_skip, fit_biases=False)

    def train_generator_on_neuron(self, optimize_neuron, data_len, max_activation=None, perc=0.9, epochs=5, noise_input=None, output=None, train_log=None):
        if output is not None and noise_input is None:
            raise ValueError('Output specified, but no input provided')

        # Create input if not specified
        if noise_input is None:
            input_shape = (data_len, self.noise_size)
            # Create input
            noise_input = np.random.normal(size=input_shape, scale=1)
            # TODO: Is normalizing necesary?
            #input_norm = np.sqrt(np.sum(noise_input**2,axis=1))/np.sqrt(self.noise_size)
        else:
            noise_input = noise_input

        # Create output if not specified
        if output is None:
            output_shape = (data_len, self.net_with_generator.output_sizes[0])
            # Create_output
            output = np.zeros(output_shape)
            if max_activation is not None:
                output[:, optimize_neuron] = output[:, optimize_neuron] + \
                    (perc*np.ones(output_shape[0]) * max_activation)

        # Setup data filter to filter only desired neuron
        tmp_filters = np.zeros(
            (data_len, self.net_with_generator.output_sizes[0]))
        tmp_filters[:, optimize_neuron] = 1

        output = [output]
        tmp_filters = [tmp_filters]
        if self.encoder_subnet_id is not None:
            output.append(noise_input)
            tmp_filters.append(
                np.ones((data_len, self.noise_size))
            )
        print(len(output))
        # Set L2-norm on output
        if self.current_norm == 'online':
            # if not isinstance(l2_norm,list):
            #     l2_norm = [l2_norm]
            # if not self.is_aegan:
            self.net_with_generator.networks[self.generator_subnet_id].layers[-1].normalize_output = True
            # else:
            #    self.net_with_generator.networks[-1].layers[0].normalize_output = l2_norm

        # Generator training
        self.net_with_generator.train(
            noise_input,
            output,
            fit_variables=self.generator_fit_vars,
            data_filters=tmp_filters,
            learning_alg='adam',
            train_indxs=np.arange(data_len*0.9),
            test_indxs=np.arange(data_len*0.9, data_len),
            output_dir=train_log,
            opt_params={
                'display': 1,
                'batch_size': 256,
                'epochs_summary': 1,
                'use_gpu': False,
                'epochs_training': epochs,
                'learning_rate': 0.0001
            }
        )
        # print(self.net_with_generator.eval_preds(noise_input,output_data=output))

    def extract_generator(self):
        # Extracts generator as a simple 1-layer net with biases
        generator_subnet = NDN(
            [copy.deepcopy(
                self.net_with_generator.network_list[self.generator_subnet_id])],
            noise_dist='max',
            input_dim_list=[
                self.net_with_generator.network_list[self.generator_subnet_id]['input_dims']]
        )  

        # Copy weights
        GeneratorNet._copy_net_params(
            self.net_with_generator,
            generator_subnet,
            self.generator_subnet_id,
            0
        )
        if self.current_norm == 'post':
            generator_subnet.networks[-1].layers[-1].normalize_output = True

        return generator_subnet

    def generate_stimulus(
        self,
        num_samples=1000,
        noise_input=None
    ):
        # Generate noise_input if not specified
        if noise_input is None:
            noise_input = np.random.uniform(-2, 2,
                                            size=(num_samples, self.noise_size))
        generator = self.extract_generator()
        image_out = generator.generate_prediction(noise_input)
        return image_out

    @staticmethod
    def _copy_net_params(original_NDN_net, target_NDN_net, net_num_original, net_num_target):
        for layer_source, layer_target in zip(original_NDN_net.networks[net_num_original].layers, target_NDN_net.networks[net_num_target].layers):
            layer_target.copy_layer_params(layer_source)

    def get_gan_subnet(self, input_noise_size, output_shape, generator_type='conv'):
        output_shape = output_shape[1:]
        layers=5
        if generator_type == 'conv':
            params = NDNutils.ffnetwork_params(
                input_dims=[1, input_noise_size],
                layer_sizes=[[64, 8, 8], 32, 16, 1, 1],
                layer_types=['normal', 'deconv', 'deconv', 'deconv', 'mask'],
                act_funcs=['relu', 'relu', 'relu', 'tanh', 'lin'],
                conv_filter_widths=[None, 5, 5, 5, None],
                shift_spacing=[None, 2, 2, 1, None],
                reg_list={
                    'd2x': [None, None, 0.01, 0.01, None]
                },
                verbose=False
            )
            params['output_shape'] = [None, None,
                                      output_shape, output_shape, None]

        elif generator_type =='deepconv':
            params = NDNutils.ffnetwork_params(
                input_dims=[1, input_noise_size],
                layer_sizes=[[512, 4, 4], 256, 128, 1, 1],
                layer_types=['normal', 'deconv', 'deconv','deconv', 'mask'],
                act_funcs=['relu', 'relu', 'relu','tanh', 'lin'],
                conv_filter_widths=[None, 5, 5, 5, None],
                shift_spacing=[None, 2, 2, 2, None],
                reg_list={
                    'd2x': [None, None, None, 0.01, None]
                },
                verbose=False
            )
            params['output_shape'] = [None, None,
                                      None, output_shape, None]

        elif generator_type == 'lin' or generator_type=='lin_tanh':
            act = 'lin' if generator_type=='lin' else 'tanh'
            params = NDNutils.ffnetwork_params(
                input_dims=[1, input_noise_size],
                layer_sizes=[512, 1024, [1, 31, 31], 1],
                layer_types=['normal', 'normal', 'normal', 'mask'],
                act_funcs=['relu', 'relu', act, 'lin'],
                reg_list={
                    'l2': [0.01, 0.01, 0.01, None],
                },
                verbose=False
            )
            layers=4

        elif generator_type == 'hybrid':
            params = NDNutils.ffnetwork_params(
                input_dims=[1, input_noise_size],
                layer_sizes=[256, [16, 16, 16], 8, 1, 1],
                layer_types=['normal', 'normal', 'deconv', 'deconv', 'mask'],
                act_funcs=['relu', 'relu', 'relu', 'tanh', 'lin'],
                conv_filter_widths=[None, 5, 5, 5, None],
                shift_spacing=[None, 2, 2, 1, None],
                reg_list={
                    'd2x': [None, None, 0.01, 0.01, None]
                },
                verbose=False
            )
            params['output_shape'] = [None, None,
                                      output_shape, output_shape, None]
        else:
            raise ValueError(
                f'Generator type \'{generator_type}\' not implemented.')

        params['xstim_n'] = [0]
        params['normalize_output'] = [None]*layers
        params['weights_initializers'] = ['normal']*(layers-1)+['ones']
        params['biases_initializers'] = ['zeros']*layers
        return params

    def get_encoder(self, noise_size, input_shape, ffnet_in, generator_type='conv'):
        if generator_type=='conv':
            params = NDNutils.ffnetwork_params(
                input_dims=input_shape,
                layer_sizes=[8, 8, 16, noise_size],
                layer_types=['conv', 'conv', 'conv', 'normal'],
                act_funcs=['relu', 'relu', 'relu', 'lin'],
                conv_filter_widths=[5, 5, 7, None],
                shift_spacing=[1, 2, 2, None],
                reg_list={
                    'd2x': [0.1, 0.1, None, None]
                },
                verbose=False
            )
        elif generator_type=='lin':
            params = NDNutils.ffnetwork_params(
                input_dims=input_shape,
                layer_sizes=[8, 8, 16, noise_size],
                layer_types=['normal', 'normal', 'normal'],
                act_funcs=['relu', 'relu', 'relu',],
                reg_list={
                    'd2x': [0.1, 0.1, None, None]
                },
                verbose=False
            )
        elif generator_type=='hybrid':
            params = NDNutils.ffnetwork_params(
                input_dims=input_shape,
                layer_sizes=[8, 8, 16, noise_size],
                layer_types=['conv', 'conv', 'normal', 'normal'],
                act_funcs=['relu', 'relu', 'relu', 'lin'],
                conv_filter_widths=[5, 5, 7, None],
                shift_spacing=[2, 2, None, None],
                reg_list={
                    'd2x': [0.1, 0.1, None, None]
                },
                verbose=False
            )
        else:
            raise ValueError(f'Generator type \'{generator_type}\' not implemented.')
        
        params['xstim_n'] = None
        params['ffnet_n'] = [ffnet_in]
        return params
