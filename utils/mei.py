import copy
import math

import numpy as np
import tensorflow as tf

from NDN3.layer import VariableLayer
from NDN3.NDN import NDN

from .experiment_params import experiment_args
from .misc_utils import norm_samples
from .plotting import plot_grid


def get_filter(net, reshape=True):
    """
    Extracts image from Variable layes weights
        Parameters:
            net (NDN): A trained neural network with variable layer
            reshape (bool): If true reshape weights to image sizes
        Returns:
            filter (np.array): Filter from weights of variable layer.
    """
    var_layer_net, var_layer_layer = find_var_layer(net)
    _, x, y = net.input_sizes[0]
    raw_filter = net.networks[var_layer_net].layers[var_layer_layer].weights
    if reshape:
        return np.reshape(raw_filter, (x, y))
    else:
        return raw_filter


def fourier_filter(sizex, sizey, alpha):
    """
    Creates fourier filter as described in https://www.nature.com/articles/s41593-019-0517-x?proof=t#ref-CR15
    """
    fy = np.fft.fftfreq(sizey)[:, None]
    fx = np.fft.fftfreq(sizex)[:]
    freqs = (fx * fx + fy * fy)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(sizex, sizey)) ** alpha
    scale /= (2*math.pi)**2
    return scale


def define_mei_train_step(net, var_layer, sizex, sizey, alpha):
    """
    Defines training step of MEI activation maximization. Applies gradient filtering
        Parameters:
            net (NDN): A trained neural network with variable layer
            var_layer (NDN.Layer): Variable layer
            sizex (int): stimuli width
            sizex (int): stimuli height
            alpha (float): coefficient for fourier filter

        Returns:
            train_step: tensorflow train step
            learning_rate (tf.placeholder): Placeholder for learning rate
    """
    learning_rate = tf.placeholder(tf.float32)
    cost = -net.cost
    # Get gradients
    d_weights = tf.reduce_mean(
        tf.gradients(cost, [var_layer.weights_var]),
        axis=0)

    # Filter gradient in fourier domain
    d_in_fourier_domain = tf.signal.fft2d(
        tf.cast(
            tf.reshape(d_weights, (sizex, sizey)),
            dtype=tf.complex64
        )
    )
    d_in_fourier_domain_filtered = tf.multiply(
        d_in_fourier_domain,
        fourier_filter(sizex, sizey, alpha)
    )
    d_weights_filtered = tf.cast(
        tf.reshape(
            tf.ifft2d(d_in_fourier_domain_filtered),
            (1, -1)
        ),
        dtype=tf.float32
    )

    # Apply gradient
    train_step = var_layer.weights_var.assign(
        var_layer.weights_var + learning_rate*d_weights_filtered
    )
    return train_step, learning_rate


def find_var_layer(net):
    """
    Find Variable layer in NDN network
        Parameters:
            net (NDN): A trained neural network with variable layer

        Returns:
            var_layer_net (NDN.FFnetwork): FFnetwork object with variable layer in it
            learning_rate (NDN.Layer): Layer object instance of  a Variable layer
    """
    var_layer_net = None
    var_layer_layer = None
    for i, network in enumerate(net.networks):
        for j, layer in enumerate(network.layers):
            if isinstance(layer, VariableLayer):
                if var_layer_net is None:
                    var_layer_net = i
                    var_layer_layer = j
                else:
                    raise AssertionError(
                        'Network has more than one Variable layer')
    if var_layer_net is None:
        raise AssertionError('Network has no Variable layer')
    return var_layer_net, var_layer_layer


def train_MEI(net: NDN, input_data, output_data, data_filters, opt_args, fit_vars, var_layer):
    """
    Optimization of weights in variable layer to maximize neurons output
        Parameters:
            net (NDN): A trained neural network with variable layer
            input_data (np.array): input data
            output_data (np.array): output data
            data_filters (np.array): data_filters will be passed to NDN.train
            opt_args (dict): optimizer arguments
            fit_vars (dict): fit variables will be passed to NDN.train
            var_layer (NDN.Layer): Variable layer

    """
    opt_params = net.optimizer_defaults(
        opt_args['opt_params'], opt_args['learning_alg'])
    epochs = opt_args['opt_params']['epochs_training']
    lr = opt_args['opt_params']['learning_rate']
    _, sizex, sizey = net.input_sizes[0]
    input_data, output_data, data_filters = net._data_format(
        input_data, output_data, data_filters)
    net._build_graph(
        opt_args['learning_alg'],
        opt_params,
        fit_vars,
        batch_size=1
    )

    with tf.Session(graph=net.graph, config=net.sess_config) as sess:
        # Define train step
        train_step, learning_rate = define_mei_train_step(
            net, var_layer, sizex, sizey, 0.1)

        # Restore parameters and setup metrics
        net._restore_params(sess, input_data, output_data, data_filters)
        i = 0
        cost = float('inf')
        best_cost = float('inf')
        without_increase = 0

        # Training
        while without_increase < 100 and i < epochs:
            sess.run(train_step, feed_dict={
                     net.indices: [0], learning_rate: lr})
            cost = sess.run(net.cost, feed_dict={net.indices: [0]})
            cost_reg = sess.run(net.cost_reg, feed_dict={net.indices: [0]})
            if i % 20 == 0:
                print(
                    f'Cost: {cost:.5f}+{cost_reg:.5f}, best: {best_cost:.3f}')
            if i % 10 == 0:
                # Apply gausssian blur
                sess.run(
                    var_layer.gaussian_blur_std_var.assign(
                        var_layer.gaussian_blur_std_var*0.99
                    )
                )
            if cost < best_cost:
                best_cost = cost+cost_reg
                without_increase = 0

            sess.run(var_layer.gaussian_blur)
            i += 1
            without_increase += 1

        # Save trained weights
        net._write_model_params(sess)

    print(f'Trained with {i} epochs')


def find_MEI(net, optimize_neuron, epochs=400):
    """
    Optimization of weights in variable layer to maximize neurons output
        Parameters:
            net (NDN): A trained neural network with variable layer
            optimize_neuron (int): Neuron id
            epochs (int): Number of epochs for training
    """
    # Find Variable layer
    var_layer_net, var_layer_layer = find_var_layer(net)

    # Activate variable layer
    net.network_list[var_layer_net]['as_var'] = True

    # Change loss function
    original_noise_dist = net.noise_dist
    net.noise_dist = 'max'

    # Get network input and output shapes
    input_shape = (1, np.prod(net.input_sizes))
    output_shape = (1, np.prod(net.output_sizes))

    # Create dummy input and output
    net.batch_size = 1
    dummy_input = np.zeros(input_shape)
    dummy_output = np.zeros(output_shape)

    # Reset Variable Layer initial weights
    net.networks[var_layer_net].layers[var_layer_layer].weights = np.random.normal(
        size=input_shape, scale=1).astype('float32')

    # Setup data filter to filter only desired neuron
    tmp_filters = np.zeros((1, np.prod(output_shape)))
    tmp_filters[0, optimize_neuron] = 1

    # Weight of variable layer as the only training variable
    layers_to_skip = []
    for i, network in enumerate(net.networks):
        if var_layer_net == i:
            l = []
            for x in range(len(network.layers)):
                if x != var_layer_layer:
                    l.append(x)
            layers_to_skip.append(l)
        else:
            layers_to_skip.append([x for x in range(len(network.layers))])

    fit_vars = net.fit_variables(
        layers_to_skip=layers_to_skip, fit_biases=False)

    opt_args = {
        'learning_alg': 'adam',
        'train_indxs': np.arange(1),
        'test_indxs': np.arange(1),
        'opt_params': {

            'display': 1000,
            'batch_size': 1,
            'use_gpu': False,
            'epochs_training': epochs,
            'learning_rate': 0.0001
        }
    }
    # Optimize input (Variable layer)
    train_MEI(net, dummy_input, dummy_output, tmp_filters, opt_args,
              fit_vars, net.networks[var_layer_net].layers[var_layer_layer])

    # Restore original settings
    net.network_list[var_layer_net]['as_var'] = False
    net.noise_dist = original_noise_dist
    return net


@experiment_args
def generate_mei(net, dataset, experiment='000'):
    """
    Generate MEI for every neuron
        Parameters:
            dataset (DataLoader): dataset 
            net (NDN): trained Neural network
            experiment (str): experiment ID
    """
    _, y = dataset.train()
    num_neurons = np.shape(y)[1]
    net2 = copy.deepcopy(net)
    meis = []
    meis_n = []
    activations = []
    activations_n = []
    for neuron in range(num_neurons):
        net = find_MEI(net, neuron, 80)
        mei = get_filter(net)
        mei_activation = net2.generate_prediction(
            np.reshape(mei, (1, -1)))[0, neuron]
        mei_normalized = norm_samples(mei)
        mei_activation_normalized = net2.generate_prediction(
            np.reshape(mei_normalized, (1, -1)))[0, neuron]
        meis.append(mei)
        activations.append(mei_activation)
        meis_n.append(mei_normalized)
        activations_n.append(mei_activation_normalized)

    titles = [f'{i} - {act:.3f}' for i, act in enumerate(activations_n)]
    np.save(f'output/04_mei/{experiment}_mei.npy', meis)
    np.save(f'output/04_mei/{experiment}_mei_activations.npy', activations)
    np.save(f'output/04_mei/{experiment}_mei_n.npy', meis_n)
    np.save(f'output/04_mei/{experiment}_mei_activations_n.npy', activations_n)

    plot_grid(meis_n, titles, num_cols=8,
              save_path=f'output/04_mei/{experiment}_mei.png', show=True)
