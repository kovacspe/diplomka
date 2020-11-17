import numpy as np
import tensorflow as tf
from NDN3.layer import VariableLayer
import matplotlib.pyplot as plt

def find_var_layer(net):
    var_layer_net = None
    var_layer_layer = None
    for i, network in enumerate(net.networks):
        for j, layer in enumerate(network.layers):
            if isinstance(layer,VariableLayer):
                if var_layer_net is None:
                    var_layer_net = i
                    var_layer_layer = j
                else:
                    raise AssertionError(
                        'Network has more than one Variable layer')
    if var_layer_net is None:
        raise AssertionError('Network has no Variable layer')
    return var_layer_net,var_layer_layer

def find_MEI(net, optimize_neuron):
    # Find Variable layer
    var_layer_net,var_layer_layer = find_var_layer(net)

    # Activate variable layer
    net.network_list[var_layer_net]['as_var'] = True

    # Change loss function
    original_noise_dist = net.noise_dist
    net.noise_dist = 'max'

    # Get network input and output shapes
    input_shape = (1,np.prod(net.input_sizes))
    output_shape = (1, np.prod(net.output_sizes))

    # Create dummy input and output
    net.batch_size = 1
    dummy_input = np.zeros(input_shape)
    dummy_output = np.zeros(output_shape)

    # Reset Variable Layer initial weights
    net.networks[var_layer_net].layers[var_layer_layer].weights = np.random.normal(size=input_shape, scale=0.1).astype('float32')
    
    # Setup data filter to filter only desired neuron
    tmp_filters = np.zeros((1, np.prod(output_shape)))
    tmp_filters[0, optimize_neuron] = 1
    print(tmp_filters)

    # Weight of variable layer as the only training variable
    layers_to_skip = []
    for i,network in enumerate(net.networks):
        if var_layer_net==i:
            l=[]
            for x in range(len(network.layers)):
                if x!=var_layer_layer:
                    l.append(x)
            layers_to_skip.append(l)
        else:
            layers_to_skip.append([x for x in range(len(network.layers))])
    
    fit_vars = net.fit_variables(layers_to_skip=layers_to_skip, fit_biases=False)

    # Optimize input (Variable layer)
    net.train(dummy_input, dummy_output, fit_variables=fit_vars,
          data_filters=tmp_filters, learning_alg='lbfgs')

    # Restore original settings
    net.network_list[var_layer_net]['as_var'] = False
    net.noise_dist = original_noise_dist
    return net


def plot_filter(net):
    var_layer_net,var_layer_layer = find_var_layer(net)
    w = np.reshape(net.networks[var_layer_net].layers[var_layer_layer].weights, (5, 5))
    plt.imshow(w,vmin=-1,vmax=1,cmap=plt.cm.RdYlBu)
    plt.show()
