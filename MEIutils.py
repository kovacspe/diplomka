import numpy as np
import tensorflow as tf
from NDN3.layer import VariableLayer
import matplotlib.pyplot as plt
import numpy.linalg
from utils import evaluate_performance
from scipy import stats
from NDN3.NDN import NDN
from data_loaders import AntolikDataLoader
from NDN3 import NDNutils
from sklearn.cluster import KMeans,AgglomerativeClustering
import os
from sklearn.metrics.pairwise import pairwise_distances
from generator_net import GeneratorNet
import fire


def STA_LR(training_inputs, training_set, laplace_bias):
    """
    This function implements the spike triggered averaging with laplacian regularization.
    It takes the training inputs and responses, and returns the estimated kernels.
    """
    kernel_size = numpy.shape(training_inputs)[1]
    laplace = laplaceBias(int(numpy.sqrt(kernel_size)),
                          int(numpy.sqrt(kernel_size)))
    return numpy.linalg.pinv(training_inputs.T*training_inputs + laplace_bias*laplace) * training_inputs.T * training_set


def laplaceBias(sizex, sizey):
    S = numpy.zeros((sizex*sizey, sizex*sizey))
    for x in range(0, sizex):
        for y in range(0, sizey):
            norm = numpy.mat(numpy.zeros((sizex, sizey)))
            norm[x, y] = 4
            if x > 0:
                norm[x-1, y] = -1
            if x < sizex-1:
                norm[x+1, y] = -1
            if y > 0:
                norm[x, y-1] = -1
            if y < sizey-1:
                norm[x, y+1] = -1
            S[x*sizex+y, :] = norm.flatten()
    S = numpy.mat(S)
    return S*S.T


def find_var_layer(net):
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

def find_MEI(net, optimize_neuron):
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

    # Optimize input (Variable layer)
    net.train(
        dummy_input, 
        dummy_output, 
        fit_variables=fit_vars,
        data_filters=tmp_filters, 
        learning_alg='adam',
        train_indxs=np.arange(1),
        test_indxs=np.arange(1),
        opt_params={
                'display': 1,
                'batch_size': 1, 
                'use_gpu': False, 
                'epochs_training': 400 , 
                'learning_rate': 0.001
            }
        )

    # Restore original settings
    net.network_list[var_layer_net]['as_var'] = False
    net.noise_dist = original_noise_dist
    return net


def plot_filter(net, ax=None, n=None):
    w = get_filter(net)
    if ax is None:
        plt.imshow(w, cmap=plt.cm.RdYlBu)
        plt.show()

    else:
        return plt.imshow(w, cmap=plt.cm.RdYlBu)


def get_filter(net,reshape=True):
    var_layer_net, var_layer_layer = find_var_layer(net)
    _, x, y = net.input_sizes[0]
    raw_filter = net.networks[var_layer_net].layers[var_layer_layer].weights
    if reshape:
        return np.reshape(raw_filter, (x, y))
    else:
        return raw_filter


def plot_all(net, MEI_STA=False):
    outputs = np.prod(net.output_sizes)
    n_rows = int(np.ceil(np.sqrt(outputs)))
    fig, ax1 = plt.subplots(n_rows, n_rows)
    for i, neuron in enumerate(range(10)):
        print(f'{i}/{outputs}')
        if MEI_STA:
            w = STA(net, neuron)
        else:
            net = find_MEI(net, neuron)
            w = get_filter(net)
        ax1[i % n_rows, i//n_rows].imshow(w, cmap=plt.cm.RdYlBu)
    plt.show()


def STA(net, optimize_neuron, num_examples=2000):
    # Get network input and output shapes
    input_shape = (num_examples, np.prod(net.input_sizes))
    output_shape = (1, np.prod(net.output_sizes))

    # Create dummy input and output
    noise_input = np.random.laplace(
        size=input_shape, scale=1).astype('float32')

    # Predict responses
    predictions = net.generate_prediction(noise_input)[:, optimize_neuron]
    # Weight
    average_response = STA_LR(noise_input, predictions, 0.5)
    print(np.shape(average_response))
    # np.average(noise_input,weights=predictions,axis=0)
    _, x, y = net.input_sizes[0]
    return np.reshape(average_response, (x, y))


class NeuralResult:
    def __init__(self, i, sta, corr_sta, mei, corr_mei, max_activation=None):
        self.i = i
        self.corr_mei = corr_mei
        self.corr_sta = corr_sta
        self.sta = sta
        self.mei = mei
        self.mei_max_activation = max_activation

    def plot(self, ax1, x, y):
        ax1[x, y].imshow(self.sta, cmap=plt.cm.RdYlBu)
        ax1[x, y].title.set_text(f'{self.i} {self.corr_sta:.2f}')
        ax1[x, y+1].imshow(self.mei, cmap=plt.cm.RdYlBu)
        ax1[x, y+1].title.set_text(f'{self.i} {self.corr_mei:.2f}')

    def __str__(self):
        return f'Neuron {self.i:2}: STA: {self.corr_sta:2f} NN: {self.corr_mei:2f}'


def compare_sta_mei(net):

    # Load data
    loader = AntolikDataLoader('data', 1)
    x, y = loader.train()
    test_x, test_y = loader.val()
    print(test_x)
    x = np.matrix(x)
    y = np.matrix(y)

    # Get unit correlations
    net.batch_size = 50
    pred = net.generate_prediction(test_x)

    unit_corr = np.zeros(test_y.shape[1])
    for i in range(test_y.shape[1]):
        unit_corr[i] = stats.pearsonr(test_y[:, i], pred[:, i])[0]

    # STA
    sta = STA_LR(x, y, 10000)
    test_x = np.matrix(test_x)
    # Plot setup
    n_rows = 8  # int(np.ceil(np.sqrt(np.shape(sta)[1])))
    fig, ax1 = plt.subplots(n_rows, 2*n_rows)
    results = []
    sta_corrs = []
    # Process each neuron
    for i, neuron in enumerate(range(np.shape(sta)[1])):
        # Reshape STA RF and count correlation

        pred = np.array(test_x*sta[:, i])
        sta_corr = stats.pearsonr(pred[:, 0], test_y[:, i])[0]
        sta_corrs.append(sta_corr)
        sta_resh = np.reshape(sta[:, i], (31, 31))
        # Find MEI
        net = find_MEI(net, i)
        mei = get_filter(net)

        # Create result
        res = NeuralResult(i, sta_resh, sta_corr, mei, unit_corr[i])

        results.append(res)
    print('net correlation: ', np.mean(unit_corr))
    print('sta correlation: ', np.mean(sta_corrs))
    results.sort(key=lambda r: r.corr_mei-r.corr_sta, reverse=True)
    for r in results:
        print(r)
    for i, res in enumerate(results[:64]):
        res.plot(ax1, i % n_rows, 2*(i//n_rows))

    plt.show()

def plot_rfs(image_out,activations,save_path,scale_by_first=True,plot_diff=False):
    
    vmin,vmax = np.min(image_out[0,:]),np.max(image_out[0,:])
    fig, ax1 = plt.subplots(5,5,figsize=(20,20))

    for i in range(25):
        if plot_diff:
            rf = np.reshape(image_out[i, :]-image_out[0,:], (31, 31))
            ax1[i % 5, i//5].imshow(rf, cmap=plt.cm.RdYlBu)
        else:
            rf = np.reshape(image_out[i, :], (31, 31))
            if scale_by_first:
                ax1[i % 5, i//5].imshow(rf, cmap=plt.cm.RdYlBu,vmin=vmin,vmax=vmax)
            else:
                ax1[i % 5, i//5].imshow(rf, cmap=plt.cm.RdYlBu)
        title = 'MEI - ' if i==0 else ''
        ax1[i % 5, i//5].set_title(f'{title}{100*(activations[i]/activations[0]):.2f}',fontsize=20)
    if save_path:
        print(f'Saving to {save_path}')
        plt.savefig(save_path)
    else:
        plt.show()

def generate_equivariance(noise_len,neuron,save_path,perc,name,model,train_set_len=1000000,epochs=5,is_aegan=True):
    net = NDN.load_model(model)
    _, input_size_x, input_size_y = net.input_sizes[0]
    net = find_MEI(net,neuron)
    mei_stimuli = get_filter(net,reshape=False)
    l2_norm = np.sum(mei_stimuli**2)
    max_activation = net.generate_prediction(mei_stimuli)[0,neuron]

    net = NDN.load_model(model)

    generator_net = GeneratorNet(net,input_noise_size=noise_len,is_aegan=is_aegan)
    generator_net.train_generator_on_neuron(
        neuron,
        data_len=train_set_len,
        l2_norm=l2_norm,
        max_activation=max_activation,
        perc=perc,
        epochs=epochs)
    image_out = generator_net.generate_stimulus(num_samples=10000)

    # Cluster images
    kmeans = AgglomerativeClustering(n_clusters=24,affinity='cosine',linkage='complete').fit(image_out)
    x=[]
    for i in range(24):
        x.append(image_out[kmeans.labels_==i][0,:])
    image_out[1:25,:]=x

    # Compute activations
    net = NDN.load_model(model)
    
    activations = net.generate_prediction(image_out)
    image_out[0,:] = mei_stimuli
    activations[0,neuron]= max_activation
    activations = activations[:,neuron]
    model_slug = model.split('/')[1][:10]
    # Plot receptive fields 
    np.save(
        os.path.join(save_path,'rfs',f'rf{neuron}-{noise_len}-{perc}.npy'),
        np.reshape(image_out,(-1,input_size_x,input_size_y))
        )
    if save_path is not None:
        save_path = os.path.join(save_path,f'{name}-neuron-{neuron}_p-{perc}_noiselen-{noise_len}_model-{model_slug}.png')
    plot_rfs(image_out,activations,save_path,scale_by_first=True)

    # Save MEI 
    



if __name__ == '__main__':
    fire.Fire(generate_equivariance)