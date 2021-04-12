import numpy as np
import tensorflow as tf
from NDN3.layer import VariableLayer
import matplotlib.pyplot as plt
import numpy.linalg
from utils.misc_utils import evaluate_performance
from scipy import stats
from NDN3.NDN import NDN
from utils.data_loaders import AntolikDataLoader
from NDN3 import NDNutils
from sklearn.cluster import KMeans,AgglomerativeClustering
import os
from sklearn.metrics.pairwise import pairwise_distances
from generator_net import GeneratorNet
from utils.experiment_params import experiment_args
import fire
from tqdm import tqdm
import math

def compute_mask(net: NDN, neuron: int, images: np.array, try_pixels=range(0, 2, 1)):
    num_images, num_pixels = np.shape(images)
    activations = net.generate_prediction(images)
    mask = np.zeros(num_pixels)
    net.batch_size = 512

    inputs = []
    for pixel_position in tqdm(range(num_pixels)):
        for i, pixel in enumerate(try_pixels):
            modified_images = np.copy(images)
            modified_images[:, pixel_position]+= np.repeat(pixel, num_images)
            inputs.append(modified_images)
    modified_images = np.vstack(inputs)
    predicted_activations = net.generate_prediction(
                    modified_images
                    )

    differences = predicted_activations - np.tile(activations,(len(try_pixels)*num_pixels,1))
    differences = np.reshape(differences,(31,31,-1,103))
    mask = np.std(differences,axis=2).transpose((2,0,1))
    return mask

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

def fourier_filter(sizex,sizey,alpha):
    fy = np.fft.fftfreq(sizey)[:, None]
    fx = np.fft.fftfreq(sizex)[:]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    freqs = np.sqrt(fx * fx + fy * fy)
    scale = 1.0 / np.maximum(freqs, 1.0 / max(sizex, sizey)) ** -alpha
    scale *= np.sqrt(sizex * sizey)
    return scale

def define_mei_train_step(net,var_layer,sizex,sizey,alpha):
    learning_rate = tf.placeholder(tf.float32)
    cost = -net.cost_penalized
    d_weights = tf.reduce_mean(
        tf.gradients(cost,[var_layer.weights_var]),
        axis=0)
    d_in_fourier_domain = tf.signal.fft2d(
        tf.cast(
            tf.reshape(d_weights,(sizex,sizey)),
            dtype=tf.complex64
        )
    )

    d_in_fourier_domain_filtered = tf.multiply(
        d_in_fourier_domain,
        fourier_filter(sizex,sizey,alpha)
        )
    d_weights_filtered = tf.cast(
        tf.reshape(
            tf.ifft2d(d_in_fourier_domain_filtered),
            (1,-1)
        ),
        dtype=tf.float32
    )
    
    train_step = var_layer.weights_var.assign(
        var_layer.weights_var + learning_rate*d_weights_filtered
    ) 
    return train_step,learning_rate

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

def train_MEI(net:NDN,input_data, output_data,data_filters,opt_args,fit_vars,var_layer):
    opt_params = net.optimizer_defaults(opt_args['opt_params'], opt_args['learning_alg'])
    epochs = opt_args['opt_params']['epochs_training']
    lr = opt_args['opt_params']['learning_rate']
    _,sizex,sizey = net.input_sizes[0]
    input_data, output_data, data_filters = net._data_format(input_data, output_data, data_filters)
    net._build_graph(
        opt_args['learning_alg'],
        opt_params,
        fit_vars,
        batch_size=1
        )

    with tf.Session(graph=net.graph, config=net.sess_config) as sess:
        # Define train step
        train_step,learning_rate = define_mei_train_step(net,var_layer,sizex,sizey,0.1)

        #Training
        net._restore_params(sess, input_data, output_data, data_filters)
        i=0
        cost = float('inf')
        best_cost = float('inf')
        without_increase=0

        while without_increase<100 and i<epochs: 
            sess.run(train_step,feed_dict={net.indices: [0],learning_rate: lr})
            cost = sess.run(net.cost, feed_dict={net.indices: [0]})
            cost_reg = sess.run(net.cost_reg, feed_dict={net.indices: [0]})
            if i%20==0:
                print(f'Cost: {cost:.5f}+{cost_reg:.5f}, best: {best_cost:.3f}')
            sess.run(
                var_layer.gaussian_blur_std_var.assign(
                    var_layer.gaussian_blur_std_var*0.99
                )
            )
            if cost<best_cost:
                best_cost=cost+cost_reg
                without_increase=0
            
            sess.run(var_layer.gaussian_blur)
            i+=1
            without_increase+=1
        net._write_model_params(sess)
    
    print(f'Trained with {i} epochs')

def find_MEI(net, optimize_neuron,epochs=400):
    # Find Variable layer
    var_layer_net, var_layer_layer = find_var_layer(net)

    # Activate variable layer
    net.network_list[var_layer_net]['as_var'] = True
    net.networks[var_layer_net].layers[var_layer_layer].set_regularization('l2', 1.0)

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
        'train_indxs':np.arange(1),
        'test_indxs':np.arange(1),
        'opt_params': {

                'display': 1000,
                'batch_size': 1, 
                'use_gpu': False, 
                'epochs_training': epochs, 
                'learning_rate': 0.0001
            }
    }
    # Optimize input (Variable layer)
    train_MEI(net,dummy_input,dummy_output,tmp_filters,opt_args,fit_vars,net.networks[var_layer_net].layers[var_layer_layer])

    # Restore original settings
    net.network_list[var_layer_net]['as_var'] = False
    net.noise_dist = original_noise_dist
    return net

def get_filter(net,reshape=True):
    var_layer_net, var_layer_layer = find_var_layer(net)
    _, x, y = net.input_sizes[0]
    raw_filter = net.networks[var_layer_net].layers[var_layer_layer].weights
    if reshape:
        return np.reshape(raw_filter, (x, y))
    else:
        return raw_filter




class NeuralResult:
    def __init__(self, i, sta, sta_activation, corr_sta, mei, mei_activation, max_activation=None, neuron_correlation=None):
        self.i = i
        self.mei_activation = mei_activation
        self.corr_sta = corr_sta
        self.sta_activation = sta_activation
        self.sta = sta
        self.mei = mei
        self.max_activation = max_activation
        self.neuron_correlation = neuron_correlation

    def plot(self, ax1, x, y):
        ax1[x, y].imshow(self.sta, cmap=plt.cm.RdYlBu)
        ax1[x, y].title.set_text(f'{self.i} {self.corr_sta:.2f}')
        ax1[x, y+1].imshow(self.mei, cmap=plt.cm.RdYlBu)
        ax1[x, y+1].title.set_text(f'{self.i} {self.neuron_correlation:.2f}')

    def __str__(self):
        return f'Neuron {self.i:2}: STA: {self.corr_sta:.2f} NN: {self.neuron_correlation:.2f} STA_ACT: {self.sta_activation:.2f} MEI_ACT: {self.mei_activation:.2f} MAX_ACT: {self.max_activation:.2f}'


def compare_sta_mei(net_path,output_file=None,epochs=None):
    # Load data
    net = NDN.load_model(net_path)
    loader = AntolikDataLoader('data', 1)
    x, y = loader.train()
    test_x, test_y = loader.val()
    x = np.matrix(x)
    y = np.matrix(y)

    # Get unit correlations
    net.batch_size = 50
    pred = net.generate_prediction(test_x)

    unit_corr = np.zeros(test_y.shape[1])
    max_activation = np.zeros(test_y.shape[1])
    for i in range(test_y.shape[1]):
        unit_corr[i] = stats.pearsonr(test_y[:, i], pred[:, i])[0]
        max_activation[i] = np.max(pred[:,i])

    # STA
    sta = STA_LR(x, y, 10000)
    test_x = np.matrix(test_x)

    results = []
    sta_corrs = []
    # Process each neuron
    for i, neuron in enumerate(range(np.shape(sta)[1])):
        # Reshape STA RF and count correlation
        if i>10: 
            break
        pred = np.array(test_x*sta[:, i])
        sta_corr = stats.pearsonr(pred[:, 0], test_y[:, i])[0]
        sta_corrs.append(sta_corr)
        sta_resh = np.reshape(sta[:, i], (31, 31))
        # Find MEI
        net = find_MEI(net, 0,i*50)
        mei = get_filter(net)
        net = NDN.load_model(net_path)
        mei_activation = net.generate_prediction(np.reshape(mei,(1,-1)))[0,i]
        sta_activation = net.generate_prediction(np.reshape(sta[:,i],(1,-1)))[0,i]
        # Create result
        res = NeuralResult(i, sta_resh, sta_activation, sta_corr, mei, mei_activation, max_activation[i], unit_corr[i])
        print(f'MEI L2: {np.linalg.norm(mei)}')
        print(f'STA L2: {np.linalg.norm(sta[:,i])}')
        print(f'IMG L2: {np.linalg.norm(test_x[0])}')
        print()
        results.append(res)
    print('net correlation: ', np.mean(unit_corr))
    print('sta correlation: ', np.mean(sta_corrs))
    if output_file is not None:
        with open(output_file+'.log','a') as out_file:
            for r in results:
                out_file.write(str(r)+'\n')

    results.sort(key=lambda r: r.neuron_correlation-r.corr_sta, reverse=True)
    for r in results:
        print(r)
    
    # Plot setup
    n_rows = 7  # int(np.ceil(np.sqrt(np.shape(sta)[1])))
    fig, ax1 = plt.subplots(n_rows, 2*8,figsize=(40,25))

    for i, res in enumerate(results[:56]):
        res.plot(ax1, i % n_rows, 2*(i//n_rows))
    plt.savefig(f'{output_file}_pict_1.png')
    fig, ax1 = plt.subplots(n_rows, 2*n_rows,figsize=(40,25))

    for i, res in enumerate(results[56:]):
        res.plot(ax1, i % n_rows, 2*(i//n_rows))
    plt.savefig(f'{output_file}_pict_2.png')


@experiment_args
def generate_equivariance(noise_len,neuron,perc,net,num_equivariance_clusters,train_set_len=10000,epochs=5,is_aegan=True,experiment='000'):
    _, input_size_x, input_size_y = net.input_sizes[0]
    # Load precomputed MEI
    mei_stimuli = np.load(f'output/04_mei/{experiment}_mei.npy')[neuron]
    max_activation = np.load(f'output/04_mei/{experiment}_mei_activations.npy')[neuron]
    l2_norm = np.sum(mei_stimuli**2)

    generator_net = GeneratorNet(net,input_noise_size=noise_len,is_aegan=is_aegan)
    generator_net.train_generator_on_neuron(
        neuron,
        data_len=train_set_len,
        l2_norm=l2_norm,
        max_activation=max_activation,
        perc=perc,
        epochs=epochs)
    image_out = generator_net.generate_stimulus(num_samples=1000)

    # Cluster images
    kmeans = AgglomerativeClustering(n_clusters=num_equivariance_clusters,affinity='cosine',linkage='complete').fit(image_out)
    x=[]
    for i in range(num_equivariance_clusters):
        x.append(image_out[kmeans.labels_==i][0,:])
    x = np.vstack(x)

    # Compute activations  
    activations = net.generate_prediction(x)[:,neuron]
    # Plot receptive fields 
    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_equivariance.npy',
        np.reshape(x,(-1,input_size_x,input_size_y))
    )
    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_activations.npy',
        np.reshape(activations,(-1,input_size_x,input_size_y))
    )

def plot_grid(images,titles=None,num_cols=8,save_path=None,show=False,cmap=plt.cm.RdYlBu):
    num_images = len(images)
    if titles is None:
        titles=['']*num_images
    else:
        assert num_images==len(titles)
    num_rows = math.ceil(num_images/num_cols)
    fig, ax1 = plt.subplots(num_rows, num_cols,figsize=(3*num_cols,2.5*num_rows))

    for i, (img,tit) in enumerate(zip(images,titles)):
        ax1[i // num_cols, i%num_cols].imshow(img,cmap=cmap)
        ax1[i // num_cols, i%num_cols].set_xticklabels([],[])
        ax1[i // num_cols, i%num_cols].set_yticklabels([],[])
        if len(tit)>0:
            ax1[i // num_cols, i%num_cols].set_title(tit,fontsize=20)
    for i in range(num_images,num_cols*num_rows):
        fig.delaxes(ax1[i // num_cols, i%num_cols])

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        fig.show()


@experiment_args
def generate_sta(dataset,experiment='000'):
    x, y = dataset.train()
    x = np.matrix(x)
    y = np.matrix(y)
    sta = STA_LR(x, y, 10000).transpose()
    sta = np.array(sta).reshape((-1,31,31))
    np.save(f'output/03_sta/{experiment}_sta.npy', sta)
    titles = [str(x) for x in range(len(sta))]
    plot_grid(list(sta),titles,save_path=f'output/03_sta/{experiment}_sta.png',show=True)

@experiment_args
def generate_mei(net,dataset,experiment='000'):
    _,y = dataset.train()
    num_neurons = np.shape(y)[1]
    meis = []
    activations = []
    for i, neuron in enumerate(range(num_neurons)):
        net = find_MEI(net, i, 400)
        mei = get_filter(net)
        mei_activation = net.generate_prediction(np.reshape(mei,(1,-1)))[0,i]
        meis.append(mei)
        activations.append(mei_activation)
    titles = [f'{i} - {act:.3f}' for i,act in enumerate(activations)]
    np.save(f'output/04_mei/{experiment}_mei.npy',meis)
    np.save(f'output/04_mei/{experiment}_mei_activations.npy',activations)
    plot_grid(meis,titles,num_cols=2,save_path=f'output/04_mei/{experiment}_mei.png')

@experiment_args
def generate_masks(net,dataset,mask_threshold,num_images=50,experiment='000'):
    x, y = dataset.train()
    x = x[:num_images]
    def mask_pixel(pixel):
        return 1 if pixel>mask_threshold else 0
    # Compute masks
    mask = compute_mask(net,0,x,np.linspace(-1.6,1.6,10))[:16]
    hard_mask = np.vectorize(mask_pixel)(mask)[:16]
    # Plot masks
    plot_grid(mask,save_path=f'output/02_masks/{experiment}_masks_plot.png',cmap=plt.cm.hot)
    plot_grid(hard_mask,save_path=f'output/02_masks/{experiment}_hardmasks_plot.png',cmap=plt.cm.hot)
    # Save masks to npy
    np.save(f'output/02_masks/{experiment}_masks.npy',mask)
    np.save(f'output/02_masks/{experiment}_hardmasks.npy',hard_mask)

@experiment_args
def plot_equivariances(neuron,experiment='000',mask=False,include_mei=False):
    invariances = np.load(f'output/06_invariances/{experiment}_neuron{neuron}_equivariance.npy')
    #TODO: Reshape when saving
    mask_text = ''
    invariances = np.reshape(np.tile(invariances,(16,1)),(-1,31,31))
    if mask:
        neuron_mask = np.load(f'output/02_masks/{experiment}_hardmasks.npy')[neuron]
        neuron_mask = np.where(neuron_mask==0,np.nan,neuron_mask)
        neuron_mask = np.where(neuron_mask==1,0,neuron_mask)
        invariances = invariances + np.tile(neuron_mask,(np.shape(invariances)[0],1,1))
        mask_text = '_masked'
    if include_mei:
        pass
    plot_grid(invariances,save_path=f'output/06_invariances/{experiment}_plot{mask_text}.png')


if __name__ == '__main__':
    fire.Fire({
        'generate_equivariance': generate_equivariance,
        'analyze_mei': compare_sta_mei,
        'sta': generate_sta,
        'mei': generate_mei,
        'mask': generate_masks,
        'plot_equivariances': plot_equivariances
        }
    )
