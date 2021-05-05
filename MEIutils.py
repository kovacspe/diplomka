import copy
import os
import math

import fire
import numpy as np
import numpy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
import tensorflow as tf
from tqdm import tqdm


from NDN3.NDN import NDN
from NDN3.layer import VariableLayer
from NDN3 import NDNutils

from generator_net import GeneratorNet
from utils.data_loaders import AntolikDataLoader
from utils.experiment_params import experiment_args
from utils.misc_utils import evaluate_performance

def norm_samples(samples):
    axis = tuple(range(1,samples.ndim))
    samples = np.array(samples)
    return (samples - np.mean(samples,axis=axis,keepdims=True)) / np.std(samples,axis=axis,keepdims=True) 

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
    cost = -net.cost
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
            if i%10==0:
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
    #net.networks[var_layer_net].layers[var_layer_layer].set_regularization('l2', 0.01)

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

@experiment_args
def correlations(net,dataset,experiment='000'):
    train_x, train_y = dataset.train()
    test_x,test_y = dataset.val()
    train_predictions = net.generate_prediction(train_x)
    test_predictions = net.generate_prediction(test_x)
    print(np.std(train_predictions,axis=1)[84])
    print('----------')
    print(test_predictions[:84])
    train_correlations = [stats.pearsonr(train_predictions[:,i],train_y[:,i])[0] for i in range(np.shape(train_y)[1])]
    test_correlations = [stats.pearsonr(test_predictions[:,i],test_y[:,i])[0] for i in range(np.shape(test_y)[1])]
    np.save(f'output/07_correlations/{experiment}_train_correlations.npy',train_correlations)
    np.save(f'output/07_correlations/{experiment}_test_correlations.npy',test_correlations)


@experiment_args
def neuron_description(experiment='000'):
    sta_corrs = np.load(f'output/03_sta/{experiment}_sta_correlations.npy')
    #np.load(f'output/07_correlations/{experiment}_train_correlations.npy',train_correlations)
    net_corrs = np.load(f'output/07_correlations/{experiment}_test_correlations.npy')
    # Neuron correlation on validation
    corrs_df = pd.DataFrame(zip(sta_corrs,net_corrs),columns=['STA correlation', 'Model correlation'])
    corrs_df['score'] = (1-corrs_df['STA correlation'])*corrs_df['Model correlation']
    corrs_df.sort_values('score',ascending=False,inplace=True)
    with open(f'output/07_correlations/{experiment}_corr_table.tex','w') as f:
        f.write(corrs_df.to_latex())

@experiment_args
def compare_sta_mei(chosen_neurons=range(103),experiment='000'):
    stas = np.load(f'output/03_sta/{experiment}_sta.npy')[chosen_neurons]
    sta_activations = np.load(f'output/03_sta/{experiment}_sta_activations.npy')[chosen_neurons]
    stas_n = np.load(f'output/03_sta/{experiment}_sta_n.npy')[chosen_neurons]
    sta_activations_n = np.load(f'output/03_sta/{experiment}_sta_activations_n.npy')[chosen_neurons]
    meis = np.load(f'output/04_mei/{experiment}_mei.npy')[chosen_neurons]
    mei_activations = np.load(f'output/04_mei/{experiment}_mei_activations.npy')[chosen_neurons]
    meis_n = np.load(f'output/04_mei/{experiment}_mei_n.npy')[chosen_neurons]
    mei_activations_n = np.load(f'output/04_mei/{experiment}_mei_activations_n.npy')[chosen_neurons]

    #titles = [f'{i} - STA - {act:.2f}' for i,act in zip(chosen_neurons,sta_activations)]
    titles = [f'{i} - STAn - {act:.2f}' for i,act in zip(chosen_neurons,sta_activations_n)]
    #titles = titles + [f'{i} - MEI - {act:.2f}' for i,act in zip(chosen_neurons,mei_activations)]
    titles = titles + [f'{i} - MEIn - {act:.2f}' for i,act in zip(chosen_neurons,mei_activations_n)]
    plot_grid(np.concatenate([stas_n,meis_n]),titles,save_path=f'output/05_compare_mei_sta/{experiment}_comparison.png',show=False)
    acts_df = pd.DataFrame(zip(sta_activations,sta_activations_n,mei_activations,mei_activations_n),columns=['STA activation', 'STA norm activation', 'MEI activation', 'MEI norm activation'])
    acts_df['score'] = acts_df['MEI norm activation']-acts_df['STA norm activation']
    acts_df.sort_values('score',ascending=False,inplace=True)
    with open(f'output/07_correlations/{experiment}_act_table.tex','w') as f:
        f.write(acts_df.to_latex())

@experiment_args
def generate_equivariance(
        neuron:int,
        noise_len:int,
        perc:float,
        net:NDN,
        num_equivariance_clusters:int,
        eq_train_set_len=10000,
        eq_epochs=5,
        is_aegan=False,
        loss='oneside-gaussian',
        mask=False,
        gen_type='conv',
        experiment='000',
        norm='post'
    ):
    _, input_size_x, input_size_y = net.input_sizes[0]
    # Load precomputed MEI
    max_activation = np.load(f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]
    print(max_activation)
    if mask:
        mask = np.load(f'output/02_masks/{experiment}_hardmasks.npy')[neuron]
    else:
        mask = None
    train_log = f'output/tf/{experiment}-{neuron}'
    net2 = copy.deepcopy(net)
    generator_net = GeneratorNet(
        net,
        input_noise_size=noise_len,
        loss=loss,
        is_aegan=is_aegan,
        mask=mask,
        gen_type=gen_type,
        norm=norm
    )
    generator_net.train_generator_on_neuron(
        neuron,
        data_len=eq_train_set_len,
        max_activation=max_activation,
        perc=perc,
        epochs=eq_epochs,
        train_log=train_log
    )
    image_out = generator_net.generate_stimulus(num_samples=10000)

    # Cluster images
    try:
        kmeans = AgglomerativeClustering(n_clusters=num_equivariance_clusters,affinity='cosine',linkage='complete').fit(image_out)
        x=[]
        for i in range(num_equivariance_clusters):
            x.append(image_out[kmeans.labels_==i][0,:])
        x = np.vstack(x)
    except ValueError:
        x = image_out[:24,:]

    # Compute activations  
    activations = net2.generate_prediction(x)[:,neuron]
    
    # Plot receptive fields 
    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_equivariance.npy',
        np.reshape(x,(-1,input_size_x,input_size_y))
    )
    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_activations.npy',
        activations
    )
    generator = generator_net.extract_generator()
    print(generator.networks[-1].layers[-1].normalize_output)
    generator.save_model(f'output/08_generators/{experiment}_neuron{neuron}_generator.pkl')

def plot_grid(images,titles=None,num_cols=8,save_path=None,show=False,cmap=plt.cm.RdYlBu,highlight=None):
    num_images = len(images)
    if titles is None:
        titles=['']*num_images
    else:
        assert num_images==len(titles)
    num_rows = math.ceil(num_images/num_cols)
    fig, ax1 = plt.subplots(num_rows, num_cols,figsize=(3*num_cols,2.5*num_rows))
    mpl.rcParams['axes.linewidth'] = 20
    for i, (img,tit) in enumerate(zip(images,titles)):
        np.testing.assert_almost_equal(np.mean(img),0.0,decimal=2,err_msg='Mean is not 0')
        np.testing.assert_almost_equal(np.std(img),1.0,decimal=2,err_msg='Standard deviation is not 1')
        ax1[i // num_cols, i%num_cols].imshow(img,cmap=cmap)
        ax1[i // num_cols, i%num_cols].set_xticklabels([],[])
        ax1[i // num_cols, i%num_cols].set_yticklabels([],[])
        if highlight is not None and i in highlight:
            [sp.set_linewidth(3) for _,sp in ax1[i // num_cols, i%num_cols].spines.items()]
        if len(tit)>0:
            ax1[i // num_cols, i%num_cols].set_title(tit,fontsize=10)
    for i in range(num_images,num_cols*num_rows):
        fig.delaxes(ax1[i // num_cols, i%num_cols])

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()


def mask_stimuli(stimuli,experiment,neuron):
    neuron_mask = np.load(f'output/02_masks/{experiment}_hardmasks.npy')[neuron]
    neuron_mask = np.where(neuron_mask==0,np.nan,neuron_mask)
    neuron_mask = np.where(neuron_mask==1,0,neuron_mask)
    return stimuli + np.tile(neuron_mask,(np.shape(stimuli)[0],1,1))

@experiment_args
def generate_sta(dataset,net,experiment='000',chosen_neurons=None):
    x, y = dataset.train()
    x = np.matrix(x)
    y = np.matrix(y)
    test_x, test_y = dataset.val()

    sta = STA_LR(x, y, 10000).transpose()
    sta_normalized =  norm_samples(sta)

    activations = net.generate_prediction(sta)
    sta_activations = [activations[i,i] for i in range(len(activations))]

    activations_n = net.generate_prediction(sta_normalized)
    sta_activations_n = [activations_n[i,i] for i in range(len(activations_n))]

    sta_predictions = np.array(test_x*sta.T)
    sta_correlations = [stats.pearsonr(sta_predictions[:,i],test_y[:,i])[0] for i in range(np.shape(y)[1])]

    _, input_size_x, input_size_y = net.input_sizes[0]
    sta = np.array(sta).reshape((-1,input_size_x,input_size_y))
    sta_normalized = np.array(sta_normalized).reshape((-1,input_size_x,input_size_y))
    np.save(f'output/03_sta/{experiment}_sta.npy', sta)
    np.save(f'output/03_sta/{experiment}_sta_n.npy', sta_normalized)
    np.save(f'output/03_sta/{experiment}_sta_activations.npy', sta_activations)
    np.save(f'output/03_sta/{experiment}_sta_activations_n.npy', sta_activations_n)
    np.save(f'output/03_sta/{experiment}_sta_correlations.npy', sta_correlations)
    titles = [f'{i} - A:{act:.2f} C:{corr:.2f}' for i,(act,corr) in enumerate(zip(sta_activations_n,sta_correlations))]
    plot_grid(list(sta_normalized),titles,save_path=f'output/03_sta/{experiment}_sta.png',show=True,highlight=chosen_neurons)

@experiment_args
def generate_mei(net,dataset,experiment='000'):
    _,y = dataset.train()
    num_neurons = np.shape(y)[1]
    net2 = copy.deepcopy(net)
    meis = []
    meis_n = []
    activations = []
    activations_n = []
    for neuron in range(num_neurons):
        net = find_MEI(net,neuron, 80)
        mei = get_filter(net)
        mei_activation = net2.generate_prediction(np.reshape(mei,(1,-1)))[0,neuron]
        mei_normalized =  norm_samples(mei)
        mei_activation_normalized = net2.generate_prediction(np.reshape(mei_normalized,(1,-1)))[0,neuron]
        meis.append(mei)
        activations.append(mei_activation)
        meis_n.append(mei_normalized)
        activations_n.append(mei_activation_normalized)
    # meis = np.load(f'output/04_mei/{experiment}_mei.npy',)
    # activations = np.load(f'output/04_mei/{experiment}_mei_activations.npy')
    titles = [f'{i} - {act:.3f}' for i,act in enumerate(activations_n)]
    np.save(f'output/04_mei/{experiment}_mei.npy',meis)
    np.save(f'output/04_mei/{experiment}_mei_activations.npy',activations)
    np.save(f'output/04_mei/{experiment}_mei_n.npy',meis_n)
    np.save(f'output/04_mei/{experiment}_mei_activations_n.npy',activations_n)

    plot_grid(meis_n,titles,num_cols=8,save_path=f'output/04_mei/{experiment}_mei.png',show=True)

@experiment_args
def generate_masks(net,dataset,mask_threshold,num_images=50,experiment='000'):
    x, y = dataset.train()
    x = x[:num_images]
    def mask_pixel(pixel):
        return 1 if pixel>mask_threshold else 0
    # Compute masks
    mask = compute_mask(net,0,x,np.linspace(-1.6,1.6,10))
    hard_mask = np.vectorize(mask_pixel)(mask)
    # Plot masks
    plot_grid(mask,save_path=f'output/02_masks/{experiment}_masks_plot.png',cmap=plt.cm.hot)
    plot_grid(hard_mask,save_path=f'output/02_masks/{experiment}_hardmasks_plot.png',cmap=plt.cm.hot)
    # Save masks to npy
    np.save(f'output/02_masks/{experiment}_masks.npy',mask)
    np.save(f'output/02_masks/{experiment}_hardmasks.npy',hard_mask)

@experiment_args
def plot_interpolations(net,neuron,experiment='000',mask=False,num_interpolations=8,num_samples=8):
    inputs = []
    generator = NDN.load_model(f'output/08_generators/{experiment}_neuron{neuron}_generator.pkl')
    print(generator.networks[-1].layers[-1].normalize_output)
    noise_shape = generator.input_sizes[0][1]
    mei_act = np.load(f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]
    for i in range(num_interpolations):
        first_point = np.random.normal(0.0,1.0,noise_shape)
        first_point /=np.linalg.norm(first_point,axis=0)
        second_point = np.random.normal(0.0,1.0,noise_shape)
        #second_point /=np.linalg.norm(second_point,axis=0)
        print(np.linspace(first_point,-first_point,num_samples).shape)
        inputs.append(np.linspace(first_point,-first_point,num_samples))
    noise_samples = np.vstack(inputs)
    print(noise_samples.shape)
    invariances = generator.generate_prediction(noise_samples)
    mask_text = ''
    if mask:
        invariances = mask_stimuli(invariances,experiment,neuron)
        mask_text = '_masked'
    activations = net.generate_prediction(invariances)[:,neuron]
    print(activations)
    titles = [f'{act/mei_act:.2f}' for act in activations]
    invariances = np.reshape(invariances,(-1,31,31))
    plot_grid(invariances,titles,num_cols=8,save_path=f'output/06_invariances/{experiment}_interpolations_plot{mask_text}.png',show=True)

@experiment_args
def plot_equivariances(net,neuron,experiment='000',mask=False,include_mei=False):
    invariances = np.load(f'output/06_invariances/{experiment}_neuron{neuron}_equivariance.npy')
    activations = np.load(f'output/06_invariances/{experiment}_neuron{neuron}_activations.npy')
    mei_act = np.load(f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]
    print('Mean invariance images activations' +str(np.mean(activations[0])))
    print(f'MEI activation: {mei_act}')

    if include_mei:
        mei = np.load(f'output/04_mei/{experiment}_mei_n.npy')[neuron]
        invariances[-1] = mei
        activations[-1] = mei_act
    mask_text = ''
    invariances = np.reshape(invariances,(-1,31,31))

    if mask:
        invariances = mask_stimuli(invariances,experiment,neuron)
        mask_text = '_masked'
    
    titles = [f'{act/mei_act:.2f}' for act in activations]
    plot_grid(invariances,titles,num_cols=4,save_path=f'output/06_invariances/{experiment}_plot{mask_text}.png',show=True)

def basic_setup(experiment='000'):
    print(f'running experiment {experiment}')
    generate_sta(experiment=experiment)
    generate_mei(experiment=experiment)
    generate_masks(experiment=experiment)


if __name__ == '__main__':
    fire.Fire({
        'generate_equivariance': generate_equivariance,
        'analyze_mei': compare_sta_mei,
        'sta': generate_sta,
        'mei': generate_mei,
        'mask': generate_masks,
        'corr': correlations,
        'neuron_desc': neuron_description,
        'plot_equivariances': plot_equivariances,
        'plot_interpolations': plot_interpolations,
        'compare': compare_sta_mei,
        'basic': basic_setup
        }
    )
