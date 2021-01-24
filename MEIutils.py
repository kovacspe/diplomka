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


def generate_gan_net(input_noise_size, output_shape,l2_norm):
    out = output_shape[:]
    out[0] = 4
    out = [16,8,8]
    params = NDNutils.ffnetwork_params(
        input_dims=[1, input_noise_size],
        layer_sizes=[out,8,4,1],
        layer_types=['biasreg','deconv','deconv','deconv'],
        act_funcs=['relu','relu','relu','tanh'],
        conv_filter_widths=[None,5,5,5],
        shift_spacing=[None,2,2,1],
        reg_list={
            'l2':[0.1,0.0001,0,0]
        },
        verbose=False)
        
    params['xstim_n'] = [0]
    params['normalize_output'] =  [None,None,None,l2_norm]
    params['bias_reg_initalizer'] = [
            {'l2':2},
            {},
            {},
            {}
        ]
    params['output_shape'] = [None,None,[31,31],[31,31]]
    return params


def copy_net_params(original_NDN_net, target_NDN_net, net_num_original, net_num_target):
    for layer_source, layer_target in zip(original_NDN_net.networks[net_num_original].layers, target_NDN_net.networks[net_num_target].layers):
        layer_target.copy_layer_params(layer_source)

def extract_gan(gan_net,gan_subnet_id):
    if gan_net.network_list[gan_subnet_id]['xstim_n'] is None:
            raise AttributeError(f'Network {gan_subnet_id} is not connected to input')
    gan_only = NDN([gan_net.network_list[gan_subnet_id]],
    noise_dist='max',input_dim_list=[gan_net.network_list[gan_subnet_id]['input_dims']])

    # Copy weights
    copy_net_params(gan_net,gan_only,gan_subnet_id,0)

    return gan_only

def create_gan(original_net, input_noise_size, loss,l2_norm):
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
        gan_nets.append(generate_gan_net(input_noise_size, inp_size,l2_norm))
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
        copy_net_params(original_net, new_net, i-num_gan_nets, i)

    # Construct fit vars
    layers_to_skip = []
    for i, net in enumerate(new_net.networks):
        if i >= num_gan_nets:
            layers_to_skip.append([x for x in range(len(net.layers))])
        else:
            layers_to_skip.append([])

    gan_fit_vars = new_net.fit_variables(
        layers_to_skip=layers_to_skip, fit_biases=False)
    #gan_fit_vars[0][0]['biases']=True
    print('Fit wars',gan_fit_vars)
    return new_net, gan_fit_vars


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


def find_MEI_GAN(net,optimize_neuron,noise_len,data_len,l2_norm,max_activation=None):
    # Transform network to GAN
    noise = 'gaussian' if max_activation is not None else 'max'
    new_net, fit_vars = create_gan(net,noise_len,noise,l2_norm)

    # Get network input and output shapes
    input_shape = (data_len, noise_len)
    output_shape = (data_len, np.prod(new_net.output_sizes))

    # Setup data filter to filter only desired neuron
    tmp_filters = np.zeros((data_len, np.prod(new_net.output_sizes)))
    tmp_filters[:, optimize_neuron] = 1

    # Create input
    #noise_input = np.random.uniform(-0.5,0.5,size=input_shape)
    noise_input = np.random.normal(size=input_shape,scale=1)
    input_norm = np.sqrt(np.sum(noise_input**2,axis=1))/np.sqrt(noise_len)
    for i in range(5):
        print(input_norm[i],noise_input[i])
    #quit()

    output = np.zeros(output_shape)
    if max_activation is not None:
        output[:,optimize_neuron] = output[:,optimize_neuron]+(0.90*np.ones(output_shape[0]) *max_activation)#np.multiply((np.ones(output_shape[0]) *max_activation),(np.clip(1-input_norm,a_min=0,a_max=1)))
        
    print(output[:,optimize_neuron])
    #fit_vars[0][0]['biases']=True

    # Optimize
    new_net.train(noise_input, output, fit_variables=fit_vars,
              data_filters=tmp_filters, learning_alg='adam',
              train_indxs=np.arange(data_len*0.9),
              test_indxs=np.arange(data_len*0.9,data_len),
              opt_params={'display': 1,'batch_size': 256, 'use_gpu': False, 'epochs_summary': 1, 'epochs_training': 2, 'learning_rate': 0.001},
              output_dir=f'gan_output/gan10')

    mean_pred = new_net.generate_prediction(noise_input[:10,:])
    # Extract GAN 
    extracted_gan = extract_gan(new_net,0)
    return extracted_gan

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
    net.train(dummy_input, dummy_output, fit_variables=fit_vars,
              data_filters=tmp_filters, learning_alg='lbfgs')

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


# Load net
model_name = '.\\models\\basicFC-basicFC-hidden0.2-reg_h0.1-reg_l0.1-exp_namebasicFC.pkl'
NEURON = 28
NOISE_LEN=100
#model_name = '.\\models\\conv3-conv-c_size3-channels9-cd2x0.1-hidden_tmax-hidden_s1-hidden_ltnormal-exp_nameconv3.pkl'
net = NDN.load_model(model_name)
compare_sta_mei(net)
quit()
net = find_MEI(net,NEURON)
mei_stimuli = get_filter(net,reshape=False)
l2_norm = np.sum(mei_stimuli**2)
max_activation = net.generate_prediction(mei_stimuli)[0,NEURON]
print('Max activation: ',max_activation)
print('L2-norm', l2_norm)

net = NDN.load_model(model_name)
gan = find_MEI_GAN(net,NEURON,noise_len=NOISE_LEN,data_len=1000000,l2_norm=l2_norm,max_activation=max_activation)
gan.save_model('./gan2.model')
xy = np.mgrid[-1:1:0.2, -1:1:0.2].reshape(2,-1).T
vecs = np.random.normal(size=(100,NOISE_LEN),scale=1)
normalized  = 0.01*np.apply_along_axis(lambda x: x / np.linalg.norm(x),axis=1,arr=vecs)

noise_input = vecs

#noise_input = np.random.normal(size=(100,NOISE_LEN),scale=0.1)
noise_input[1,:] = np.zeros((1,NOISE_LEN))
image_out = gan.generate_prediction(noise_input)

net = NDN.load_model(model_name)
activations = net.generate_prediction(image_out)
image_out[0,:] = mei_stimuli
activations[0,NEURON]= max_activation

vmin,vmax = np.min(mei_stimuli),np.max(mei_stimuli)
fig, ax1 = plt.subplots(10,10)
for i in range(np.shape(image_out)[0]):
    rf = np.reshape(image_out[i, :], (31, 31))
    diff_rf = np.reshape(image_out[i, :]-mei_stimuli, (31, 31))
    ax1[i % 10, i//10].imshow(rf, cmap=plt.cm.RdYlBu,vmin=vmin,vmax=vmax)
    if i==0:
        title='MEI - '
    elif i==1:
        title='Zero noise -'
    else: 
        title = ''
    ax1[i % 10, i//10].set_title(f'{title}{100*(activations[i,NEURON]/max_activation):.2f}',fontsize=8)
plt.show()

