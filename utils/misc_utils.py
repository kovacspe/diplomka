import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering

filter_dict = {
    'gauss5x5': np.float32([
        [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
        [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
        [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
        [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
        [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]),
    'gauss3x3': np.float32([
        [1 / 16, 1 / 8, 1 / 16],
        [1 / 8, 1 / 4, 1 / 8],
        [1 / 16, 1 / 8, 1 / 16]]
    ),
    'laplace5x5': np.outer(np.float32([1, 4, 6, 4, 1]), np.float32([1, 4, 6, 4, 1])) / 256,

}


def create_filter(filter_name, in_channels, out_channels):
    """
    Get convolutional filters
    """
    if filter_name not in filter_dict:
        raise AttributeError(f'filter {filter_name} doesnt exist')
    conv_filter = filter_dict[filter_name]
    conv_filter = np.tile(conv_filter, (in_channels, out_channels, 1, 1))
    conv_filter = np.transpose(conv_filter, (3, 2, 0, 1))
    return conv_filter


def reshape_weights_to_NDN(inp):
    return np.reshape(inp, [np.prod(inp.shape[:])])


def reshape_input_to_NDN(inp):
    return np.reshape(inp, [-1, np.prod(inp.shape[1:])])


def evaluate_performance(pred, gold):
    """
    Computes mean correaltion
    """
    rho = np.zeros(pred.shape[0])
    for i, (res, pred) in enumerate(zip(gold, pred)):
        if True:  # np.std(res) > 1e-5 and np.std(pred) > 1e-5:
            rho[i] = stats.pearsonr(res, pred)[0]
    return rho.mean()


def merge_train_and_val_set(train_x, train_y, val_x, val_y):
    # Reshape to NDN
    train_x = reshape_input_to_NDN(train_x)
    train_y = reshape_input_to_NDN(train_y)
    val_x = reshape_input_to_NDN(val_x)
    val_y = reshape_input_to_NDN(val_y)

    data_x = np.concatenate((train_x, val_x), axis=0)
    data_y = np.concatenate((train_y, val_y), axis=0)
    train_len = train_x.shape[0]
    val_len = val_x.shape[0]
    train_idxs = np.arange(train_len)
    val_idxs = np.arange(train_len, train_len+val_len)
    return data_x, data_y, train_idxs, val_idxs


def norm_samples(samples):
    """
    Normalize samples to zero mean and unit std
    """
    axis = tuple(range(1, samples.ndim))
    samples = np.array(samples)
    return (samples - np.mean(samples, axis=axis, keepdims=True)) / np.std(samples, axis=axis, keepdims=True)


def STA_LR(training_inputs, training_set, laplace_bias):
    """
    This function implements the spike triggered averaging with laplacian regularization.
    It takes the training inputs and responses, and returns the estimated kernels.
    """
    kernel_size = np.shape(training_inputs)[1]
    laplace = laplaceBias(int(np.sqrt(kernel_size)),
                          int(np.sqrt(kernel_size)))
    return np.linalg.pinv(training_inputs.T*training_inputs + laplace_bias*laplace) * training_inputs.T * training_set


def laplaceBias(sizex: int, sizey: int):
    """
    Creates laplacian matrix with size `sizex` x `sizey`
    """
    S = np.zeros((sizex*sizey, sizex*sizey))
    for x in range(0, sizex):
        for y in range(0, sizey):
            norm = np.mat(np.zeros((sizex, sizey)))
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
    S = np.mat(S)
    return S*S.T


def sample_sphere(num_samples, ndims):
    """
    Generates `num_samples` points uniformly from `ndims`-dimensional unit sphere
    """
    vec = np.random.randn(ndims, num_samples)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def choose_representant(num_representative_samples, net, neuron, stimuli, activation_lowerbound=-np.inf, activation_upperbound=np.inf):
    """
     Optimization of weights in variable layer to maximize neurons output
        Parameters:
            num_representative_samples (int): Number of stimuli to chose
            net (NDN): A trained neural network with variable layer
            neuron (int): neuron id
            stimuli (np.array): Array of generated stimuli from which to choose
            data_filters (np.array): data_filters will be passed to NDN.train
            activation_lowerbound (float): 
            activation_upperbound (float): 
        Returns:
            chosen_stimuli (np.array): 
    """
    # Filter stimuli
    activations = net.generate_prediction(stimuli)[:, neuron]
    stimuli = stimuli[
        (activations <= activation_upperbound) & (
            activations >= activation_lowerbound)
    ]
    activations = activations[(activations <= activation_upperbound) & (
        activations >= activation_lowerbound)]

    # Cluster stimuli
    try:
        kmeans = AgglomerativeClustering(
            n_clusters=num_representative_samples, affinity='cosine', linkage='complete').fit(stimuli)
        images = []
        acts = []
        for i in range(num_representative_samples):
            images.append(stimuli[kmeans.labels_ == i][0, :])
            acts.append(activations[kmeans.labels_ == i][0])
        images = np.vstack(images)
        acts = np.array(acts)
    except ValueError:
        # If clustering failed choose first `num_representative_samples` samples
        print(
            f'Clustering failed ... taking first {num_representative_samples} images')
        images = stimuli[:num_representative_samples, :]
        acts = activations[:num_representative_samples]

    return images, acts

def mask_stimuli(stimuli, experiment, neuron):
    """
    Apply mask on stimuli
        Parameters:
            stimuli (np.array):
            experiment (str): experiment ID
            neuron (int): neuron ID
        Returns:
            masked_stimuli
    """
    neuron_mask = np.load(
        f'output/02_masks/{experiment}_hardmasks.npy')[neuron]
    neuron_mask = np.where(neuron_mask == 0, np.nan, neuron_mask)
    neuron_mask = np.where(neuron_mask == 1, 0, neuron_mask)
    return stimuli + np.tile(neuron_mask, (np.shape(stimuli)[0], 1, 1))