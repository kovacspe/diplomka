import numpy as np
from scipy import stats

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
    rho = np.zeros(pred.shape[0])
    for i, (res, pred) in enumerate(zip(gold, pred)):
        if np.std(res) > 1e-5 and np.std(pred) > 1e-5:
            rho[i] = stats.pearsonr(res, pred)[0]
    return rho.mean()

def load_cnn_sys_weight(network, np_file):
    with open(np_file, 'rb') as f:
        # Core
        di = np.load(f,allow_pickle=True)
        for i in range(3):
            w = di[f'core{i}w']
            b = di[f'core{i}b']
            network.networks[-1].layers[i].weights = w
            network.networks[-1].layers[i].biases = b
        # Readout
        w_sp = di['masks']
        w_f = di['features']
        b = di['readout_biases']
        network.networks[-1].layers[-1].weights = np.concatenate(
            [reshape_weights_to_NDN(w_sp), reshape_weights_to_NDN(w_f)], axis=0)
        network.networks[-1].layers[-1].biases = b

def merge_train_and_val_set(train_x,train_y,val_x,val_y):
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
    val_idxs = np.arange(train_len,train_len+val_len)
    return data_x,data_y,train_idxs,val_idxs