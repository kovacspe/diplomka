from load_rotation_data import get_data, Dataset
from define_rotation_net import define_spatial_feature_readout_network, get_opt_params
from NDN3.NDNutils import ffnetwork_params
from NDN3.NDN import NDN
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
from utils import reshape_input_to_NDN,merge_train_and_val_set, evaluate_performance


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=0.002,
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs')
parser.add_argument('--channels', type=int, default=16,
                    help='Number of channels ic convolutional core')
parser.add_argument('--loadmodel', type=str, default=None,
                    help='Load model weights')
parser.add_argument('--name', type=str, default=None,
                    help='Experiment name')
parser.add_argument('--gpu', action='store_true',
                    help='Run on gpu')
args = parser.parse_args()


# Tensorboard logdir setting
name = f'{args.name}-bs{args.batch_size}-lr{args.lr}-chan{args.channels}-ep{args.epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'


# Load data
# Train set
data = get_data('data.pkl')
inp, out = data.train()
_, H, W, C = inp.shape
_, Neurons = out.shape
train_x, train_y = data.train(NDN_reshape=True)
val_x, val_y = data.val(NDN_reshape=True)

data_x,data_y,train_idxs,test_idxs = merge_train_and_val_set(train_x,train_y,val_x,val_y)
# Count trian responses means
means = np.mean(train_y, axis=0)/2


# Test set
test_x, test_y = data.test(NDN_reshape=True)



# Define network
net = define_spatial_feature_readout_network(
    args.batch_size, W, H, C, Neurons, args.channels, means)
opt_params = get_opt_params()
opt_params['batch_size'] = args.batch_size
opt_params['learning_rate'] = args.lr
opt_params['epochs_summary'] = 25
opt_params['display'] = 1
opt_params['use_gpu'] = args.gpu
opt_params['epochs_training'] = args.epochs


# Train network
if args.loadmodel is not None:
    net = NDN.load_model(args.loadmodel)
    print(args.loadmodel, ' loaded')
pred = net.generate_prediction(test_x)
corr = evaluate_performance(pred, test_y)
print('correlation', corr)
net.log_correlation = 'filter-low-std-gold'
net.train(
    data_x,
    data_y,
    train_indxs=train_idxs,
    test_indxs=test_idxs,
    learning_alg='adam',
    opt_params=opt_params,
    output_dir=f'output/{name}'
)
pred = net.generate_prediction(train_x)
corr = evaluate_performance(pred, train_y)
print('Train correlation :', corr)

pred = net.generate_prediction(val_x)
corr = evaluate_performance(pred, val_y)
print('Train correlation :', corr)

pred = net.generate_prediction(test_x)
corr = evaluate_performance(pred, test_y)
print('Test correlation :', corr)
print('LR decay')
opt_params['learning_rate'] = opt_params['learning_rate']/10
name = f'2{args.name}-bs{args.batch_size}-lr{args.lr}-chan{args.channels}-ep{args.epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
net.train(
    data_x,
    data_y,
    train_indxs=train_idxs,
    test_indxs=test_idxs,
    learning_alg='adam',
    opt_params=opt_params,
    output_dir=f'output/{name}'
)

net.save_model(f'./{name}.pkl')

