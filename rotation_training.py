from load_rotation_data import get_data, Dataset
from define_rotation_net import define_spatial_feature_readout_network
from NDN3.NDNutils import ffnetwork_params
from NDN3.NDN import NDN
from datetime import datetime
from scipy import stats
import tensorflow as tf
import numpy as np
import argparse

print('Start rotation training script')
print('tf imported with version ', tf.__version__)
print('importing NDN...')
# Import NDN
print('NDN imported')
# Import network architecture
print('MEI net imported')
# Import data loader
print('data imported')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=0.002,
                    help='Learning rate')
parser.add_argument('--epochs', type=int, default=2000,
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


def reshape_to_NDN(inp):
    return np.reshape(inp, [-1, np.prod(inp.shape[1:])])


# Tensorboard logdir setting
name = f'{args.name}-bs{args.batch_size}-lr{args.lr}-chan{args.channels}-ep{args.epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

logdir = "output/" + name
GPU = args.gpu

# Load data
# Train set
data = get_data('data.pkl')
train_x, train_y = data.train()
B_train, H, W, C = train_x.shape
B_train, Neurons = train_y.shape
print(f'Train set: image shape: {train_x.shape}, output shape {train_y.shape}')

# Validation set
val_x, val_y = data.val()
B_val = val_x.shape[0]
print(f'Validation set size: {B_val}')

# Merge train and validation set
train_x = np.concatenate((train_x, val_x), axis=0)
train_y = np.concatenate((train_y, val_y), axis=0)

train_x = reshape_to_NDN(train_x)
train_y = reshape_to_NDN(train_y)

val_x = reshape_to_NDN(val_x)
val_y = reshape_to_NDN(val_y)
# Count trian responses means
means = np.mean(train_y, axis=0)/2


# Test set
test_x, test_y = data.test()
print(
    f'Train+val set: image shape: {train_x.shape}, output shape {train_y.shape}')
print(f'Test set: image shape: {test_x.shape}, output shape {test_y.shape}')
test_x = reshape_to_NDN(test_x)
test_y = reshape_to_NDN(test_y)


# Define network
net = define_spatial_feature_readout_network(
    args.batch_size, W, H, C, Neurons, args.channels, means)

opt_params = net.optimizer_defaults({}, 'adam')
opt_params['batch_size'] = args.batch_size
opt_params['learning_rate'] = args.lr
opt_params['epochs_summary'] = 25
opt_params['display'] = 1
opt_params['use_gpu'] = GPU
opt_params['epochs_training'] = args.epochs
#opt_params['early_stop_mode'] = 3
#opt_params['early_stop'] = 20
#opt_params['MAPest'] = True
#opt_params['epochs_ckpt'] = 20
#opt_params['run_diagnostics'] = True


def evaluate_performance(pred, gold):
    rho = np.zeros(pred.shape[0])
    for i, (res, pred) in enumerate(zip(gold, pred)):
        if np.std(res) > 1e-5 and np.std(pred) > 1e-5:
            rho[i] = stats.pearsonr(res, pred)[0]
    return rho.mean()


# Train network
b_v = min(B_val, args.batch_size)
validation_to_train = B_val - b_v
if args.loadmodel is not None:
    net = NDN.load_model(args.loadmodel)
    print(args.loadmodel, ' loaded')
    pred = net.generate_prediction(test_x)
    corr = evaluate_performance(pred, test_y)
    print('correlation', corr)

net.log_correlation = 'filter-low-std-gold'
net.train(
    train_x,
    train_y,
    train_indxs=np.arange(B_train),
    test_indxs=np.arange(B_train,
                         B_train+B_val),
    learning_alg='adam',
    opt_params=opt_params,
    output_dir=logdir
)

net.save_model(f'./{name}.pkl')

# Evaluation


#evaluation = net.eval_models(test_x,test_y)
#rint('MSE: ',np.mean(evaluation))

# Correlation
pred = net.generate_prediction(train_x)
corr = evaluate_performance(pred, train_y)
print('Train correlation :', corr)

pred = net.generate_prediction(val_x)
corr = evaluate_performance(pred, val_y)
print('Train correlation :', corr)

pred = net.generate_prediction(test_x)
corr = evaluate_performance(pred, test_y)
print('Test correlation :', corr)
