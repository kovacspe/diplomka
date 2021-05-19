import copy

import fire
import numpy as np

import pandas as pd
from scipy import stats


from NDN3.NDN import NDN

from generator_net import GeneratorNet
from utils.data_loaders import AntolikDataLoader
from utils.experiment_params import experiment_args
from utils.misc_utils import (evaluate_performance,
                              choose_representant,
                              STA_LR,
                              norm_samples,
                              mask_stimuli
                              )
from utils.mask import generate_masks
from utils.plotting import (
    plot_grid, plot_invariances,
    plot_all_from_generator,
    plot_from_generator,
    plot_interpolations,
    plot_invariance_summary,
    plot_sphere_samples,
    compare_generators
)
from utils.mei import generate_mei


@experiment_args
def correlations(net, dataset, experiment='000'):
    """
    Compute correlation of each neuron and summary correlation both on test and train set.
        Parameters:
            net (NDN): A trained neural network with variable layer
            dataset (DataLoader): dataset for evaluation
    """
    train_x, train_y = dataset.train()
    test_x, test_y = dataset.val()
    train_predictions = net.generate_prediction(train_x)
    test_predictions = net.generate_prediction(test_x)
    train_correlations = [stats.pearsonr(train_predictions[:, i], train_y[:, i])[
        0] for i in range(np.shape(train_y)[1])]
    test_correlations = [stats.pearsonr(test_predictions[:, i], test_y[:, i])[
        0] for i in range(np.shape(test_y)[1])]
    print(f'Train prediciton {np.mean(train_correlations)}')
    print(f'Test prediciton {np.mean(test_correlations)}')
    np.save(
        f'output/07_correlations/{experiment}_train_correlations.npy', train_correlations)
    np.save(
        f'output/07_correlations/{experiment}_test_correlations.npy', test_correlations)


@experiment_args
def neuron_description(experiment='000'):
    """
    Generate LaTeX table with comparison of STA correlation and net correlation
        Parameters:
            experiment (str): experiment ID
    """
    sta_corrs = np.load(f'output/03_sta/{experiment}_sta_correlations.npy')
    # np.load(f'output/07_correlations/{experiment}_train_correlations.npy',train_correlations)
    net_corrs = np.load(
        f'output/07_correlations/{experiment}_test_correlations.npy')
    # Neuron correlation on validation
    corrs_df = pd.DataFrame(zip(sta_corrs, net_corrs), columns=[
                            'STA correlation', 'Model correlation'])
    corrs_df['score'] = (1-corrs_df['STA correlation']) * \
        corrs_df['Model correlation']
    corrs_df.sort_values('score', ascending=False, inplace=True)
    with open(f'output/07_correlations/{experiment}_corr_table.tex', 'w') as f:
        f.write(corrs_df.to_latex())


@experiment_args
def compare_sta_mei(chosen_neurons=range(103), experiment='000'):
    """
    Generate plot comparing MEI and linear RFs computed by STA. Both in normalized and unnormalized version. 
    Comparison of activation is saved as LaTeX table
        Parameters:
            chosen_neurons (iterable): Neurons to be on a plot
            experiment (str): experiment ID
    """
    # Load generated meis and stas
    stas = np.load(f'output/03_sta/{experiment}_sta.npy')[chosen_neurons]
    sta_activations = np.load(
        f'output/03_sta/{experiment}_sta_activations.npy')[chosen_neurons]
    stas_n = np.load(f'output/03_sta/{experiment}_sta_n.npy')[chosen_neurons]
    sta_activations_n = np.load(
        f'output/03_sta/{experiment}_sta_activations_n.npy')[chosen_neurons]
    meis = np.load(f'output/04_mei/{experiment}_mei.npy')[chosen_neurons]
    mei_activations = np.load(
        f'output/04_mei/{experiment}_mei_activations.npy')[chosen_neurons]
    meis_n = np.load(f'output/04_mei/{experiment}_mei_n.npy')[chosen_neurons]
    mei_activations_n = np.load(
        f'output/04_mei/{experiment}_mei_activations_n.npy')[chosen_neurons]

    # Mask normalized stimuli
    stas_n = np.vstack([mask_stimuli(sta[np.newaxis, :], experiment, neuron)
                        for neuron, sta in zip(chosen_neurons, stas_n)])
    meis_n = np.vstack([mask_stimuli(mei[np.newaxis, :], experiment, neuron)
                        for neuron, mei in zip(chosen_neurons, meis_n)])

    # Plot unnormalized comparison of linear RFs and MEIs
    col_names = [str(n) for n in chosen_neurons]
    row_names = ['Linear RF', 'MEI']
    titles = [f'{act:.2f}' for act in sta_activations]
    titles += [f'{act:.2f}' for act in mei_activations]
    plot_grid(np.concatenate([stas, meis]), titles, save_path=f'output/05_compare_mei_sta/{experiment}_comparison.png',
              show=False, row_names=row_names, ignore_assertion=True, col_names=col_names, common_scale=False)

    # Plot normalized masked comparison of linear RFs and MEIs
    row_names = ['Linear RF\nnorm', 'MEI\nnorm']
    titles = [f'{act:.2f}' for act in sta_activations_n]
    titles += [f'{act:.2f}' for act in mei_activations_n]
    plot_grid(np.concatenate([stas_n, meis_n]), titles,
              save_path=f'output/05_compare_mei_sta/{experiment}_n_comparison.png', show=False, row_names=row_names, ignore_assertion=True, col_names=col_names)

    # Computes non-linearity score or neuron and save report as latex table
    acts_df = pd.DataFrame(zip(sta_activations, sta_activations_n, mei_activations, mei_activations_n), columns=[
                           'STA activation', 'STA norm activation', 'MEI activation', 'MEI norm activation'])
    acts_df['score'] = acts_df['MEI norm activation'] - \
        acts_df['STA norm activation']
    acts_df.sort_values('score', ascending=False, inplace=True)
    with open(f'output/07_correlations/{experiment}_act_table.tex', 'w') as f:
        f.write(acts_df.to_latex())


@experiment_args
def generate_invariance(
    neuron: int,
    noise_len: int,
    perc: float,
    net: NDN,
    num_representants: int,
    invariance_train_set_len=10000,
    invariance_epochs=5,
    is_aegan=False,
    loss='gaussian',
    mask=False,
    gen_type='conv',
    experiment='000',
    norm='post'
):
    """
    Generate plot comparing MEI and linear RFs computed by STA. Both in normalized and unnormalized version. 
    Comparison of activation is saved as LaTeX table
        Parameters:
            neuron (int): neuron id
            noise_len (int): size of latent space from which generator will sample
            perc (int): Percentage of MEI activation, which will be set as target activation for generator
            net (NDN): Trained neural network
            num_representants (int): number of representants chosen from generated stimuli
            invariance_train_set_len (int): Length of training set
            invariance_epochs (int): Number of epochs for invariance training
            is_aegan (bool): If true, a decoder is added to a generator for regularization purposes
            loss (str): Loss. Same as in NDN ('gaussian','max','poisson','bernoulli')
            mask (bool): Apply mask during learning
            gen_type (str): Generator type. One of : 'lin', 'lin_tanh', 'conv', 'hybrid'
            experiment (str): Experiment ID
            norm (str): Normalization regime one of `online`(normalize also during training) and `post`(normalizing only on evaluation)
            chosen_neurons (iterable): Neurons to be on a plot
    """
    _, input_size_x, input_size_y = net.input_sizes[0]
    # Load precomputed MEI
    max_activation = np.load(
        f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]
    print(max_activation)
    if mask:
        mask = np.load(f'output/02_masks/{experiment}_hardmasks.npy')[neuron]
    else:
        mask = None
    train_log = f'output/tf/{experiment}-{neuron}'
    net_copy = copy.deepcopy(net)
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
        data_len=invariance_train_set_len,
        max_activation=max_activation,
        perc=perc,
        epochs=invariance_epochs,
        train_log=train_log
    )
    image_out = generator_net.generate_stimulus(num_samples=10000)

    # Choose representants with maximal 5 percent deviation from target activation
    x, activations = choose_representant(
        num_representants, net_copy, neuron, image_out, 0.05, 0.05)

    # Plot receptive fields
    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_equivariance.npy',
        np.reshape(x, (-1, input_size_x, input_size_y))
    )
    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_activations.npy',
        activations
    )
    # Save generator model
    generator = generator_net.extract_generator()
    generator.save_model(
        f'output/08_generators/{experiment}_neuron{neuron}_generator.pkl')


@experiment_args
def generate_sta(dataset, net, experiment='000', chosen_neurons=None):
    """
    Generate linear RFs by STA
        Parameters:
            dataset (DataLoader): dataset 
            net (NDN): trained Neural network
            experiment (str): experiment ID
            chosen_neurons (itearble): neuron IDs
    """
    x, y = dataset.train()
    x = np.matrix(x)
    y = np.matrix(y)
    test_x, test_y = dataset.val()

    # Run STA
    sta = STA_LR(x, y, 10000).transpose()
    sta_normalized = norm_samples(sta)

    # Predict activations on both unnormalized and normalized RFs
    activations = net.generate_prediction(sta)
    sta_activations = [activations[i, i] for i in range(len(activations))]

    activations_n = net.generate_prediction(sta_normalized)
    sta_activations_n = [activations_n[i, i]
                         for i in range(len(activations_n))]

    # Compute correlation of linear model
    sta_predictions = np.array(test_x*sta.T)
    sta_correlations = [stats.pearsonr(sta_predictions[:, i], test_y[:, i])[
        0] for i in range(np.shape(y)[1])]

    # Save results
    _, input_size_x, input_size_y = net.input_sizes[0]
    sta = np.array(sta).reshape((-1, input_size_x, input_size_y))
    sta_normalized = np.array(sta_normalized).reshape(
        (-1, input_size_x, input_size_y))
    np.save(f'output/03_sta/{experiment}_sta.npy', sta)
    np.save(f'output/03_sta/{experiment}_sta_n.npy', sta_normalized)
    np.save(f'output/03_sta/{experiment}_sta_activations.npy', sta_activations)
    np.save(
        f'output/03_sta/{experiment}_sta_activations_n.npy', sta_activations_n)
    np.save(
        f'output/03_sta/{experiment}_sta_correlations.npy', sta_correlations)
    titles = [f'{i} - A:{act:.2f} C:{corr:.2f}' for i,
              (act, corr) in enumerate(zip(sta_activations_n, sta_correlations))]
    plot_grid(list(sta_normalized), titles, num_cols=9,
              save_path=f'output/03_sta/{experiment}_sta.png', show=False, highlight=chosen_neurons)


def basic_setup(experiment='000'):
    """
    Generate STA, MEI and mask for `experiment`
    """
    print(f'running experiment {experiment}')
    generate_sta(experiment=experiment)
    generate_mei(experiment=experiment)
    generate_masks(experiment=experiment)


if __name__ == '__main__':
    # Command line interface
    fire.Fire({
        'generate_invariance': generate_invariance,
        'analyze_mei': compare_sta_mei,
        'sta': generate_sta,
        'mei': generate_mei,
        'mask': generate_masks,
        'corr': correlations,
        'neuron_desc': neuron_description,
        'plot_invariances': plot_invariances,
        'plot_interpolations': plot_interpolations,
        'plot_from_generator': plot_from_generator,
        'plot_all_from_generator': plot_all_from_generator,
        'plot_invariance_summary': plot_invariance_summary,
        'plot_sphere': plot_sphere_samples,
        'compare': compare_sta_mei,
        'compare_generators': compare_generators,
        'basic': basic_setup,

    }
    )
