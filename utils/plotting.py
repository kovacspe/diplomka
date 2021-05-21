import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from NDN3.NDN import NDN

from .experiment_params import experiment_args
from .misc_utils import sample_sphere, choose_representant, mask_stimuli


def plot_grid(
    images,
    titles=None,
    num_cols=8,
    save_path=None,
    show=False,
    cmap=plt.cm.RdYlBu,
    highlight=None,
    row_names=None,
    col_names=None,
    ignore_assertion=False,
    common_scale=True,
    scale_first_separately=False
):
    """
    Function for generating plot with stimuli
        Parameters:
            images (np.array): neuron id
            titles (List[str]): List of titles for each stimuli
            num_cols (int): Number of columns
            save_path (str): Path to save figure
            cmap (int): colormap
            highlight (iterable): List of image indexes to be highlighted (with bold frame)
            row_names (List[str]):  List of row titles
            col_names (List[str]):  List of column titles
            ignore_assertion (bool): If true assert, zero mean and unit varaince for each image
            common_scale (bool): If true images have common scale and colorbar will be placed at the bottom
            scale_first_separately (bool): If True scale first column with different scale than rest of the figure
    """
    num_images = len(images)
    if titles is None:
        titles = ['']*num_images
    else:
        assert num_images == len(titles)
    min_pixel = np.nanmin(images) if common_scale else None
    max_pixel = np.nanmax(images) if common_scale else None
    if scale_first_separately:
        min_pixel_first = np.nanmin(images[range(0,num_images,num_cols)])
        max_pixel_first = np.nanmax(images[range(0,num_images,num_cols)])

    num_rows = math.ceil(num_images/num_cols)
    fig, ax1 = plt.subplots(
        num_rows, num_cols, figsize=(3*num_cols, 3.05*num_rows))

    for i, (img, tit) in enumerate(zip(images, titles)):
        if not ignore_assertion:
            # Assert zero mean and unit variance of plotted images
            np.testing.assert_almost_equal(
                np.mean(img), 0.0, decimal=2, err_msg='Mean is not 0')
            np.testing.assert_almost_equal(
                np.std(img), 1.0, decimal=2, err_msg='Standard deviation is not 1')
        if scale_first_separately and i%num_cols==0:
            imgp_f = ax1[i // num_cols, i %
                   num_cols].imshow(img, cmap=cmap, vmin=min_pixel_first, vmax=max_pixel_first)
        else:
            imgp = ax1[i // num_cols, i %
                   num_cols].imshow(img, cmap=cmap, vmin=min_pixel, vmax=max_pixel)
        # Remove scale numbers form axis
        ax1[i // num_cols, i % num_cols].set_xticklabels([], [])
        ax1[i // num_cols, i % num_cols].set_yticklabels([], [])

        # Highlight selected images with bold frame
        if highlight is not None and i in highlight:
            [sp.set_linewidth(5) for _, sp in ax1[i //
                                                  num_cols, i % num_cols].spines.items()]

        # Titles
        if row_names is not None:
            ax1[i // num_cols, 0].annotate(row_names[i // num_cols], xy=(0, 0.5), xytext=(-ax1[i // num_cols, 0].yaxis.labelpad - 5, 0),
                                           xycoords=ax1[i // num_cols,
                                                        0].yaxis.label, textcoords='offset points',
                                           size=25, ha='right', va='center')
        if col_names is not None:
            ax1[0, i % num_cols].annotate(col_names[i % num_cols], xy=(0.5, 1.15), xytext=(0, -ax1[0, i % num_cols].xaxis.labelpad + 20),
                                          xycoords='axes fraction', textcoords='offset points',
                                          size=25, ha='center', va='center')
        if len(tit) > 0:
            ax1[i // num_cols, i % num_cols].set_title(tit, fontsize=20)
    for i in range(num_images, num_cols*num_rows):
        fig.delaxes(ax1[i // num_cols, i % num_cols])

    # Adjust axis
    bottom = (0.05 if num_rows > 4 else 0.1) + \
        num_rows*0.002 if common_scale else 0.01
    top = 0.8 if col_names else 0.95
    left = 0.1 if row_names else 0.01
    fig.subplots_adjust(
        bottom=bottom,
        top=top,
        left=left,
        right=0.99,
        wspace=0.02,
        hspace=0.2
    )

    # Colorbar
    if scale_first_separately:
        cb_ax1 = fig.add_axes([left, 0.03 if num_rows >
                              4 else 0.06, 0.85/num_cols, max(0.02, 0.002*(num_rows))])
        cbar1 = fig.colorbar(imgp, cax=cb_ax1, orientation='horizontal')
        left+=0.85/num_cols
        for t in cbar1.ax.get_xticklabels():
            t.set_fontsize(20)
    if common_scale:
        cb_ax = fig.add_axes([left + 0.05, 0.03 if num_rows >
                              4 else 0.06, 0.9-left, max(0.02, 0.002*(num_rows))])
        cbar = fig.colorbar(imgp, cax=cb_ax, orientation='horizontal')
        for t in cbar.ax.get_xticklabels():
            t.set_fontsize(20)

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()


@experiment_args
def plot_interpolations(net, neuron, experiment='000', mask=False, num_interpolations=3, num_samples=6):
    """
    Generate invariances from diameters of unit sphere in latent space
        Parameters:
            net (NDN): trained Neural network
            neuron (int): neuron ID
            experiment (str): experiment ID
            mask (bool): If true stimuli will be masked on a plot
            num_interpolations (int): number of diameters
            num_samples (int): number of sampled points from diameter

    """
    inputs = []
    generator = NDN.load_model(
        f'output/08_generators/{experiment}_neuron{neuron}_generator.pkl')
    size_x, size_y = net.input_sizes[0][1:]
    noise_shape = generator.input_sizes[0][1]
    mei_act = np.load(
        f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]

    # Generate points on sphere
    for i in range(num_interpolations):
        first_point = np.random.normal(0.0, 1.0, noise_shape)
        first_point /= np.linalg.norm(first_point, axis=0)
        inputs.append(np.linspace(first_point, -first_point, num_samples))
    noise_samples = np.vstack(inputs)

    # Predict
    invariances = generator.generate_prediction(noise_samples)
    mask_text = ''
    if mask:
        invariances = mask_stimuli(invariances, experiment, neuron)
        mask_text = '_masked'
    activations = net.generate_prediction(invariances)[:, neuron]
    titles = [f'{act/mei_act:.2f}' for act in activations]
    invariances = np.reshape(invariances, (-1, size_x, size_y))
    plot_grid(invariances, titles, num_cols=6,
              save_path=f'output/06_invariances/{experiment}_{neuron}_interpolations_plot{mask_text}.png', show=False)


@experiment_args
def plot_sphere_samples(net, neuron, experiment='000', mask=False, num_samples=18):
    """
    Generate invariances from points in latent space uniformly sampled from unit sphere
        Parameters:
            dataset (DataLoader): dataset 
            net (NDN): trained Neural network
            experiment (str): experiment ID
            mask (bool): If true stimuli will be masked on a plot
            num_samples (int): number of samples from unit sphere in latent space
    """
    inputs = []
    generator = NDN.load_model(
        f'output/08_generators/{experiment}_neuron{neuron}_generator.pkl')
    size_x, size_y = net.input_sizes[0][1:]
    noise_shape = generator.input_sizes[0][1]

    mei_act = np.load(
        f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]
    noise_samples = sample_sphere(noise_shape, num_samples)
    invariances = generator.generate_prediction(noise_samples)
    mask_text = ''
    if mask:
        invariances = mask_stimuli(invariances, experiment, neuron)
        mask_text = '_masked'
    activations = net.generate_prediction(invariances)[:, neuron]
    titles = [f'{act/mei_act:.2f}' for act in activations]
    invariances = np.reshape(invariances, (-1, size_x, size_y))
    plot_grid(invariances, titles, num_cols=6,
              save_path=f'output/06_invariances/{experiment}_{neuron}_sphere_plot{mask_text}.png', show=False)


@experiment_args
def plot_from_generator(net, neuron, num_representants, experiment='000', include_mei=False, mask=False, max_error=0.05, perc=0.95):
    """
    Generate and plot invariances from generator
        Parameters:
            net (NDN): trained Neural network
            neuron (int): neuron ID
            num_representants (int): number of represntats to plot
            experiment (str): experiment ID
            include_mei (bool): If true MEI will be placed as last image in the figure
            mask (bool): If true stimuli will be masked on a plot
            max_error (float): Max deviation from target activation
            perc (float): Target percentage of MEI activation
    """
    inputs = []
    generator = NDN.load_model(
        f'output/08_generators/{experiment}_neuron{neuron}_generator.pkl')
    _, input_size_x, input_size_y = net.input_sizes[0]
    noise_shape = input_size_x
    mei_act = np.load(
        f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]
    noise_input = np.random.uniform(-2, 2, size=(10000, noise_shape))
    invariances = generator.generate_prediction(noise_input)

    images, activations = choose_representant(num_representants, net, neuron, invariances,
                                              activation_lowerbound=(
                                                  perc-max_error)*mei_act,
                                              activation_upperbound=(perc+max_error)*mei_act)

    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_equivariance.npy',
        np.reshape(images, (-1, input_size_x, input_size_y))
    )
    np.save(
        f'output/06_invariances/{experiment}_neuron{neuron}_activations.npy',
        activations
    )
    plot_invariances(experiment=experiment, neuron=neuron,
                     include_mei=include_mei, mask=mask)


@experiment_args
def plot_invariances(net, neuron, experiment='000', mask=False, include_mei=False):
    """
    Generate and plot invariances from generator
        Parameters:
            net (NDN): trained Neural network
            neuron (int): neuron ID
            experiment (str): experiment ID
            mask (bool): If true stimuli will be masked on a plot
            include_mei (bool): If true MEI will be placed as last image in the figure
    """
    invariances = np.load(
        f'output/06_invariances/{experiment}_neuron{neuron}_equivariance.npy')
    activations = np.load(
        f'output/06_invariances/{experiment}_neuron{neuron}_activations.npy')
    mei_act = np.load(
        f'output/04_mei/{experiment}_mei_activations_n.npy')[neuron]
    print('Mean invariance images activations' + str(np.mean(activations[0])))
    print(f'MEI activation: {mei_act}')
    size_x, size_y = net.input_sizes[0][1:]

    if include_mei:
        mei = np.load(f'output/04_mei/{experiment}_mei_n.npy')[neuron]
        invariances = np.vstack(
            [np.reshape(invariances, (-1, np.shape(mei)[1])), mei])
    mask_text = ''
    invariances = np.reshape(invariances, (-1, size_x, size_y))

    if mask:
        invariances = mask_stimuli(invariances, experiment, neuron)
        mask_text = '_masked'

    titles = [f'{act/mei_act:.2f}' for act in activations] + ['1.00']
    plot_grid(invariances, titles, num_cols=5,
              save_path=f'output/06_invariances/{experiment}_{neuron}_plot{mask_text}.png', show=False)


@experiment_args
def plot_all_from_generator(chosen_neurons, experiment='000', mask=False, include_mei=False, max_error=0.05):
    """
    Generate and plot invariances from generator for all `chosen_neurons`. Each neuron is plotted in a single plot.
    Note that all unfilled arguments for plot_from_generator must be in experiment definition
        Parameters:
            chosen_neurons (List[int]): neuron IDs
            experiment (str): experiment ID
            include_mei (bool): If true MEI will be placed as last image in the figure
            max_error (float): Max deviation from target activation
    """
    for neuron in chosen_neurons:
        plot_from_generator(neuron=neuron, experiment=experiment,
                            include_mei=include_mei, mask=mask, max_error=max_error)


@experiment_args
def plot_invariance_summary(net, chosen_neurons, perc, experiment='000', mask=False, max_error=0.05, num_samples=8,scale_first_separately=False):
    """
    Generate invariances from generator for all `chosen_neurons` and plot them in common summary with MEIs
        Parameters:
            net (NDN): trained Neural network
            chosen_neurons (List[int]): neuron IDs
            perc (float): Target percentage of MEI activation
            experiment (str): experiment ID
            mask (bool): If true stimuli will be masked on a plot
            max_error (float): Max deviation from target activation
            num_samples (int): number of samples for each neuron
    """
    row_names = [str(n) for n in chosen_neurons]
    mei = np.load(f'output/04_mei/{experiment}_mei_n.npy')
    mei_act = np.load(f'output/04_mei/{experiment}_mei_activations_n.npy')
    size_x, size_y = net.input_sizes[0][1:]
    titles = []
    all_images = []

    for neuron in chosen_neurons:
        generator = NDN.load_model(
            f'output/08_generators/{experiment}_neuron{neuron}_generator.pkl')
        noise_shape = generator.input_sizes[0][1]
        noise_input = np.random.uniform(-2, 2, size=(1000, noise_shape))
        invariances = generator.generate_prediction(noise_input)
        images, activations = choose_representant(num_samples, net, neuron, invariances,
                                                  activation_lowerbound=(
                                                      perc-max_error)*mei_act[neuron] if perc is not None else -np.inf,
                                                  activation_upperbound=(perc+max_error) *
                                                  mei_act[neuron] if perc is not None else np.inf
                                                  )
        #print(mask_stimuli)
        images[0] = np.reshape(mask_stimuli(mei[[neuron]],experiment,neuron), (1, -1))
        all_images.append(images)
        activations[0] = mei_act[neuron]
        titles += [f'{act/activations[0]:.2f}' for act in activations]
    all_images = np.reshape(np.vstack(all_images), (-1, size_x, size_y))
    plot_grid(all_images, titles, num_cols=num_samples,
              save_path=f'output/08_generators/{experiment}_invariance_overview.png', show=False, row_names=row_names,scale_first_separately=scale_first_separately,ignore_assertion=True)


@experiment_args
def compare_generators(neuron, net, experiment='000', generator_experiment=[], generator_names=[], percentage=[], num_per_net=6, mask=False, max_error=0.05):
    """
    Compare generators from different experiments on same neuron and plot results. Experiments must share net and dataset parameters
        Parameters:
            neuron (int): neuron ID
            net (NDN): trained Neural network
            experiment (str): experiment ID
            generator_experiment (List[str]): List of experiment IDs to compare
            generator_names (List[str]): List of generator names. Will be displayed on a plot
            percentage (List[float]): list of target percentage of MEI for each experiment
            num_per_net (int): Number of representants per experiment
            mask (bool): If true stimuli will be masked on a plot
            max_error (float): Max deviation from target activation
    """
    images = []
    titles = []
    base_exp = generator_experiment[0]
    mei_act = np.load(
        f'output/04_mei/{base_exp}_mei_activations_n.npy')[neuron]
    noise_input = np.random.uniform(-2, 2, size=(10000, 128))
    mei = np.load(f'output/04_mei/{base_exp}_mei_n.npy')[neuron]
    size_x, size_y = net.input_sizes[0][1:]

    for generator_exp, generator_name, perc in zip(generator_experiment, generator_names, percentage):
        generator = NDN.load_model(
            f'output/08_generators/{generator_exp}_neuron{neuron}_generator.pkl')
        invariances = generator.generate_prediction(noise_input)
        invariances, activations = choose_representant(num_per_net, net, neuron, invariances,
                                                       activation_lowerbound=(
                                                           perc-max_error)*mei_act if perc is not None else -np.inf,
                                                       activation_upperbound=(perc+max_error)*mei_act if perc is not None else np.inf)
        if len(invariances) < num_per_net:
            raise ValueError('Cannot generate samples with given conditions')
        images.append(invariances)
        titles += list(activations)
    images.append(np.reshape(mei, (1, -1)))
    images = np.vstack(images)

    titles.append(mei_act)
    titles = [f'{tit/mei_act:.2f}' for tit in titles]

    images = np.reshape(images, (-1, size_x, size_y))
    if mask:
        images = mask_stimuli(images, base_exp, neuron)

    plot_grid(
        images,
        titles,
        num_cols=num_per_net,
        save_path=f'output/08_generators/neuron{neuron}_compare_generator.png',
        row_names=generator_names+['MEI'],
        ignore_assertion=True
    )
