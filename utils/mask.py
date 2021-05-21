
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from NDN3.NDN import NDN

from .experiment_params import experiment_args
from .plotting import plot_grid

def compute_mask(net: NDN, images: np.array, try_pixels=range(0, 2, 1)):
    """
    Computes ROI/masks 
        Parameters:
            net (NDN): A trained neural network
            images (np.array): Images to be presented to the `net` with changing values of pixel luminance
            try_pixels (iterable): Iterable object of values of pixel luminance to add on `images`

        Returns:
            mask (np.array): Array of masks for each neuron
    """
    num_images, num_pixels = np.shape(images)
    activations = net.generate_prediction(images)
    mask = np.zeros(num_pixels)
    net.batch_size = 512
    size_x, size_y = net.input_sizes[0][1:]
    size_out = net.output_sizes[0]

    # Generate images by changing single pixel value
    inputs = []
    for pixel_position in tqdm(range(num_pixels)):
        for i, pixel in enumerate(try_pixels):
            modified_images = np.copy(images)
            modified_images[:, pixel_position] += np.repeat(pixel, num_images)
            inputs.append(modified_images)
    modified_images = np.vstack(inputs)

    # Predict and subtract base activation
    predicted_activations = net.generate_prediction(
        modified_images
    )
    differences = predicted_activations - \
        np.tile(activations, (len(try_pixels)*num_pixels, 1))

    # Compute std of each pixel and arrange result into grid shaped like stimuli
    differences = np.reshape(differences, (size_x, size_y, -1, size_out))
    mask = np.std(differences, axis=2).transpose((2, 0, 1))
    return mask


@experiment_args
def generate_masks(net, dataset, mask_threshold, num_images=50, experiment='000', skip_computing=False):
    """
    Generate masks for every neuron and plot them
        Parameters:
            dataset (DataLoader): dataset 
            net (NDN): trained Neural network
            mask_threshold (float): Threshold of computed deviation for binary mask
            num_images (int): number of samples from which masks will be computed
            experiment (str): experiment ID
            skip_computing (bool): If true skip generating masks and instead of that load them from a file
    """
    x, y = dataset.train()
    x = x[:num_images]

    def mask_pixel(pixel):
        return 1 if pixel > mask_threshold else 0
    # Compute masks
    if skip_computing:
        mask = np.load(f'output/02_masks/{experiment}_masks.npy')
        hard_mask = np.vectorize(mask_pixel)(mask)
    else:
        mask = compute_mask(net, x, np.linspace(-1.6, 1.6, 10))

        # Save masks to npy
        np.save(f'output/02_masks/{experiment}_masks.npy', mask)

    # Binary mask
    hard_mask = np.vectorize(mask_pixel)(mask)
    np.save(f'output/02_masks/{experiment}_hardmasks.npy', hard_mask)
    # Plot masks
    plot_grid(
        mask, save_path=f'output/02_masks/{experiment}_masks_plot.png', cmap=plt.cm.hot, ignore_assertion=True)
    plot_grid(hard_mask, save_path=f'output/02_masks/{experiment}_hardmasks_plot.png',
              cmap=plt.cm.hot, ignore_assertion=True, common_scale=False)
