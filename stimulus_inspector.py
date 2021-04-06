import os

import fire
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm
from NDN3.NDN import NDN

from MEIutils import plot_rfs


def create_shifts(stimulus, axis):
    shifts = []
    for i in range(np.shape(stimulus)[axis]):
        shifts.append(np.reshape(np.roll(stimulus, i, axis=axis), (-1)))
    shifts = np.stack(shifts)
    return shifts


def inspect_stimulus(net_path, neuron, stimulus_path, save_path=None, image_index=1):
    net = NDN.load_model(net_path)
    stimulus = np.load(stimulus_path)
    stimulus = stimulus[image_index]
    horizontal_shifts = create_shifts(stimulus, 0)
    vertical_shifts = create_shifts(stimulus, 1)
    for stimulus, name in zip([horizontal_shifts, vertical_shifts],
                              ['horizontal shift', 'vertical shift']):
        activations = net.generate_prediction(stimulus)
        activations = activations[:, neuron]
        if save_path is not None:
            save_path = os.path.join(
                save_path, f'{name.split()[0]}-{neuron}.png')
        plot_rfs(stimulus, activations, save_path)



def _compute_mask(net: NDN, neuron: int, images: np.array, try_pixels=range(0, 255, 1)):
    num_images, x_size, y_size = np.shape(images)
    def NDNreshape(images):
        return np.reshape(images,(-1,x_size*y_size))
    activations = net.generate_prediction(NDNreshape(images))
    activations = activations[:, neuron]
    mask = np.zeros((x_size, y_size))
    net.batch_size = 512
    inputs = []
    for x in tqdm(range(x_size)):
        for y in range(y_size):
            for i, pixel in enumerate(try_pixels):
                modified_images = np.copy(images)
                modified_images[:, x, y]+= np.repeat(pixel, num_images)
                inputs.append(modified_images)
    modified_images = np.vstack(inputs)
    print('Prediction')
    predicted_activations = net.generate_prediction(
                    NDNreshape(modified_images)
                    )[:, neuron]
    print('End of prediction')
    differences = predicted_activations - np.tile(activations,len(try_pixels)*x_size*y_size)
    differences = np.reshape(differences,(31,31,-1))
    mask = np.std(differences,axis=2)
    print(mask)
    return mask


def compute_mask(net_path, neuron, image_path, min_pixel_val=-6, max_pixel_val=1.6):
    net = NDN.load_model(net_path)
    images = np.load(image_path)
    num_images, x_size, y_size = np.shape(images)
    for i in range(10):
        images[10+i] = np.zeros((1,x_size,y_size))+(i-3)
    
    mask = _compute_mask(
        net,
        neuron,
        images,
        np.linspace(
            -3,
            3,
            30
        )
    )
    # Gaussian blur
    plt.imshow(mask)
    plt.show()
    mask = sc.ndimage.gaussian_filter(mask,1.0)
    plt.imshow(mask)
    plt.show()
    return mask


if __name__ == '__main__':
    fire.Fire()
