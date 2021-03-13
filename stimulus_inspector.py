import os

import fire
import numpy as np
import matplotlib.pyplot as plt
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
    activations = net.generate_prediction(images)
    activations = activations[:, neuron]
    mask = np.zeros((x_size, y_size))
    for x in range(x_size):
        for y in range(y_size):
            differences = np.zeros(len(try_pixels)*num_images)
            for i, pixel in enumerate(try_pixels):
                modified_images = images[:]
                modified_images[:, x, y] = np.repeat(pixel, num_images)
                predicted_activations = net.generate_prediction(modified_images)[
                    :, neuron]
                differences[i*len(try_pixels):(i+1)*len(try_pixels)
                            ] = predicted_activations - activations
            mask[x, y] = np.std(differences)
    return mask


def compute_mask(net_path, neuron, image_path, min_pixel_val=0, max_pixel_val=0):
    net = NDN.load_model(net_path)
    images = np.load(image_path)
    mask = _compute_mask(
        net,
        neuron,
        images,
        range(
            min_pixel_val,
            max_pixel_val,
            (max_pixel_val-min_pixel_val)/100
        )
    )
    plt.imshow(mask)
    plt.show()
    return mask


if __name__ == '__main__':
    fire.Fire()
