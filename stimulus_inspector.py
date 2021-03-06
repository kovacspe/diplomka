import os

import fire
import numpy as np
from NDN3.NDN import NDN

from MEIutils import plot_rfs

def create_shifts(stimulus,axis):
    shifts = []
    for i in range(np.shape(stimulus)[axis]):
        shifts.append(np.reshape(np.roll(stimulus,i,axis=axis),(-1)))
    shifts = np.stack(shifts)
    return shifts


def inspect_stimulus(net_path, neuron, stimulus_path,save_path=None):
    net = NDN.load_model(net_path)
    stimulus = np.load(stimulus_path)
    stimulus = np.reshape(stimulus,(31,31))
    horizontal_shifts = create_shifts(stimulus,0)
    vertical_shifts = create_shifts(stimulus,1)
    for stimulus, name in zip([horizontal_shifts, vertical_shifts],
                             ['horizontal shift', 'vertical shift']):
        activations = net.generate_prediction(stimulus)
        activations = activations[:, neuron]
        if save_path is not None:
            save_path = os.path.join(save_path,f'{name.split()[0]}-{neuron}.png')
        plot_rfs(stimulus,activations,save_path)

if __name__ == '__main__':
    fire.Fire(inspect_stimulus)
