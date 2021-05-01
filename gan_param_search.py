import os
import yaml
import sys

nets = sys.argv[1:]
with open('utils/experiments.yml','r') as conf_file:
    loaded_yaml = yaml.load(conf_file)
for exp in nets:
    loaded_params = loaded_yaml.get(exp,{})
    neurons = loaded_params['chosen_neurons']
    for neuron in neurons:
        print(f'Running {exp}-{neuron}')
        os.system(
            f'qsub -q cpu.q -cwd -pe smp 4 -l mem_free=8G,act_mem_free=8G,h_data=20G  \
            -o job_output/gan/o-{neuron}.log \
            -e job_output/gan/e-{neuron}.log \
            ./scripts/run_gan_cpu.sh generate_equivariance --neuron={neuron} --experiment {exp}')