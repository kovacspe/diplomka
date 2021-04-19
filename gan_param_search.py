import os
import yaml
import sys
NN = [83,28,49,23,18,102]
NEURONS = [28,74,66,23,83,]#53,54,72,50,24]
name = sys.argv[1] if len(sys.argv)>1 else 'untitled'
models = [
    'models/best_dog.pkl',
    #'models/conv-dog-convDoG-c_size15-layersep-reg_h0.01-hidden9-exp_nameconv-dog.pkl'
]

for exp in ['000dog', '001dog', '002dog', '003dog']:
    with open('utils/experiments.yml','r') as conf_file:
        loaded_yaml = yaml.load(conf_file)
    loaded_params = loaded_yaml.get(exp,{})
    neurons = loaded_params['chosen_neurons']
    for neuron in neurons:
        os.system(
            f'qsub -q cpu.q -cwd -pe smp 4 -l mem_free=8G,act_mem_free=8G,h_data=20G  \
            -o job_output/gan/o-{neuron}.log \
            -e job_output/gan/e-{neuron}.log \
            ./scripts/run_gan_cpu.sh generate_equivariance --neuron={neuron} --experiment {exp}')