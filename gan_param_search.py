import os
import sys
NEURONS = [28,74,66,23,83,53,54,72,50,24]
PERC= [0.95,0.85,0.75]
name = sys.argv[1] if len(sys.argv)>1 else 'untitled'
models = [
    #'models/dog-basicFC-exp_namedog.pkl',
    'models/conv-dog-convDoG-c_size15-layersep-reg_h0.01-hidden9-exp_nameconv-dog.pkl'
]
for neuron in NEURONS:
    for perc in PERC:
        for model in models:
            for noise_len in [256,512]:
                os.system(
                    f'qsub -q cpu.q -cwd -pe smp 4 -l mem_free=8G,act_mem_free=8G,h_data=20G  \
                    -o job_output/gan/o-{neuron}.log \
                    -e job_output/gan/e-{neuron}.log \
                    ./scripts/run_gan_cpu.sh --neuron={neuron} --save_path="output/gan_output" --noise_len={noise_len} --perc={perc} --name="{name}" --model="{model}" --epochs=10')