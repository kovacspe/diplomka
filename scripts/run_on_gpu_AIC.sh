source ~/Envs/diplomka-gpu/bin/activate
LD_LIBRARY_PATH="/lnet/aic/opt/cuda/cuda-10.0/lib64"
~/Envs/diplomka-gpu/bin/python3 ~/diplomka/trainer.py --gpu $@ 