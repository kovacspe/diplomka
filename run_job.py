import os


def prepare_paths(exp_folder, exp):
    exp_path = f"./experiments/{exp_folder}"
    logs_folder = f"./training_data/job_logs/{exp_folder}/{exp}"
    os.makedirs(logs_folder, exist_ok=True)

    return exp_path, logs_folder


def run_qsub_cpu(exp_name, exp_args, run):
    '''
    Runs `./experiments/exp_folder/exp exp_args` and logs everything along the way.
    '''
    # CPU: qsub -q cpu.q -cwd -pe smp 4 -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G

    os.system(
        f"qsub -q cpu.q -cwd -pe smp 4 -l mem_free=8G,act_mem_free=8G,h_data=20G \
        -o job_output/o-{exp_name}-{run}.log \
        -e job_output/e-{exp_name}-{run}.log \
            ./scripts/run_on_cpu_AIC.sh {exp_args} --exp_name={exp_name}")


def run_qsub_gpu(exp_name, exp_args, run):
    '''
    Runs `./experiments/exp_folder/exp exp_args` and logs everything along the way.
    '''
    # GPU: qsub -q gpu.q -cwd -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G

    os.system(
        f"qsub -q gpu.q -cwd -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G \
        -o job_output/{exp_name}/o-{run}.log \
        -e job_output/{exp_name}/e-{run}.log \
            ./scripts/run_on_gpu_AIC.sh {exp_args} --exp_name={exp_name}")

def run_local(exp_name, exp_args, run):
    '''
    Runs `./experiments/exp_folder/exp exp_args` and logs everything along the way.
    '''
    # CPU: qsub -q cpu.q -cwd -pe smp 4 -l gpu=1,mem_free=8G,act_mem_free=8G,h_data=20G

    os.system(
        f"python trainer.py {exp_args} --exp_name={exp_name}")