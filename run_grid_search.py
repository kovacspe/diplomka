import os
import sys
import run_job
import fire
sys.path.append(os.getcwd())


def basic_grid_search(run_id="",dataset=1):
    exp = f'{run_id}-basicFC'
    run = 0
    for hidden in [0.1, 0.2]:
        for reg_h in [0.5, 1, 10]:
            for reg_l in [0.1, 0.5]:
                run_job.run_qsub_cpu(
                    exp,
                    f'--model_type=\'base\' --data_type=\'antolik{dataset}\' --hidden={hidden} --reg_h={reg_h} --reg_l={reg_l}',
                    run
                )
                run += 1


def conv_sep_grid_search(run_id="",dataset=1):
    exp = f'{run_id}-convsep'
    run = 0
    for hidden_lt in ['sep']:
        for c_size in [3, 7, 15]:
            for c_filters in [9, 30]:
                for cd2x in [0.01, 0.1]:
                    for hidden_t in ['l1', 'l2']:  # l1 as described in the original paper
                        for hidden_s in [0.01, 0.1, 1]:
                            run_job.run_qsub_cpu(
                                exp,
                                f'--model_type=\'conv\' --data_type=\'antolik{dataset}\' --c_size={c_size} --channels={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt={hidden_lt}',
                                run
                            )
                            run += 1


def conv_normal_grid_search(run_id='000',dataset=1):
    run = 0
    exp = f'{run_id}-conv'
    for c_size in [3, 7, 15]:
        for c_filters in [9, 30]:
            for cd2x in [0.01, 0.1]:
                for hidden_t in ['max', 'l2']:
                    for hidden_s in [0.01, 0.1, 1]:
                        run_job.run_qsub_cpu(
                            exp,
                            f'--model_type=\'conv\' --data_type=\'antolik{dataset}\' --c_size={c_size} --channels={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt=\'normal\'',
                            run
                        )
                        run += 1

def iclr_grid_search(run_id='000'):
    run = 0
    exp = f'{run_id}-iclr'

    run_job.run_qsub_gpu(
        exp,
        f'--model_type=\'iclr\' --data_type=\'iclr\' --channels=16',
        run
    )
    run += 1

def iclr_antolik_grid_search(run_id='000'):
    run = 0
    exp = f'{run_id}-conv_iclrdata'
    for c_size in [3, 15]:
        for c_filters in [15]:
            for cd2x in [0.01, 0.1]:
                for hidden_t in ['l2']:
                    for hidden_s in [0.01, 0.1]:
                        run_job.run_qsub_cpu(
                            exp,
                            f'--model_type=\'conv\' --data_type=\'iclr\' --c_size={c_size} --channels={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt=\'normal\'',
                            run
                        )
                        run += 1

def dog_models(run_id='000',dataset=1):
    # Dog
    run=0
    exp = f'{run_id}-dog'
    for filt_size in [9,16,32]:
        for perc_output in [0.2,0.4,0.6]:
            run_job.run_qsub_cpu(
                exp,
                f'--model_type=\'dog\' --data_type=\'antolik{dataset}\' --filt_size={filt_size} --perc_output={perc_output}',
                run
            )

def convdog_models(run_id='000',dataset=1):
    run = 0
    # Convolutional DoG
    exp = f'{run_id}-convdog'
    for c_size in [7,15]:
        for layer in ['normal','sep']:
            for reg in [0.1,0.01]:
                run_job.run_qsub_cpu(
                    exp,
                    f'--model_type=\'convdog\' --data_type=\'antolik{dataset}\' --c_size={c_size} --layer=\'{layer}\' --reg_h={reg} --hidden=9',
                    run
                )


def best_models(run_id='000',dataset=1):
    # Fully connected
    exp = f'{run_id}-best_'
    #exp = "basicFC"
    run = 0
    #run_job.run_local(
    #   exp,
    #    f'--model_type=\'base\' --data_type=\'antolik1\' --hidden={0.2} --reg_h={0.#1} --reg_l={0.1}',
    #    run
    #)
    #run += 1

    # Convolutional
    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'conv\' --data_type=\'antolik{dataset}\' --c_size={7} --channels={30} --cd2x={0.1} --hidden_t=max --hidden_s={1} --hidden_lt=normal',
        run
    )
    run += 1

    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'conv\' --data_type=\'antolik{dataset}\' --c_size={3} --channels={9} --cd2x={0.1} --hidden_t=max --hidden_s={1} --hidden_lt=normal',
        run
    )
    run += 1

    # Convolutional
    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'conv\' --data_type=\'antolik{dataset}\' --c_size={15} --channels={30} --cd2x={0.1} --hidden_t=l1 --hidden_s={0.2} --hidden_lt=sep',
        run
    )
    run += 1
    # Dog
    run=0
    
    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'dog\' --data_type=\'antolik{dataset}\' --filt_size=9 --perc_output=0.2',
        run
    )
    run += 1
    # Convolutional DoG
    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'convdog\' --data_type=\'antolik{dataset}\' --c_size={15} --layer=\'sep\' --reg_h=0.01 --hidden=9',
        run
    )

    


if __name__ == "__main__":
    fire.Fire(
        {
            'conv_normal': conv_normal_grid_search(),
            'conv_sep': conv_sep_grid_search(),
            'fc': basic_grid_search(),
            'rotation': iclr_grid_search(),
            'rotation_antolik_data': iclr_antolik_grid_search(),
            'best': best_models(),
            'dog': dog_models(),
            'convdog': convdog_models()
        }
    )
