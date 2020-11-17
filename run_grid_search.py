import os
import sys
import run_job
sys.path.append(os.getcwd())


def basic_grid_search():
    exp = "basicFC"
    run = 0
    for hidden in [0.1, 0.2]:
        for reg_h in [0.5, 1, 10]:
            for reg_l in [0.1, 0.5]:
                run_job.run_qsub_cpu(
                    exp,
                    f'--model_type=\'base\' --data_type=\'antolik1\' --hidden={hidden} --reg_h={reg_h} --reg_l={reg_l}',
                    run
                )
                run += 1

def basic_test():
    exp = "basicFC"
    run = 0
    for hidden in [0.1,0.2]:
        for reg_h in [1]:
            for reg_l in [0.5]:
                run_job.run_qsub_cpu(
                    exp,
                    f'--model_type=\'base\' --data_type=\'antolik1\' --hidden={hidden} --reg_h={reg_h} --reg_l={reg_l}',
                    run
                )
                run += 1


def conv_sep_grid_search():
    exp = "conv-sep"
    run = 0
    for hidden_lt in ['sep']:
        for c_size in [3, 7, 15]:
            for c_filters in [9, 30]:
                for cd2x in [0.01, 0.1]:
                    for hidden_t in ['l1', 'l2']:  # l1 as described in the original paper
                        for hidden_s in [0.01, 0.1, 1]:
                            run_job.run_qsub_cpu(
                                exp,
                                f'--model_type=\'conv\' --data_type=\'antolik1\' --c_size={c_size} --channels={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt={hidden_lt}',
                                run
                            )
                            run += 1


def conv_normal_grid_search():
    run = 0
    exp = "conv-normal"
    for c_size in [3, 7, 15]:
        for c_filters in [9, 30]:
            for cd2x in [0.01, 0.1]:
                for hidden_t in ['max', 'l2']:
                    for hidden_s in [0.01, 0.1, 1]:
                        run_job.run_qsub_cpu(
                            exp,
                            f'--model_type=\'conv\' --data_type=\'antolik1\' --c_size={c_size} --channels={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt=\'normal\'',
                            run
                        )
                        run += 1

def iclr_grid_search():
    run = 0
    exp = "iclr-basic"

    run_job.run_qsub_gpu(
        exp,
        f'--model_type=\'iclr\' --data_type=\'iclr\' --channels=16',
        run
    )
    run += 1

def iclr_antolik_grid_search():
    run = 0
    exp = "iclr"
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

def best_models():
    # Fully connected
    exp = "basicFC"
    run = 0
    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'base\' --data_type=\'antolik1\' --hidden={0.2} --reg_h={0.1} --reg_l={0.1}',
        run
    )
    run += 1

    # Convolutional
    exp = "conv"
    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'conv\' --data_type=\'antolik1\' --c_size={15} --channels={30} --cd2x={0.1} --hidden_t=max --hidden_s={1} --hidden_lt=conv',
        run
    )
    run += 1

    # Convolutional
    exp = "sep"
    run_job.run_qsub_cpu(
        exp,
        f'--model_type=\'conv\' --data_type=\'antolik1\' --c_size={3} --channels={9} --cd2x={0.1} --hidden_t=l2 --hidden_s={0.1} --hidden_lt=sep',
        run
    )
    run += 1

    


if __name__ == "__main__":
    # conv_normal_grid_search()
    # conv_sep_grid_search()
    # basic_grid_search()
    #iclr_antolik_grid_search()
    best_models()