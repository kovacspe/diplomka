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
                for seed in range(0, 10):
                    run_job.run_qsub_cpu(
                        exp,
                        f'--model_type=\'base\' --data_type=\'antolik1\' --hidden={hidden} --reg_h={reg_h} --reg_l={reg_l} --seed={seed}',
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
                        run_job.run_qsub_gpu(
                            exp,
                            f'--model_type=\'conv\' --data_type=\'antolik1\' --c_size={c_size} --channels={c_filters} --cd2x={cd2x} --hidden_t={hidden_t} --hidden_s={hidden_s} --hidden_lt=\'normal\'',
                            run
                        )
                        run += 1


if __name__ == "__main__":
    conv_normal_grid_search()
