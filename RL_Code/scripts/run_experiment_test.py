import datetime
import os
import sys
import time
from pathlib import Path

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

sys.path.insert(0, './')
if Path(os.getcwd()).name == 'scripts':
    os.chdir(str(Path().cwd().parents[1]))

from RL_Code.modules.utils.experiment_routine import experiment_train_eval_from_scratch_with_checkpoints
from RL_Code.modules.utils.file_handling.yaml_to_dict_parser import get_config_pars
from RL_Code.modules.utils.jsp_handling.collect_jsp_tasks import collect_transform_jsp_tasks


#
# if Path(os.getcwd()).name == 'scripts':
#     os.chdir(str(Path().cwd().parents[1]))
#     # os.chdir([p for p in Path(os.getcwd()).parents if p.name == 'RL_Code'][0])
# sys.path.insert(0, str(Path().cwd().parents[1]))

def run_experiment_test(run_config_file_names, wandb_logs, save_models_to_wandb, wandb_project, exp_alias):
    start_t = time.time()
    print(f'tasks to execute: {run_config_file_names}')

    # execute a list of experiments
    for run_config_file_name in run_config_file_names:
        # get parameters dict
        run_config_file_path = Path.cwd() / Path('Data/run_config_files/{}'.format(run_config_file_name))
        run_config_pars = get_config_pars(run_config_file_path)

        print(f'RUNNING: {run_config_pars["rl_framework_tag"]}')

        # get jsp tasks for training
        jsp_file_path = Path.cwd() / Path(run_config_pars['jsp_path']) / Path(run_config_pars['jsp_size'])
        # ToDo: read multiple jsp sizes
        jsp_trn_lst = collect_transform_jsp_tasks(jsp_ind_start=run_config_pars['jsp_ind_start_trn'],
                                                  jsp_ind_end=run_config_pars['jsp_ind_end_trn'],
                                                  read_path=jsp_file_path, **run_config_pars)
        # get jsp task for evaluation
        jsp_eval_lst = collect_transform_jsp_tasks(jsp_ind_start=run_config_pars['jsp_ind_start_eval'],
                                                   jsp_ind_end=run_config_pars['jsp_ind_end_eval'],
                                                   read_path=jsp_file_path, **run_config_pars)

        # train and evaluate
        date_stamp = str(datetime.datetime.now())[:19].replace(' ', '_').replace('-', '.').replace(':', '.')  # used
        # to group related runs together
        wandb_project = 'validate_rl4jsp_submission'  # os.environ.get('WANDB_PROJECT')
        wandb_entity = os.environ.get('WANDB_ENTITY')
        for rand_seed in run_config_pars['rand_seed_lst']:
            run_metadata = experiment_train_eval_from_scratch_with_checkpoints(rand_seed=rand_seed,
                                                                               jsp_trn_lst=jsp_trn_lst,
                                                                               jsp_eval_lst=jsp_eval_lst,
                                                                               date_stamp=date_stamp,
                                                                               wandb_logs=wandb_logs,
                                                                               save_models_to_wandb=save_models_to_wandb,
                                                                               exp_alias=exp_alias,
                                                                               wandb_project=wandb_project,
                                                                               wandb_entity=wandb_entity,
                                                                               **run_config_pars)

        end_t = time.time()
        print(f'execution time: {(end_t - start_t) / 60} mins')


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('run_experiment_test()', filename='restats_1000_inst_40000_steps')

    # config parameters
    exp_alias = 'clean_up_test2'
    wandb_project='validate_rl4jsp_submission'
    run_config_file_names = [
                            # comparison_of_reward_shapes
                            # 'test/comparison_of_reward_shapes/dense_lb_rew_dqn_6x6x6_900jssp_eps02_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/dense_lb_rew_dqn_8x8x8_900jssp_eps02_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/dense_lb_rew_dqn_10x10x10_900jssp_eps02_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/dqn_6x6x6_900jssp_eps02_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/dqn_8x8x8_900jssp_eps02_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/dqn_10x10x10_900jssp_eps02_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/r2_70_makespan_900jsp_eps02_explr_incent_01_6x6x6_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/r2_70_makespan_900jsp_eps02_explr_incent_01_8x8x8_0i_run_config.yml',
                            # 'test/comparison_of_reward_shapes/r2_70_makespan_900jsp_eps02_explr_incent_01_10x10x10_0i_run_config.yml',
                            # direct_vs_indirect_action_space_mapping
                            # 'test/direct_vs_indirect_action_space_mapping/direct_act_space_900jsp_6x6x6_0i_run_config.yml',
                            # 'test/direct_vs_indirect_action_space_mapping/direct_act_space_900jsp_8x8x8_0i_run_config.yml',
                            # 'test/direct_vs_indirect_action_space_mapping/direct_act_space_900jsp_10x10x10_0i_run_config.yml',
                            # 'test/direct_vs_indirect_action_space_mapping/dqn_6x6x6_900jssp_eps02_0i_run_config.yml',
                            # 'test/direct_vs_indirect_action_space_mapping/dqn_8x8x8_900jssp_eps02_0i_run_config.yml',
                            # 'test/direct_vs_indirect_action_space_mapping/dqn_10x10x10_900jssp_eps02_0i_run_config.yml',
                            # enhanced_baseline_scalability_6x6x6_2_15x15x15
                            # 'test/enhanced_baseline_scalability_6x6x6_2_15x15x15/test_r2_70_makespan_6x6x6_2_15x15x15_900jsp_eps02_shuffle_explr_incent_01_6x6x6_0i_eval_config.yml',
                            'test/enhanced_baseline_scalability_6x6x6_2_15x15x15/test_r2_70_makespan_6x6x6_2_15x15x15_900jsp_eps02_shuffle_explr_incent_01_6x6x6_0i_run_config.yml',
                            # enhanced_baseline_vs_l2d_Zhang_et_al
                            # 'test/enhanced_baseline_vs_l2d_Zhang_et_al/bench_l2d_6x6x6_cpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/enhanced_baseline_vs_l2d_Zhang_et_al/bench_l2d_8x8x8_cpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/enhanced_baseline_vs_l2d_Zhang_et_al/bench_l2d_10x10x10_cpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/enhanced_baseline_vs_l2d_Zhang_et_al/r2_70_makespan_900jsp_eps02_explr_incent_01_6x6x6_0i_run_config.yml',
                            # 'test/enhanced_baseline_vs_l2d_Zhang_et_al/r2_70_makespan_900jsp_eps02_explr_incent_01_8x8x8_0i_run_config.yml',
                            # 'test/enhanced_baseline_vs_l2d_Zhang_et_al/r2_70_makespan_900jsp_eps02_explr_incent_01_10x10x10_0i_run_config.yml',
                            # evaluation_l2d_Zhang_et_al
                            # 'test/evaluation_l2d_Zhang_et_al/bench_l2d_6x6x6_cpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/evaluation_l2d_Zhang_et_al/bench_l2d_6x6x6_gpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/evaluation_l2d_Zhang_et_al/bench_l2d_8x8x8_cpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/evaluation_l2d_Zhang_et_al/bench_l2d_8x8x8_gpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/evaluation_l2d_Zhang_et_al/bench_l2d_10x10x10_cpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # 'test/evaluation_l2d_Zhang_et_al/bench_l2d_10x10x10_gpu_4_env_rl4jsp_gen_0i_run_config.yml',
                            # shuffling_vs_constant_state
                            # 'test/shuffling_vs_constant_state/baseline_900jsp_dqn_6x6x6_0i_run_config.yml',
                            # 'test/shuffling_vs_constant_state/shuffle_900jsp_dqn_6x6x6_0i_run_config.yml',
                            ]

    run_experiment_test(run_config_file_names=run_config_file_names, wandb_logs=True, save_models_to_wandb=True,
                        wandb_project=wandb_project, exp_alias=exp_alias)

