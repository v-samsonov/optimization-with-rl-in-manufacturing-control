import datetime
import os
import sys
import time
from pathlib import Path

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

def run_experiment(run_config_file_names, exp_alias, wandb_logs):
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
            experiment_train_eval_from_scratch_with_checkpoints(rand_seed=int(rand_seed),
                                                                jsp_trn_lst=jsp_trn_lst,
                                                                jsp_eval_lst=jsp_eval_lst,
                                                                date_stamp=date_stamp,
                                                                wandb_logs=wandb_logs,
                                                                save_models_to_wandb=True,
                                                                exp_alias=exp_alias,
                                                                wandb_project=wandb_project,
                                                                wandb_entity=wandb_entity,
                                                                **run_config_pars)

        end_t = time.time()
        print(f'execution time: {(end_t - start_t) / 60} mins')


if __name__ == '__main__':
    # get run config parameters
    exp_alias = sys.argv[1]
    run_config_file_names = sys.argv[2:]  # if file executed from terminal - get arguments passed in terminal
    run_experiment(run_config_file_names=run_config_file_names,
                   wandb_logs=True,
                   exp_alias=exp_alias)
