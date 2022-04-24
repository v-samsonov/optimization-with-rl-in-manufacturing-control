import os
import sys
from pathlib import Path

import pandas as pd
import wandb

sys.path.insert(0, './')
if Path(os.getcwd()).name == 'scripts':
    os.chdir(str(Path().cwd().parents[1]))

from RL_Code.modules.utils.file_handling.yaml_to_dict_parser import get_config_pars
from RL_Code.modules.utils.jsp_handling.collect_jsp_tasks import collect_transform_jsp_tasks
from RL_Code.modules.utils.experiment_routine import experiment_eval_pretrained_agents

# fix wandb config path to avoid access permission problems in docker
os.environ['WANDB_SILENT'] = 'true'
WANDB_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "wandb")
print('init config dir', WANDB_CONFIG_DIR)
if (WANDB_CONFIG_DIR == '/.config') or (WANDB_CONFIG_DIR == '/.config/wandb'):
    os.environ['WANDB_CONFIG_DIR'] = Path.cwd().as_posix()
    print('changed config dir', WANDB_CONFIG_DIR)


def get_agents_from_wandb(filt_dicts, project, entity):
    api = wandb.Api()
    runs = api.runs(entity + "/" + project)  # filters={"experiment_id": "dqn_6x6x6_900jssp_0i"}
    run_dict_eval = {}
    run_dict_train = {}
    exp_alias_lst = []
    for exp_key, filt_dict in filt_dicts.items():
        run_dict_eval[exp_key] = []
        run_dict_train[exp_key] = []
    for run in runs:
        for exp_key, filt_dict in filt_dicts.items():
            try:
                bool_list = [run.config[k] == v for k, v in filt_dict.items()]
            except:
                next
            if (all(bool_list) is True) & ("eval_res_table" in run.summary):
                # print(run)
                run_dict_eval[exp_key].append(run)
            elif (all(bool_list) is True) & ("eval_res_table" not in run.summary):
                run_dict_train[exp_key].append(run)
    print(run_dict_eval)
    # get evaluation data
    df_merged = pd.DataFrame()
    for exp, runs in run_dict_eval.items():
        for run in runs:
            fl_list = api.run(run.entity + '/' + run.project + '/' + run.id).files()
            config = api.run(run.entity + '/' + run.project + '/' + run.id).config
            exp_alias_lst.extend(config['exp_alias'])
            for f in fl_list:
                if ("summary" not in f.name) & ("table" in f.name):
                    fl_path = f"./wandb_data/{f.name}"
                    print(fl_path)
                    if os.path.isfile(fl_path):
                        df = pd.read_json(fl_path, orient="split")
                    else:
                        print("download")
                        f.download(root="./wandb_data", replace=True)
                        df = pd.read_json(fl_path, orient="split")
                    df["random_seed"] = [config["rand_seed"]] * df.shape[0]
                    df["experiment_name"] = [exp] * df.shape[0]
                    df_merged = pd.concat([df_merged, df], axis=0)
    # select best eval run using mean on seen jssp instances
    df_filt = df_merged[df_merged.seen_in_training == "seen"]
    if df_filt.shape[0] == 0:  # l2d case with no seen jsps
        df_filt = df_merged[df_merged.seen_in_training == "unseen"]
    df_mean_per_group = df_filt.groupby(
        ["experiment_name", "training_steps_elapsed", "random_seed"]).optimality_gap.mean().reset_index()
    indices = df_mean_per_group.groupby(["experiment_name", "random_seed"]).optimality_gap.idxmin()
    df_best_run_per_exp = df_mean_per_group.loc[indices]
    # get agents
    agent_lst = []
    merged_fl_list = []
    for exp, runs in run_dict_train.items():
        for run in runs:
            fl_list = api.run(run.entity + '/' + run.project + '/' + run.id).files()
            merged_fl_list.extend(fl_list)
            # config = api.run(run.entity + '/' + run.project + '/' + run.id).config
    agent_path_lst = []
    for f in merged_fl_list:
        mask_tr_steps_and_rs = any(
            (f"_{rw[1].training_steps_elapsed}_steps" in f.name) & (f"rs-{rw[1].random_seed}_" in f.name) for rw in
            df_best_run_per_exp.iterrows())
        if ("agents" in f.name) & mask_tr_steps_and_rs:
            f.download(root="./wandb_data", replace=True)
            fl_path = f"./wandb_data/{f.name}"
            print(fl_path)
            agent_path_lst.append(fl_path)
    return agent_path_lst, exp_alias_lst




if __name__ == "__main__":
    run_config_file_name = "enhanced_baseline_scalability_6x6x6_2_15x15x15/r2_70_makespan_6x6x6_2_15x15x15_900jsp_eps02_shuffle_explr_incent_01_6x6x6_0i_eval_config.yml"
    run_config_file_path = Path.cwd() / Path('Data/run_config_files/{}'.format(run_config_file_name))
    run_config_pars = get_config_pars(run_config_file_path)

    print(f'RUNNING: {run_config_pars["rl_framework_tag"]}')

    # get jsp task for evaluation
    jsp_file_path = Path.cwd() / Path(run_config_pars['jsp_path']) / Path(run_config_pars['jsp_size'])
    jsp_eval_lst = collect_transform_jsp_tasks(jsp_ind_start=run_config_pars['jsp_ind_start_eval'],
                                               jsp_ind_end=run_config_pars['jsp_ind_end_eval'],
                                               read_path=jsp_file_path, **run_config_pars)

    # train and evaluate
    date_stamp = 'na'
    wandb_project = 'validate_rl4jsp_submission'
    wandb_entity = os.environ.get('WANDB_ENTITY')

    filt_dicts = {
        run_config_pars["experiment_id"]:
            {"experiment_id": run_config_pars["experiment_id"],
             # "datestamp_run": date_stamp
             },
    }
    # get related runs
    agent_path_lst, exp_alias_lst = get_agents_from_wandb(filt_dicts, wandb_project, wandb_entity)
    print('Agents extraction finished: ', agent_path_lst)
    experiment_eval_pretrained_agents(agent_paths=agent_path_lst, exp_alias_lst=exp_alias_lst,
                                      jsp_eval_lst=jsp_eval_lst, wandb_logs=True, wandb_entity=wandb_entity,
                                      wandb_project=wandb_project,  **run_config_pars)


    # def experiment_eval_pretrained_agents(agent_paths, jsp_eval_lst, wandb_logs, date_stamp, wandb_entity, exp_alias="",
    #                                       wandb_project="validate_rl4jsp_submission",
    #                                       **kwargs):
