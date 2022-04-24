import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import wandb

from RL_Code.modules.utils.evaluation_routine import load_and_evaluate_agent
from RL_Code.modules.utils.file_handling.build_log_namings import init_log_path, init_agent_save_path, init_tb_log_path
from RL_Code.modules.utils.file_handling.find_files_in_dir import fetch_agents_locations
from RL_Code.modules.utils.file_handling.extract_metadata_from_filenames import random_seed_from_agent_filename
from RL_Code.modules.utils.training_routine import train

# fix wandb config path to avoid access permission problems in docker
os.environ['WANDB_SILENT'] = 'true'
WANDB_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "wandb")
print('init config dir', WANDB_CONFIG_DIR)
if (WANDB_CONFIG_DIR == '/.config') or (WANDB_CONFIG_DIR == '/.config/wandb'):
    os.environ['WANDB_CONFIG_DIR'] = Path.cwd().as_posix()
    print('changed config dir', WANDB_CONFIG_DIR)


def experiment_train_eval_from_scratch_with_checkpoints(training_steps, jsp_trn_lst, jsp_eval_lst, rand_seed,
                                                        wandb_logs, date_stamp, wandb_entity, exp_alias="",
                                                        wandb_project="validate_rl4jsp_submission",
                                                        save_models_to_wandb=False,
                                                        **kwargs):
    eval_res_list = []

    if kwargs['rl_framework_tag'] == 'stable_baselines2':
        from RL_Code.modules.agents.stable_baselines2_extensions.stable_baselines2_factory import \
            StableBaselines2Constructor
        framework = StableBaselines2Constructor()
    elif kwargs['rl_framework_tag'] == 'stable_baselines2_action_masking':
        from RL_Code.modules.agents.stable_baselines2_action_filtering.stable_baselines2_factory import \
            StableBaselines2Constructor
        framework = StableBaselines2Constructor()
    elif kwargs['rl_framework_tag'] == 'stable_baselines3':
        from RL_Code.modules.agents.stable_baselines3_extensions.stable_baselines3_factory import \
            StableBaselines3Constructor
        framework = StableBaselines3Constructor()
    elif kwargs['rl_framework_tag'] == 'l2d':
        from RL_Code.modules.agents.l2d_agent_extensions.l2d_factory import L2DConstructor
        framework = L2DConstructor()
    else:
        raise ValueError(f'{kwargs["rl_framework_tag"]} is an unknown framework')

    # define log paths
    log_path, log_name_suffix = init_log_path(rnd_seed=rand_seed, date_stamp=date_stamp, **kwargs)
    agent_path, agent_name_prefix = init_agent_save_path(rnd_seed=rand_seed, date_stamp=date_stamp, **kwargs)
    tensorboard_path, tb_log_name = init_tb_log_path(date_stamp=date_stamp, rnd_seed=rand_seed, **kwargs)
    wandb_logs_path = log_path

    if wandb_logs:
        wandb.init(project=wandb_project, group=kwargs['experiment_id'] + '_' + date_stamp, config=kwargs,
                   entity=wandb_entity, dir=wandb_logs_path, job_type='training', reinit=True)
        print("wand_dir", str(wandb.run.dir))
        wandb.config.update({"training_steps": training_steps, "rand_seed": rand_seed, 'datestamp_run': date_stamp})
        wandb.config.update({"exp_alias": exp_alias})
        if save_models_to_wandb:
            agent_path = Path(wandb.run.dir) / Path('agents')
            agent_path.mkdir(exist_ok=True)

    # train agents
    print(
        f'training started on: {date_stamp}')  # .format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print(f'\n \t TRAINING THE AGENT FOR {training_steps} STEPS')
    train(framework=framework, training_steps=training_steps, random_seed=rand_seed,
          agent_path=agent_path, agent_name=agent_name_prefix,
          jsp_trn_lst=jsp_trn_lst, date_stamp=date_stamp, log_path=log_path,
          log_name_suffix=log_name_suffix, tensorboard_path=tensorboard_path,
          tb_log_name=tb_log_name, wandb_logs=wandb_logs, **kwargs)

    # conduct evaluation on saved agents
    agent_files = fetch_agents_locations(agent_path=agent_path, date_stamp=date_stamp, rand_seed=rand_seed, **kwargs)
    if wandb_logs:
        wandb.init(project=wandb_project, group=kwargs['experiment_id'] + '_' + date_stamp, config=kwargs,
                   entity=wandb_entity,
                   dir=wandb_logs_path, job_type='eval', reinit=True)
        wandb.config.update({"training_steps": None, "rand_seed": rand_seed, 'datestamp_run': date_stamp})
        wandb.config.update({"exp_alias": exp_alias})

    for agent_path in agent_files:
        print(f'evaluating agent {agent_path}')
        eval_res_dict = load_and_evaluate_agent(framework=framework, agent_path=agent_path, jsp_eval_lst=jsp_eval_lst,
                                                rnd_seed=rand_seed, log_path=log_path, date_stamp=date_stamp,
                                                log_name_suffix=log_name_suffix, wandb_logs=wandb_logs, **kwargs)
        eval_res_list.extend(eval_res_dict)
    if wandb_logs:
        cumulative_eval_res_df = pd.DataFrame(eval_res_list)
        # log boxplot table for seen and unseen jsp instances
        data_ = wandb.Table(dataframe=cumulative_eval_res_df)
        wandb.log({'eval_res_table': data_})

        # log line plot tables for seen and unseen jsp instances
        cummulative_eval_res_df_group = cumulative_eval_res_df.groupby(['training_steps_elapsed', 'seen_in_training'])
        df_median = cummulative_eval_res_df_group.optimality_gap.median()
        df_mean = cummulative_eval_res_df_group.optimality_gap.mean()
        df_min = cummulative_eval_res_df_group.optimality_gap.min()
        df_max = cummulative_eval_res_df_group.optimality_gap.max()
        summary_cummulative_eval_res_df = pd.concat([df_median, df_mean, df_min, df_max], axis=1)
        summary_cummulative_eval_res_df.columns = ['optimality_gap_median', 'optimality_gap_mean',
                                                   'optimality_gap_min', 'optimality_gap_max']
        summary_cummulative_eval_res_df.reset_index(inplace=True)
        data_ = wandb.Table(dataframe=summary_cummulative_eval_res_df)
        wandb.log({'summary_eval_res_table': data_})

        # log box plot
        # select best eval run using mean on seen jsp instances
        df_filt_seen = cumulative_eval_res_df[cumulative_eval_res_df.seen_in_training == "seen"]
        df_mean_seen = df_filt_seen.groupby(
            ["training_steps_elapsed"]).optimality_gap.mean().reset_index()
        try:
            tr_steps_with_br_seen = df_mean_seen.loc[df_mean_seen.optimality_gap.idxmin()].training_steps_elapsed
        except:
            # l2d approach from Zhnag et al. has no seen parts during training. Part are generated ont the fly and seen only once.
            # Here we choose to select on the unseen parts, which makes the comparison to our approach even more conservative
            df_filt_unseen = cumulative_eval_res_df[cumulative_eval_res_df.seen_in_training == "unseen"]
            df_mean_unseen = df_filt_unseen.groupby(
                ["training_steps_elapsed"]).optimality_gap.mean().reset_index()
            tr_steps_with_br_seen = df_mean_unseen.loc[df_mean_unseen.optimality_gap.idxmin()].training_steps_elapsed
        cumulative_eval_res_df_filt = cumulative_eval_res_df[cumulative_eval_res_df.training_steps_elapsed==tr_steps_with_br_seen]

        fig = px.box(cumulative_eval_res_df_filt, x="training_steps_elapsed", y="optimality_gap", color="seen_in_training",
                     title=f'Best_Optimality_Gap_{kwargs["experiment_id"]}_{rand_seed}_{date_stamp}')
        wandb.log({'eval_run_box_plot': fig})

    return {'date_stamp': date_stamp, 'log_path': log_path}



def experiment_eval_pretrained_agents(agent_paths, exp_alias_lst, jsp_eval_lst, wandb_logs, date_stamp, wandb_entity,
                                      wandb_project="validate_rl4jsp_submission", **kwargs):

    if kwargs['rl_framework_tag'] == 'stable_baselines2':
        from RL_Code.modules.agents.stable_baselines2_extensions.stable_baselines2_factory import \
            StableBaselines2Constructor
        framework = StableBaselines2Constructor()
    elif kwargs['rl_framework_tag'] == 'stable_baselines2_action_masking':
        from RL_Code.modules.agents.stable_baselines2_action_filtering.stable_baselines2_factory import \
            StableBaselines2Constructor
        framework = StableBaselines2Constructor()
    elif kwargs['rl_framework_tag'] == 'stable_baselines3':
        from RL_Code.modules.agents.stable_baselines3_extensions.stable_baselines3_factory import \
            StableBaselines3Constructor
        framework = StableBaselines3Constructor()
    elif kwargs['rl_framework_tag'] == 'l2d':
        from RL_Code.modules.agents.l2d_agent_extensions.l2d_factory import L2DConstructor
        framework = L2DConstructor()
    else:
        raise ValueError(f'{kwargs["rl_framework_tag"]} is an unknown framework')

    counter = 0
    for agent_path in agent_paths:
        exp_alias = exp_alias_lst[counter]
        counter +=1
        eval_res_list = []
        # conduct evaluation on saved agents
        agent_path = Path(agent_path)
        rand_seed = random_seed_from_agent_filename(agent_path)

        if wandb_logs:
            wandb.init(project=wandb_project, group=kwargs['experiment_id'] + '_' + date_stamp, config=kwargs,
                       entity=wandb_entity, # dir=wandb_logs_path,
                       job_type='eval_on_target_size', reinit=True)
            wandb.config.update({"training_steps": None, "rand_seed": rand_seed, 'datestamp_run': date_stamp})
            wandb.config.update({"exp_alias": exp_alias})


        print(f'evaluating agent {agent_path}')
        eval_res_dict = load_and_evaluate_agent(framework=framework, agent_path=agent_path, jsp_eval_lst=jsp_eval_lst,
                                                rnd_seed=rand_seed, log_path=Path.cwd(), date_stamp=date_stamp,
                                                log_name_suffix='env_logs', wandb_logs=wandb_logs, **kwargs)
        eval_res_list.extend(eval_res_dict)

        if wandb_logs:
            cumulative_eval_res_df = pd.DataFrame(eval_res_list)
            # log boxplot table for seen and unseen jsp instances
            data_ = wandb.Table(dataframe=cumulative_eval_res_df)
            wandb.log({'eval_res_table': data_})

            # log line plot tables for seen and unseen jsp instances
            cummulative_eval_res_df_group = cumulative_eval_res_df.groupby(['training_steps_elapsed', 'seen_in_training'])
            df_median = cummulative_eval_res_df_group.optimality_gap.median()
            df_mean = cummulative_eval_res_df_group.optimality_gap.mean()
            df_min = cummulative_eval_res_df_group.optimality_gap.min()
            df_max = cummulative_eval_res_df_group.optimality_gap.max()
            summary_cummulative_eval_res_df = pd.concat([df_median, df_mean, df_min, df_max], axis=1)
            summary_cummulative_eval_res_df.columns = ['optimality_gap_median', 'optimality_gap_mean',
                                                       'optimality_gap_min', 'optimality_gap_max']
            summary_cummulative_eval_res_df.reset_index(inplace=True)
            data_ = wandb.Table(dataframe=summary_cummulative_eval_res_df)
            wandb.log({'summary_eval_res_table': data_})

            # log box plot
            # select best eval run using mean on seen jsp instances
            df_filt_seen = cumulative_eval_res_df[cumulative_eval_res_df.seen_in_training == "seen"]
            df_mean_seen = df_filt_seen.groupby(
                ["training_steps_elapsed"]).optimality_gap.mean().reset_index()
            tr_steps_with_br_seen = df_mean_seen.loc[df_mean_seen.optimality_gap.idxmin()].training_steps_elapsed
            cumulative_eval_res_df_filt = cumulative_eval_res_df[
                cumulative_eval_res_df.training_steps_elapsed == tr_steps_with_br_seen]

            fig = px.box(cumulative_eval_res_df_filt, x="training_steps_elapsed", y="optimality_gap",
                         color="seen_in_training",
                         title=f'Best_Optimality_Gap_{kwargs["experiment_id"]}_{rand_seed}_{date_stamp}')
            wandb.log({'eval_run_box_plot': fig})

    return True