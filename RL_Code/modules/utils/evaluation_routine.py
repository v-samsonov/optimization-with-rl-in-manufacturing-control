import wandb

from RL_Code.modules.env.environment_factory import EnvironmentConstructor
from RL_Code.modules.utils.evaluation_routine_constructor import pick_eval_routine
from RL_Code.modules.utils.file_handling.extract_metadata_from_filenames import training_steps_from_agent_filename


def load_and_evaluate_agent(framework, rl_framework_tag, rl_algorithm_tag, agent_path, jsp_eval_lst,
                            rnd_seed, evaluation_seed, log_path, log_name_suffix, wandb_logs, jsp_ind_start_trn,
                            jsp_ind_end_trn, date_stamp='na', **kwargs):
    # load an agent for evaluation
    rl_agent_eval = framework.agent_manager_factory().load(rl_framework_tag=rl_framework_tag,
                                                           rl_algorithm_tag=rl_algorithm_tag,
                                                           agent_path=agent_path, **kwargs)
    # find for how many steps the given saved agent was trained
    training_steps = training_steps_from_agent_filename(rnd_seed, agent_path)

    if wandb_logs:
        wandb.log({'training_steps_elapsed': training_steps}, commit=False)

    # Build environment for evaluation
    log_name_suffix = log_name_suffix + f'_eval_trstp_{training_steps}'
    eval_env = EnvironmentConstructor().build(jsp_lst=jsp_eval_lst,
                                              rand_seed=rnd_seed, log_path=log_path,
                                              log_name_suffix=log_name_suffix, wandb_logs=wandb_logs, eval_flag=True,
                                              **kwargs)
    eval_env.set_monitor_mode('evaluation')
    eval_process = pick_eval_routine(rl_framework_tag)

    eval_res_list = eval_process(framework=framework, rl_agent_eval=rl_agent_eval, eval_env=eval_env,
                                 jsp_eval_lst=jsp_eval_lst, evaluation_seed=evaluation_seed,
                                 jsp_ind_start_trn=jsp_ind_start_trn, jsp_ind_end_trn=jsp_ind_end_trn,
                                 training_steps=training_steps, **kwargs)
    return eval_res_list
