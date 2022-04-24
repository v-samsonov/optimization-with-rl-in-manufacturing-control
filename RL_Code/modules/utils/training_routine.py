from RL_Code.modules.env.environment_factory import EnvironmentConstructor


def train(framework, random_seed, agent_path, agent_name, jsp_trn_lst, tensorboard_path, log_path, log_name_suffix,
          wandb_logs, tb_log_name=None, **kwargs):
    """
    Main Function, builds an rl agent, eval_env, training routing + execution of training and regular
        saving of the agent during the training
    :env: (OpenAI gym object) eval_env for RL Agent training
    :return: (BaseRLModel object) trained RL Agent
    """

    # Build environment and agent
    log_name_suffix = log_name_suffix + f'_train'
    environment = EnvironmentConstructor().build(jsp_lst=jsp_trn_lst,
                                                 rand_seed=random_seed, log_path=log_path,
                                                 log_name_suffix=log_name_suffix, wandb_logs=wandb_logs, **kwargs)

    rl_agent = framework.agent_factory().build(environment=environment, rnd_seed=random_seed,
                                               tebsorboard_path=tensorboard_path, **kwargs)
    # Train the agent
    trained_rl_agent, environment = framework.agent_manager_factory().train(rl_agent=rl_agent,
                                                                            jsp_lst=jsp_trn_lst,
                                                                            environment=environment,
                                                                            tb_log_name=tb_log_name,
                                                                            agent_path=agent_path,
                                                                            agent_name=agent_name,
                                                                            rnd_seed=random_seed, wandb_logs=wandb_logs,
                                                                            **kwargs)
    return trained_rl_agent, environment


def resume_training(framework, environment, agent_path, agent_name, date_stamp, tb_log_name=None,
                    **kwargs):
    # load agent
    rl_agent = framework.agent_manager_factory().load(agent_path=agent_path, **kwargs)
    # train agent
    rl_agent, environment = framework.agent_manager_factory().retrain(environment=environment, rl_agent=rl_agent,
                                                                      agent_path=agent_path, agent_name=agent_name,
                                                                      date_stamp=date_stamp,
                                                                      tb_log_name=tb_log_name, **kwargs)
    # #save agent
    # AgentManager().save(rl_framework_tag=rl_framework_tag, rl_agent=rl_agent, agent_path=agent_path, **kwargs)
    return rl_agent, environment
