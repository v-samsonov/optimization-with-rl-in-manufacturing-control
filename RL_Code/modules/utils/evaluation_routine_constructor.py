import time

import numpy as np


def pick_eval_routine(rl_framework_tag):
    if rl_framework_tag == 'stable_baselines3' or rl_framework_tag == 'stable_baselines2' or \
            rl_framework_tag == 'stable_baselines2_action_masking':
        return validation_sb
    elif rl_framework_tag == 'l2d':
        return validate_l2d
    else:
        raise ValueError(f'no evaluation routing is defined for {rl_framework_tag} framework')


def validation_sb(framework, rl_agent_eval, eval_env, jsp_eval_lst, evaluation_seed, jsp_ind_start_trn, jsp_ind_end_trn,
                  training_steps, device=None, graph_pool_type=None, **kwargs):
    eval_res_list = []
    for jsp_ind in range(len(jsp_eval_lst)):
        eval_env.seed(evaluation_seed)
        jsp_inst = jsp_eval_lst[jsp_ind]
        jsp_index = int(jsp_inst["jssp_identification"][
                        jsp_inst["jssp_identification"].find('_') + 1:jsp_inst["jssp_identification"].rfind('_')])
        if (jsp_index >= jsp_ind_start_trn) & (jsp_index <= jsp_ind_end_trn):
            seen_flag = 'seen'
        else:
            seen_flag = 'unseen'
        # obs, info['action_mask'] = eval_env.reset(jsp_ind=jsp_ind), None
        obs, info = eval_env.reset(jsp_ind=jsp_ind), {'action_mask': None}
        init_time = time.time()
        while True:
            action = framework.agent_manager_factory().step(rl_agent_eval, obs, info['action_mask'])
            obs, reward, done, info = eval_env.step(action)
            if done:
                makespan = round(info['makespan'], 3)
                solution_time = time.time() - init_time
                break
        eval_res_dict = {}
        optim_gap = round(100 * (makespan - jsp_inst["optimal_time"]) / jsp_inst['optimal_time'], 1)
        eval_res_dict['makespan'], eval_res_dict['jsp_ident'], eval_res_dict['jsp_index'], \
        eval_res_dict['optimal_time'], eval_res_dict['optimality_gap'], eval_res_dict['solution_time'], \
        eval_res_dict['seen_in_training'], \
        eval_res_dict['training_steps_elapsed'] = makespan, jsp_inst["jssp_identification"], jsp_index, \
                                                  jsp_inst["optimal_time"], optim_gap, solution_time, seen_flag, \
                                                  training_steps
        eval_res_list.append(eval_res_dict)
    return eval_res_list


def validate_l2d(framework, rl_agent_eval, eval_env, jsp_eval_lst, evaluation_seed, jsp_ind_start_trn, jsp_ind_end_trn,
                 training_steps, device, graph_pool_type, n_j, n_m, **kwargs):
    import torch
    from RL_Code.modules.agents.l2d_agent_extensions.l2d.mb_agg import g_pool_cal
    from RL_Code.modules.agents.l2d_agent_extensions.l2d.agent_utils import greedy_select_action
    model = rl_agent_eval.policy
    device = torch.device(device)
    g_pool_step = g_pool_cal(graph_pool_type=graph_pool_type,
                             batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
                             n_nodes=n_j * n_m,
                             device=device)

    env_has_obs_manager = hasattr(eval_env, 'obs_manager')

    eval_res_list = []
    for jsp_ind in range(len(jsp_eval_lst)):
        jsp_inst = jsp_eval_lst[jsp_ind]
        jsp_index = jsp_ind
        if (jsp_index >= jsp_ind_start_trn) & (jsp_index <= jsp_ind_end_trn):
            seen_flag = 'seen'
        else:
            seen_flag = 'unseen'
        state = eval_env.reset(jsp_ind=jsp_ind)
        adj, fea, candidate, mask = state['adj'], state['fea'], state['omega'], state['mask']
        if env_has_obs_manager:
            rewards = - eval_env.obs_manager.initQuality
        else:
            rewards = - eval_env.initQuality
        init_time = time.time()
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = model(x=fea_tensor,
                              graph_pool=g_pool_step,
                              padded_nei=None,
                              adj=adj_tensor,
                              candidate=candidate_tensor.unsqueeze(0),
                              mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, candidate)
            state, reward, done, info = eval_env.step(action.item())
            adj, fea, candidate, mask = state['adj'], state['fea'], state['omega'], state['mask']
            rewards += reward
            if done:
                solution_time = time.time() - init_time
                if env_has_obs_manager:
                    makespan = eval_env.obs_manager.max_endTime
                else:
                    makespan = eval_env.max_endTime

                break
        eval_res_dict = {}
        optim_gap = round(100 * (makespan - jsp_inst["optimal_time"]) / jsp_inst['optimal_time'], 1)
        eval_res_dict['makespan'], eval_res_dict['jsp_ident'], eval_res_dict['jsp_index'], \
        eval_res_dict['optimal_time'], eval_res_dict['optimality_gap'], eval_res_dict['solution_time'], \
        eval_res_dict['seen_in_training'], \
        eval_res_dict['training_steps_elapsed'] = makespan, jsp_inst["jssp_identification"], jsp_ind, \
                                                  jsp_inst["optimal_time"], optim_gap, solution_time, seen_flag, \
                                                  training_steps
        eval_res_list.append(eval_res_dict)
    return eval_res_list
