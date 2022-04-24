from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.nn as nn
import wandb

from RL_Code.modules.agents.l2d_agent_extensions.l2d.actor_critic import ActorCritic
from RL_Code.modules.agents.l2d_agent_extensions.l2d.agent_utils import eval_actions
from RL_Code.modules.agents.l2d_agent_extensions.l2d.agent_utils import select_action
from RL_Code.modules.agents.l2d_agent_extensions.l2d.mb_agg import *


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device,
                 decay_step_size,
                 decay_ratio,
                 vloss_coef,
                 ploss_coef,
                 entloss_coef,
                 decayflag,
                 torch_seed,
                 np_seed_train,
                 num_envs,
                 graph_pool_type,
                 max_updates,
                 low,
                 high,
                 **kwargs
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = torch.device(device)
        self.n_j = n_j
        self.n_m = n_m
        self.decay_step_size = decay_step_size
        self.decay_ratio = decay_ratio
        self.vloss_coef = vloss_coef
        self.ploss_coef = ploss_coef
        self.entloss_coef = entloss_coef
        self.decayflag = decayflag
        self.torch_seed = torch_seed
        self.np_seed_train = np_seed_train
        self.num_envs = num_envs
        self.graph_pool_type = graph_pool_type
        self.max_updates = max_updates + 1 # +1 to save the last trained model e.g. 100%10->101%10 => 10th model is saved
        self.low = low
        self.high = high

        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=self.device)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.decay_step_size,
                                                         gamma=self.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def update(self, memories, n_tasks, g_pool):

        vloss_coef = self.vloss_coef
        ploss_coef = self.ploss_coef
        entloss_coef = self.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        # store data for all env
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            # process each env data
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(self.device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(self.device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(self.device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(self.device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(self.device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(self.device).squeeze().detach())

        # get batch argument for net forwarding: mb_g_pool is same for all env
        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(self.device).shape, n_tasks, self.device)

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss
                vloss_sum += v_loss
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        if self.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()

    def save(self, agent_path, agent_name, timesteps):
        torch.save(self.policy.state_dict(), Path(agent_path) / Path(agent_name + '_' + str(timesteps) + '_steps'))

    def load(self, agent_path):
        # ppo.policy.load_state_dict(torch.load(path))
        self.policy.load_state_dict(torch.load(str(agent_path), map_location=torch.device(self.device)))
        return self

    def learn(self, agent_path, agent_name, save_freq, jsp_lst, environment, wandb_logs=False, **kwargs):

        # envs = [SJSSP(jsp_lst=jsp_lst, **kwargs) for _ in range(self.num_envs)]

        # envs = [do_smth(environment, i) for i in range(self.num_envs)]
        envs = environment

        # torch.manual_seed(self.torch_seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(self.torch_seed)
        # np.random.seed(self.np_seed_train)

        memories = [Memory() for _ in range(self.num_envs)]
        g_pool_step = g_pool_cal(graph_pool_type=self.graph_pool_type,
                                 batch_size=torch.Size([1, self.n_j * self.n_m, self.n_j * self.n_m]),
                                 n_nodes=self.n_j * self.n_m,
                                 device=self.device)

        #differences simpy/l2d
        env_has_obs_manager = hasattr(envs[0], 'obs_manager')

        # training loop
        log = []
        for i_update in range(self.max_updates):

            ep_rewards = [0 for _ in range(self.num_envs)]
            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []

            for i, env in enumerate(envs):
                state = env.reset(jsp_ind=None)
                adj, fea, candidate, mask = state['adj'], state['fea'], state['omega'], state['mask']
                # print(f"update {i_update}, env {i}, fea {fea[:3]}")
                # print(f"adj {adj[:3]}")
                adj_envs.append(adj)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
                if env_has_obs_manager:
                    ep_rewards[i] = - env.obs_manager.initQuality
                else:
                    ep_rewards[i] = - env.initQuality
            # rollout the env
            while True:
                fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(self.device) for fea in fea_envs]
                adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(self.device).to_sparse() for adj in adj_envs]
                candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(self.device) for candidate in
                                         candidate_envs]
                mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(self.device) for mask in mask_envs]

                with torch.no_grad():
                    action_envs = []
                    a_idx_envs = []
                    for i in range(self.num_envs):
                        pi, _ = self.policy_old(x=fea_tensor_envs[i],
                                                graph_pool=g_pool_step,
                                                padded_nei=None,
                                                adj=adj_tensor_envs[i],
                                                candidate=candidate_tensor_envs[i].unsqueeze(0),
                                                mask=mask_tensor_envs[i].unsqueeze(0))
                        action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                        action_envs.append(action)
                        a_idx_envs.append(a_idx)

                adj_envs = []
                fea_envs = []
                candidate_envs = []
                mask_envs = []
                # Saving episode data
                for i in range(self.num_envs):
                    memories[i].adj_mb.append(adj_tensor_envs[i])
                    memories[i].fea_mb.append(fea_tensor_envs[i])
                    memories[i].candidate_mb.append(candidate_tensor_envs[i])
                    memories[i].mask_mb.append(mask_tensor_envs[i])
                    memories[i].a_mb.append(a_idx_envs[i])

                    state, reward, done, info = envs[i].step(action_envs[i].item())
                    adj, fea, candidate, mask = state['adj'], state['fea'], state['omega'], state['mask']

                    adj_envs.append(adj)
                    fea_envs.append(fea)
                    candidate_envs.append(candidate)
                    mask_envs.append(mask)
                    ep_rewards[i] += reward
                    memories[i].r_mb.append(reward)
                    memories[i].done_mb.append(done)
                if callable(envs[0].done):
                    if envs[0].done():
                        break
                else:
                    if envs[0].done:
                        break
            for j in range(self.num_envs):
                if env_has_obs_manager:
                    ep_rewards[j] -= envs[j].obs_manager.posRewards
                else:
                    ep_rewards[j] -= envs[j].posRewards


            loss, v_loss = self.update(memories, self.n_j * self.n_m, self.graph_pool_type)
            for memory in memories:
                memory.clear_memory()
            mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
            log.append([i_update, mean_rewards_all_env])
            # if i_update % save_freq == 0:
            #     file_writing_obj = open(
            #         './' + 'log_' + str(self.n_j) + '_' + str(self.n_m) + '_' + str(self.low) + '_' + str(
            #             self.high) + '.txt', 'w')
            #     file_writing_obj.write(str(log))
            if (wandb_logs) & (i_update % 10 == 0):
                wandb.log({'loss': loss, 'v_loss': v_loss})

            # log results
            print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
                i_update, mean_rewards_all_env, v_loss))

            # save
            if i_update % save_freq == 0:
                self.save(agent_path, agent_name, i_update)

        return self
