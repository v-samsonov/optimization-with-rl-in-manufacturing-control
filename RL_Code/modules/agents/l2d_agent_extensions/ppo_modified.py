from RL_Code.modules.agents.l2d_agent_extensions.l2d.PPO_jssp_multiInstances import PPO
import torch

class PPOModified(PPO):
    def __init__(self, lr,
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
                seed=1234,
                 **kwargs):
        import numpy as np
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        PPO.__init__(self,
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
                 **kwargs)

