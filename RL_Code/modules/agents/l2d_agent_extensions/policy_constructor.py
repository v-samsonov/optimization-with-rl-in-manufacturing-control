import torch as th
from stable_baselines3.sac.policies import MlpPolicy as SACPolicy

from stable_baselines3.common.policies import register_policy
import time

POLICY_OBJS = {
    'ppo': SACPolicy,

}

def assemble_policy_kwargs(rl_algorithm_tag, feature_extraction='mlp', reg_weight=None, dueling=None, layers=[64, 64],
                           act_fnc='th.nn.ReLU', layer_norm=True, **kwargs):
    if act_fnc == 'th.nn.ReLU':
        act_fnc_object = th.nn.ReLU
    elif act_fnc == 'th.nn.LeakyReLU':
        act_fnc_object = th.nn.LeakyReLU
    else:
        raise ValueError(f'{act_fnc} is an unknown activation function')
    policy_kwargs = {'feature_extraction': feature_extraction, 'reg_weight': reg_weight, 'dueling':dueling,
                     'layers': layers, 'act_fun': act_fnc_object, 'layer_norm': layer_norm}
    return policy_kwargs



def build_policy(rl_algorithm_tag, feature_extraction, reg_weight, dueling, layers, act_fun, layer_norm, *args, **kwargs, ):
    class CustomPolicy(POLICY_OBJS[rl_algorithm_tag]):
        def __init__(self, *args, **kwargs):
            if rl_algorithm_tag=='ppo':
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                      activation_fn=act_fun
                                                      )
            else:
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                      feature_extraction=feature_extraction,
                                                      dueling=dueling,
                                                      layers=layers,
                                                      act_fun=act_fun,
                                                      layer_norm=layer_norm,
                                                      )
    policy_id = hash(time.time()) # to avoid same names for the SAC policy in case of parallel execution of multiple exps
    register_policy(f'CustomPolicy_{policy_id}', CustomPolicy)
    return f'CustomPolicy_{policy_id}'
