import time

import tensorflow as tf
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.common.policies import register_policy
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.deepq.policies import FeedForwardPolicy
from RL_Code.modules.agents.stable_baselines2_action_filtering.sb2_mod_code.action_masking.sac.policies import FeedForwardPolicy as SACPolicy

POLICY_OBJS = {
    'sac': SACPolicy,
    'dqn': FeedForwardPolicy
}


def assemble_policy_kwargs(rl_algorithm_tag, feature_extraction='mlp', reg_weight=None, dueling=None, layers=[64, 64],
                           act_fnc='tf.nn.relu', layer_norm=True, **kwargs):
    if act_fnc == 'tf.nn.relu':
        act_fnc_object = tf.nn.relu
    elif act_fnc == 'tf.nn.leaky_relu':
        act_fnc_object = tf.nn.leaky_relu
    else:
        raise ValueError(f'{act_fnc} is an unknown activation function')
    policy_kwargs = {'feature_extraction': feature_extraction, 'reg_weight': reg_weight, 'dueling': dueling,
                     'layers': layers, 'act_fun': act_fnc_object, 'layer_norm': layer_norm}
    return policy_kwargs


def build_policy(rl_algorithm_tag, feature_extraction, reg_weight, dueling, layers, act_fun, layer_norm, *args,
                 **kwargs, ):
    class CustomPolicy(POLICY_OBJS[rl_algorithm_tag]):
        def __init__(self, *args, **kwargs):
            if rl_algorithm_tag == 'sac':
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   feature_extraction=feature_extraction,
                                                   reg_weight=reg_weight,
                                                   layers=layers,
                                                   act_fun=act_fun,
                                                   layer_norm=layer_norm,
                                                   )
            else:
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   feature_extraction=feature_extraction,
                                                   dueling=dueling,
                                                   layers=layers,
                                                   act_fun=act_fun,
                                                   layer_norm=layer_norm,
                                                   )

    policy_id = hash(
        time.time())  # to avoid same names for the SAC policy in case of parallel execution of multiple exps
    register_policy(f'CustomPolicy_{policy_id}', CustomPolicy)
    return f'CustomPolicy_{policy_id}'
