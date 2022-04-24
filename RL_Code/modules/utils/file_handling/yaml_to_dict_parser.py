import collections

import yaml


def flatten_dict(d, parent_key='', sep='|'):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping) & ('env_obs_dict' not in new_key) & (
                'wrappers_lst' not in new_key):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, safe_str_to_number(v)))
    return dict(items)


def safe_str_to_number(x):
    if type(x) == str:
        if x == 'None':
            return None
        try:
            x = float(x)
            if (x).is_integer():
                x = int(x)
        except ValueError:
            return x
    else:
        if type(x) == float and (x).is_integer():
            x = int(x)
    return x


def get_config_pars(conf_file):
    # read yaml file into dict
    with open(conf_file, "r") as cp:
        config_par_dict = yaml.safe_load(cp)
    # validate metadata
    experiment_id = config_par_dict['metadata']['experiment_id']
    if experiment_id not in str(conf_file):
        raise ValueError(f'execution cancelled, experiment id {experiment_id} in the run config file {conf_file} '
                         'is not equal to the one written in the filename, or filename is inconsistent')
    config_par_dict = flatten_dict(config_par_dict)
    return config_par_dict
