import re
from pathlib import Path


def training_steps_from_agent_filename(rnd_seed, agent_path):
    return int(re.search(f"-rs-{rnd_seed}_(.*?)_steps", agent_path.name).group(1))


def random_seed_from_agent_filename(agent_path):
    # strt = log_path.find('-rs-') + 4
    # rand_seed = log_path[strt:][:log_path[strt:].find('-Params_R')]
    agent_path = Path(agent_path)
    return int(re.search(f"-rs-(.*?)_(.*?)_steps", agent_path.name).group(1))


def random_seed_from_log_file(log_path):
    # strt = log_path.find('-rs-') + 4
    # rand_seed = log_path[strt:][:log_path[strt:].find('-Params_R')]
    return int(re.search(f"-rs-(.*?)-Params_Reward", log_path.name).group(1))


def viz_filename_prefix_from_log_file(log_path):
    return re.search(f"(.*?)-rs-", log_path.name).group(1)
