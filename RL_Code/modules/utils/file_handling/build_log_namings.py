import os
import warnings
from pathlib import Path

from RL_Code.modules.utils.file_handling.build_hardware_id import init_hardware_id


def init_log_path(log_folder_prefix, experiment_id, rnd_seed, date_stamp, **kwargs):
    hw_id = init_hardware_id()
    log_name_suffix = f'exp_id-{experiment_id}-hw_id-{hw_id}-{date_stamp}-rs-{rnd_seed}'
    log_folder = (Path(os.path.dirname(__file__)).parents[3] / Path('Results') /
                  Path(log_folder_prefix) / Path('logs') / Path(f'exp_id-{experiment_id}-hw_id-{hw_id}'))
    if len(log_folder.as_posix()) > 260:
        warnings.warn(
            f"Logging path {log_folder} \n \t exceeds max path length of 260 symbols, logging to the local drive might fail")
    return log_folder, log_name_suffix


def init_agent_save_path(log_folder_prefix, experiment_id, rnd_seed, date_stamp, **kwargs):
    hw_id = init_hardware_id()
    agent_path = (Path(os.path.dirname(__file__)).parents[3] / Path('Results') /
                  Path(log_folder_prefix) / Path('agents'))  # .as_posix()
    print("agent_path", agent_path)
    agent_path.mkdir(parents=True, exist_ok=True)
    agent_name_prefix = f'exp_id-{experiment_id}-hw_id-{hw_id}-{date_stamp}-rs-{rnd_seed}'
    return agent_path, agent_name_prefix


def init_visualization_path(log_folder_prefix, **kwargs):
    viz_path = (Path(os.path.dirname(__file__)).parents[3] / Path('Results') /
                Path(log_folder_prefix) / Path('viz_results'))  # .as_posix()
    viz_path.mkdir(parents=True, exist_ok=True)
    return viz_path


def init_tb_log_path(log_folder_prefix, tensorboard_log_flag, experiment_id, rl_algorithm_tag, date_stamp, rnd_seed,
                     **kwargs):
    tb_log_path = None
    tb_filename = None
    if tensorboard_log_flag:
        tb_log_path = (Path(os.path.dirname(__file__)).parents[3] / Path('Results') /
                       Path(log_folder_prefix) / Path('tb_logs'))  # .as_posix()
        tb_filename = f'{experiment_id}-{rnd_seed}-{date_stamp}-{rl_algorithm_tag}'
    return tb_log_path, tb_filename
