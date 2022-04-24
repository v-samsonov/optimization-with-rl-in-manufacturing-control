import json
from pathlib import Path

import numpy as np

from RL_Code.modules.utils.jsp_handling.jsp_format_converters import convert_l2d_to_simpy_format_lst, \
    convert_l2d_to_legacy_format_jsp_discr_lst


def find_jsp_tasks_in_dir(jsp_ind_start, jsp_ind_end, jsp_path, verbose=0, **kwargs):
    jsp_path = Path(jsp_path)
    files_list = []
    for i in range(jsp_ind_start, jsp_ind_end + 1):
        files_list += [e for e in jsp_path.iterdir() if
                       (e.is_file()) & (f"_{i}_" in e.as_posix())]
    if len(files_list) != (jsp_ind_end - jsp_ind_start + 1):
        raise ValueError(
            f"{len(files_list)} files were found, expected number is {jsp_ind_end - jsp_ind_start + 1}, check the jsp file index range"
            f"in dir {jsp_path} following files were found \n "
            f"files: {files_list}")
    if verbose > 0:
        print(f"found jsp json files: {len(files_list)}")
        print(files_list)
    return files_list


def collect_transform_jsp_tasks(jsp_ind_start, jsp_ind_end, read_path, env_tag, **kwargs):
    jsp_list = []
    read_files = find_jsp_tasks_in_dir(jsp_ind_start, jsp_ind_end, read_path)
    # read json
    for read_file in read_files:
        with open(Path(read_path) / Path(read_file), 'r') as f:
            jsp_dict = json.load(f)
            jsp_dict['jssp_instance']['durations'] = np.array(jsp_dict['jssp_instance']['durations'])
            jsp_dict['jssp_instance']['machines'] = np.array(jsp_dict['jssp_instance']['machines'])
        jsp_list += [jsp_dict]

    # transfrom jsp format
    if env_tag == 'jsp_simpy':
        jsp_trn_lst = convert_l2d_to_simpy_format_lst(jsp_list)
    elif env_tag == 'jsp':
        jsp_trn_lst = convert_l2d_to_legacy_format_jsp_discr_lst(jsp_list)
    elif env_tag == 'jsp_l2d':
        pass
    else:
        raise NotImplementedError

    return jsp_list


if __name__ == "__main__":
    # test collection jsp tasks
    read_path = Path.cwd().parents[3] / Path("Data/jsp_instances/6x6x6")
    jsp_size = '6x6'
    jsp_ind_start = 0
    jsp_ind_end = 2
    jsp_list = collect_transform_jsp_tasks(jsp_size=jsp_size, jsp_ind_start=jsp_ind_start, jsp_ind_end=jsp_ind_end,
                                           read_path=read_path, env_tag='jsp_simpy')

    jsp_list
