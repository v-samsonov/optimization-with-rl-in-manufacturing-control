def fetch_agents_locations(agent_path, date_stamp, experiment_id, rand_seed, **kwargs):
    files_list = [e for e in agent_path.iterdir() if
                  (e.is_file()) & (date_stamp in e.as_posix()) & (experiment_id in e.as_posix()) & (
                          str(rand_seed) in e.as_posix())]
    if len(files_list) == 0:
        raise ValueError(f'no agents for evaluation were found in dir {agent_path}')
    return files_list


def fetch_logs_locations(log_path, date_stamp_lst, experiment_id, rand_seed_lst, **kwargs):
    files_list = []
    for rand_seed in rand_seed_lst:
        for date_stamp in date_stamp_lst:
            files_list.extend([e for e in log_path.iterdir() if
                               (e.is_file()) & (date_stamp in e.as_posix()) & (experiment_id in e.as_posix()) & (
                                       f'-rs-{rand_seed}-' in e.as_posix()) & ('Params_Reward' in e.as_posix())])
    if len(files_list) == 0:
        raise ValueError(f'no logs for visualisation with data stamp {date_stamp} and rnd_seeds {rand_seed_lst} '
                         f'were found in {log_path}')
    return files_list
