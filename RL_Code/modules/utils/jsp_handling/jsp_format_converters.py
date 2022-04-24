from pathlib import Path

import numpy as np
import pandas as pd


def convert_l2d_to_legacy_format(times, machines, n_ops_per_job, n_jobs, n_resources):
    # transform l2d jsp format into rl4jsp format
    op_index_lst = np.tile(np.arange(0, n_ops_per_job), n_jobs)
    job_index_lst = np.repeat(np.arange(0, n_jobs), n_ops_per_job)
    times = times.flatten()
    machines = machines.flatten()
    j_r_t = pd.DataFrame({"job": job_index_lst, "order": op_index_lst, "machines": machines, "durations": times})
    one_hot_df = pd.DataFrame(np.full((n_ops_per_job * n_jobs, n_resources), 0),
                              columns=[str(i) for i in range(n_resources)])
    jsp_instance_df = pd.concat([j_r_t, one_hot_df], axis=1)
    for i in range(jsp_instance_df.shape[0]):
        machine, duration = jsp_instance_df.loc[i, 'machines'], jsp_instance_df.loc[i, 'durations']
        jsp_instance_df.loc[i, str(machine)] = duration
    jsp_instance_df["state"] = ["todispatch"] * n_ops_per_job * n_jobs
    jsp_instance_df.drop(["machines", "durations"], axis=1, inplace=True)
    jsp_instance_df = jsp_instance_df.sort_values(['job', 'order'])
    return jsp_instance_df


def convert_l2d_to_legacy_format_jsp_discr_lst(jsp_lst):
    # transform list of l2d jsps into a list of rl4jsps
    jsp_lst_trnsf = []
    for jsp_inst in jsp_lst:
        jsp_inst['jssp_instance'] = convert_l2d_to_legacy_format(jsp_inst['jssp_instance']['durations'],
                                                                 jsp_inst['jssp_instance']['machines'],
                                                                 jsp_inst['n_ops_per_job'], jsp_inst['n_jobs'],
                                                                 jsp_inst['n_resources'])
        jsp_lst_trnsf.append(jsp_inst)
    return jsp_lst_trnsf


def convert_l2d_to_simpy_format(times, machines, n_ops_per_job, n_jobs, n_resources):
    # transform l2d jsp format into simpy format
    op_index_lst = np.tile(np.arange(0, n_ops_per_job), n_jobs)
    job_index_lst = np.repeat(np.arange(0, n_jobs), n_ops_per_job)
    times = times.flatten()
    machines = machines.flatten()
    # times_machines = [[t, m] for t, m in zip(times, machines)]
    jsp_instance_df = pd.DataFrame({"job": job_index_lst, "order": op_index_lst,
                                    "machinetype": machines, "opduration": times,
                                    # "opduration_machinetype": times_machines,
                                    })
    # jsp_instance_df["state"] = ["todispatch"] * n_ops_per_job * n_jobs
    jsp_instance_df = jsp_instance_df.sort_values(['job', 'order'])
    jsp_instance_df.rename(columns={"job": "order", "order": "operation"}, inplace=True)
    return jsp_instance_df


def convert_l2d_to_simpy_format_lst(jsp_lst):
    # transform list of l2d jsps into a list of simpy jsps
    jsp_lst_trnsf = []
    for jsp_inst in jsp_lst:
        jsp_inst['jssp_instance'] = convert_l2d_to_simpy_format(jsp_inst['jssp_instance']['durations'],
                                                                jsp_inst['jssp_instance']['machines'],
                                                                jsp_inst['n_ops_per_job'], jsp_inst['n_jobs'],
                                                                jsp_inst['n_resources'])
        jsp_lst_trnsf.append(jsp_inst)
    return jsp_lst_trnsf


def convert_legacy_inst_to_l2d_format(jsp_data):
    jsp_df = pd.DataFrame.from_dict(jsp_data['jssp_instance'], orient='index').sort_values(['job', 'order'])
    dur, mch = convert_legacy_pd_df_to_l2d_format(jsp_df=jsp_df, n_jobs=jsp_data["n_jobs"],
                                                  n_ops_per_job=jsp_data["n_ops_per_job"])
    return dur, mch


def convert_legacy_pd_df_to_l2d_format(jsp_df, n_jobs, n_ops_per_job):
    jsp_df.index = jsp_df.index.astype(int)
    processing_col_lst = []
    jsp_df.rename(columns={"job": "order", "order": "operation"}, inplace=True)
    for col in jsp_df.columns:
        if col.isdigit():  # operation type column
            processing_col_lst.append(col)
            jsp_df.loc[:, str(col)] = jsp_df[str(col)].apply(lambda x: (x, col) if x > 0 else np.nan)

    jsp_df['opduration_machinetype'] = jsp_df.loc[:, processing_col_lst].apply(
        lambda x: [i for i in x if isinstance(i, tuple)][0],
        axis=1)  # collapse sparse dataframe in one column containing tuple (duration, operation_type)
    jsp_df['opduration'] = jsp_df['opduration_machinetype'].apply(lambda x: x[0])  # collapse sparse dataframe in
    # one column containing tuple (duration, operation_type)
    jsp_df['machinetype'] = jsp_df['opduration_machinetype'].apply(lambda x: x[1])
    jsp_df = jsp_df.sort_values(['order', 'operation'])

    dur = []
    mch = []
    for job in range(n_jobs):
        inner_dur = []
        inner_mch = []
        for op in range(n_ops_per_job):
            inner_dur.append(
                jsp_df[jsp_df["order"] == job]["opduration"].loc[jsp_df["operation"] == op].tolist()[0])
            inner_mch.append(
                int(jsp_df[jsp_df["order"] == job]["machinetype"].loc[jsp_df["operation"] == op].tolist()[0]))

        dur.append(inner_dur)
        mch.append(inner_mch)

    return dur, mch


def transform_jsp_to_l2d(jsp_data):
    jsp_instance = jsp_data['jssp_instance']
    # transofrm dataframe for the simpy env
    processing_col_lst = []
    for col in jsp_instance.columns:
        if col.isdigit():  # operation type column
            processing_col_lst.append(col)
            jsp_instance.loc[:, str(col)] = jsp_instance[str(col)].apply(lambda x: (x, col) if x > 0 else np.nan)

    jsp_instance['opduration_machinetype'] = jsp_instance.loc[:, processing_col_lst].apply(
        lambda x: [i for i in x if isinstance(i, tuple)][0],
        axis=1)  # collapse sparse dataframe in one column containing tuple (duration, operation_type)
    jsp_instance['opduration'] = jsp_instance['opduration_machinetype'].apply(
        lambda x: x[0])  # collapse sparse dataframe in
    # one column containing tuple (duration, operation_type)
    jsp_instance['machinetype'] = jsp_instance['opduration_machinetype'].apply(lambda x: x[1])
    jsp_instance = jsp_instance.sort_values(['order', 'operation'])

    dur = []
    mch = []
    for job in range(jsp_data["n_jobs"]):
        inner_dur = []
        inner_mch = []
        for op in range(jsp_data["n_ops_per_job"]):
            inner_dur.append(jsp_instance[jsp_instance["order"] == job]["opduration"].loc[
                                 jsp_instance["operation"] == op].tolist()[0])
            inner_mch.append(
                int(jsp_instance[jsp_instance["order"] == job]["machinetype"].loc[
                        jsp_instance["operation"] == op].tolist()[0]))

        dur.append(inner_dur)
        mch.append(inner_mch)

    jsp_instance_transformed = [np.array(dur), np.array(mch)]
    jsp_data['jssp_instance'] = jsp_instance_transformed
    return jsp_data


def transform_jsp_to_simpy(jsp_data):
    jsp_instance = jsp_data['jssp_instance']
    # transofrm dataframe for the simpy env
    processing_col_lst = []
    for col in jsp_instance.columns:
        if col.isdigit():  # operation type column
            processing_col_lst.append(col)
            jsp_instance.loc[:, str(col)] = jsp_instance[str(col)].apply(lambda x: (x, col) if x > 0 else np.nan)

    jsp_instance['opduration_machinetype'] = jsp_instance.loc[:, processing_col_lst].apply(
        lambda x: [i for i in x if isinstance(i, tuple)][0],
        axis=1)  # collapse sparse dataframe in one column containing tuple (duration, operation_type)
    jsp_instance['opduration'] = jsp_instance['opduration_machinetype'].apply(
        lambda x: x[0])  # collapse sparse dataframe in
    # one column containing tuple (duration, operation_type)
    jsp_instance['machinetype'] = jsp_instance['opduration_machinetype'].apply(lambda x: x[1])
    jsp_instance = jsp_instance.sort_values(['order', 'operation'])
    jsp_instance = jsp_instance[["order", "operation", "opduration_machinetype"]]
    jsp_data['jssp_instance'] = jsp_instance
    return jsp_data


def transform_jsp_to_jsp_discrete(jsp_data):
    return jsp_data


if __name__ == "__main__":
    from RL_Code.modules.utils.jsp_handling.collect_jsp_tasks import collect_transform_jsp_tasks

    read_path = Path.cwd().parents[3] / Path('Data/jsp_instances/6x6x6')
    # jsp_lst = collect_jsp_tasks(read_path=read_path, jsp_ind_start=1, jsp_ind_end=1, env_tag="jsp_simpy")
    # [transform_jsp_to_simpy(x) for x in jsp_lst]
    #
    # jsp_lst = collect_jsp_tasks(read_path=read_path, jsp_ind_start=1, jsp_ind_end=2, env_tag="jsp_simpy")
    # [transform_jsp_to_l2d(x) for x in jsp_lst]
    #
    # jsp_lst = collect_jsp_tasks(read_path=read_path, jsp_ind_start=1, jsp_ind_end=1, env_tag="jsp_simpy")
    # [transform_jsp_to_jsp_discrete(x) for x in jsp_lst]

    jsp_lst = collect_transform_jsp_tasks(read_path=read_path, jsp_ind_start=1, jsp_ind_end=2, env_tag="jsp_simpy")
    res = convert_l2d_to_simpy_format_lst(jsp_lst)

    jsp_lst = collect_transform_jsp_tasks(read_path=read_path, jsp_ind_start=1, jsp_ind_end=2, env_tag="jsp_simpy")
    res = convert_l2d_to_legacy_format_jsp_discr_lst(jsp_lst)
    a = 1
