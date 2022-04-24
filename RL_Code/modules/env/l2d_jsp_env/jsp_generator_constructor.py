import numpy as np

from RL_Code.modules.utils.jsp_handling.jsp_format_converters import convert_legacy_pd_df_to_l2d_format
from RL_Code.modules.utils.jsp_handling.generate_jsp_task import generate_jsp_l2d, generate_jsp_default


def generate_jsp_default_for_l2d(n_j, n_m, low, high, random_generator, n_ops_per_job=-1):
    jsp_data, j_r_t_or_solver = generate_jsp_default(n_j, n_m, low, high, random_generator, n_ops_per_job=n_ops_per_job)
    times, machines = convert_legacy_pd_df_to_l2d_format(jsp_df=jsp_data, n_jobs=n_j, n_ops_per_job=n_m)
    return np.array(times), np.array(machines)


JSP_INST_GEN = {'rl4jsp_gen': generate_jsp_default_for_l2d,
                'l2d_gen': generate_jsp_l2d,
                'jssp_list': None
                }
