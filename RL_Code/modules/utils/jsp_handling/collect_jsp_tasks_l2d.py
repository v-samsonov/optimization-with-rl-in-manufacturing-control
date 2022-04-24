import numpy as np


def collect_jsp_tasks_l2d(jsp_ind_start, jsp_ind_end, read_path):
    dataLoaded = np.load(read_path)
    jsp_list = []

    for i in range(dataLoaded.shape[0]):
        jsp_list.append((dataLoaded[i][0], dataLoaded[i][1]))

    jsp_list = jsp_list[jsp_ind_start:jsp_ind_end]

    return jsp_list
