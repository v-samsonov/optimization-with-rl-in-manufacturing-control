import numpy as np
import pandas as pd


# ===============> Used for generation of jsp instances on the fly in the env<============
def generate_jsp_default(n_j, n_m, low, high, random_generator, n_ops_per_job=-1):
    string = [str(m) for m in range(n_m)]
    j_r_t = pd.DataFrame(columns=["job", "order"])
    j_r_t["job"] = np.repeat(np.arange(0, n_j), n_ops_per_job)
    j_r_t["order"] = np.tile(np.arange(0, n_ops_per_job), n_j)
    j_r_t_or_solver = []
    for m in range(n_m):
        j_r_t[string[m]] = 0
    for j in range(n_j):
        job_data_or_solver = []
        mach_nums = np.arange(0, n_m)
        for o in range(n_ops_per_job):
            index = random_generator.integers(low=0, high=mach_nums.size, size=1, endpoint=False)[0]
            mach_num = mach_nums[index]
            mach_nums = np.delete(mach_nums, index)
            time = random_generator.integers(low=low, high=high, size=1, endpoint=True)[0]
            j_r_t.loc[(j_r_t.job == j) & (j_r_t.order == o), [str(mach_num)]] = time
            job_data_or_solver.append([mach_num, time])
        j_r_t_or_solver.append(job_data_or_solver)
    j_r_t["state"] = "todispatch"
    j_r_t.sort_values(['job', 'order'], inplace=True)
    return j_r_t, j_r_t_or_solver


def generate_jsp_l2d(n_j, n_m, low, high, random_generator, n_ops_per_job=-1):
    if n_ops_per_job!=-1:
        assert n_m==n_ops_per_job
    times = random_generator.integers(low=low, high=high, size=(n_j, n_m), endpoint=True)
    machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows_l2d(machines, random_generator)
    return times, machines

def permute_rows_l2d(x, random_generator):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = random_generator.random(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


# ===============> Used for batch generation and solving of jsp instances <============
def generate_jsp_default_one_rgnr(n_j, n_m, low, high, n_ops_per_job=-1):
    string = [str(m) for m in range(n_m)]
    j_r_t = pd.DataFrame(columns=["job", "order"])
    j_r_t["job"] = np.repeat(np.arange(0, n_j), n_ops_per_job)
    j_r_t["order"] = np.tile(np.arange(0, n_ops_per_job), n_j)
    j_r_t_or_solver = []
    for m in range(n_m):
        j_r_t[string[m]] = 0
    for j in range(n_j):
        job_data_or_solver = []
        mach_nums = np.arange(0, n_m)
        for o in range(n_ops_per_job):
            index = np.random.randint(0, mach_nums.size)
            mach_num = mach_nums[index]
            mach_nums = np.delete(mach_nums, index)
            time = np.random.randint(low=low,high=high+1)
            j_r_t.loc[(j_r_t.job == j) & (j_r_t.order == o), [str(mach_num)]] = time
            job_data_or_solver.append([mach_num, time])
        j_r_t_or_solver.append(job_data_or_solver)
    j_r_t["state"] = "todispatch"
    j_r_t.sort_values(['job', 'order'], inplace=True)
    return j_r_t, j_r_t_or_solver


def generate_jsp_l2d_one_rgnr(n_j, n_m, low, high, n_ops_per_job=-1):
    if n_ops_per_job!=-1:
        assert n_m==n_ops_per_job
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m + 1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows_l2d_one_rgnr(machines)
    return times, machines

def permute_rows_l2d_one_rgnr(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]
