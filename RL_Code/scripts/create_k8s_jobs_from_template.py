import datetime
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

from RL_Code.modules.utils.file_handling.yaml_to_dict_parser import get_config_pars

template_file = "base_job_with_secrets&antiaffinity.yaml"

git_project = "v-samsonov/check_EAAI.git"
pr_folder = "check_EAAI"
branchname = "main"  # "master", "observation_manager_extension" "l2d_observation_manager", "n+1-lookahead"
docker_hub_image = "vladimir249/rl:py37_sb23_torch140_tf114_cf_v1"  # "vladimir249/rl:rl_torch_test_v1" # "vladimir249/rl:py37_sb23_torch140_tf114_cf_v1"

script_to_execute = "RL_Code/scripts/run_experiment_k8s.py"
date_stamp = None  # None - use current datestamp or pass a string related to the experiment

wandb_project = "validate_rl4jsp_submission"
# exp_alias = "baselines", "n_steps_look_ahead"

random_seed_lst = [5231, 4821, 1292]  # , 4821] # [5231, 4821, 1292] [None]
config_files_dict = {
    'comparison_of_reward_shapes/dense_lb_rew_dqn_6x6x6_900jssp_eps02_0i_run_config.yml': 'comparison_of_reward_shapes',
    'comparison_of_reward_shapes/dense_lb_rew_dqn_8x8x8_900jssp_eps02_0i_run_config.yml': 'comparison_of_reward_shapes',
    'comparison_of_reward_shapes/dense_lb_rew_dqn_10x10x10_900jssp_eps02_0i_run_config.yml': 'comparison_of_reward_shapes',

    'comparison_of_reward_shapes/dqn_6x6x6_900jssp_eps02_0i_run_config.yml': 'comparison_of_reward_shapes',
    'comparison_of_reward_shapes/dqn_8x8x8_900jssp_eps02_0i_run_config.yml': 'comparison_of_reward_shapes',
    'comparison_of_reward_shapes/dqn_10x10x10_900jssp_eps02_0i_run_config.yml': 'comparison_of_reward_shapes',

    'comparison_of_reward_shapes/r2_70_makespan_900jsp_eps02_explr_incent_01_6x6x6_0i_run_config.yml': 'comparison_of_reward_shapes',
    'comparison_of_reward_shapes/r2_70_makespan_900jsp_eps02_explr_incent_01_8x8x8_0i_run_config.yml': 'comparison_of_reward_shapes',
    'comparison_of_reward_shapes/r2_70_makespan_900jsp_eps02_explr_incent_01_10x10x10_0i_run_config.yml': 'comparison_of_reward_shapes',

    'direct_vs_indirect_action_space_mapping/direct_act_space_900jsp_6x6x6_0i_run_config.yml': 'direct_vs_indirect_action_space_mapping',
    'direct_vs_indirect_action_space_mapping/direct_act_space_900jsp_8x8x8_0i_run_config.yml': 'direct_vs_indirect_action_space_mapping',
    'direct_vs_indirect_action_space_mapping/direct_act_space_900jsp_10x10x10_0i_run_config.yml': 'direct_vs_indirect_action_space_mapping',

    'direct_vs_indirect_action_space_mapping/dqn_6x6x6_900jssp_eps02_0i_run_config.yml': 'direct_vs_indirect_action_space_mapping',
    'direct_vs_indirect_action_space_mapping/dqn_8x8x8_900jssp_eps02_0i_run_config.yml': 'direct_vs_indirect_action_space_mapping',
    'direct_vs_indirect_action_space_mapping/dqn_10x10x10_900jssp_eps02_0i_run_config.yml': 'direct_vs_indirect_action_space_mapping',

    # 'enhanced_baseline_scalability_6x6x6_2_15x15x15/r2_70_makespan_6x6x6_2_15x15x15_900jsp_eps02_shuffle_explr_incent_01_6x6x6_0i_run_config.yml': 'enhanced_baseline_scalability_6x6x6_2_15x15x15',
    ## 'enhanced_baseline_scalability_6x6x6_2_15x15x15/r2_70_makespan_6x6x6_2_15x15x15_900jsp_eps02_shuffle_explr_incent_01_6x6x6_0i_eval_config.yml': 'enhanced_baseline_scalability_6x6x6_2_15x15x15', #ToDo change python script here
    #
    # 'shuffling_vs_constant_state/baseline_900jsp_dqn_6x6x6_0i_run_config.yml': 'shuffling_vs_constant_state',
    # 'shuffling_vs_constant_state/shuffle_900jsp_dqn_6x6x6_0i_run_config.yml': 'shuffling_vs_constant_state',
    #
    # 'enhanced_baseline_vs_l2d_Zhang_et_al/bench_l2d_6x6x6_cpu_4_env_rl4jsp_gen_0i_run_config.yml': 'enhanced_baseline_vs_l2d_Zhang_et_al',
    # 'enhanced_baseline_vs_l2d_Zhang_et_al/bench_l2d_8x8x8_cpu_4_env_rl4jsp_gen_0i_run_config.yml': 'enhanced_baseline_vs_l2d_Zhang_et_al',
    # 'enhanced_baseline_vs_l2d_Zhang_et_al/bench_l2d_10x10x10_cpu_4_env_rl4jsp_gen_0i_run_config.yml': 'enhanced_baseline_vs_l2d_Zhang_et_al',

    # 'enhanced_baseline_vs_l2d_Zhang_et_al/r2_70_makespan_900jsp_eps02_explr_incent_01_6x6x6_0i_run_config.yml': 'enhanced_baseline_vs_l2d_Zhang_et_al',
    # 'enhanced_baseline_vs_l2d_Zhang_et_al/r2_70_makespan_900jsp_eps02_explr_incent_01_8x8x8_0i_run_config.yml': 'enhanced_baseline_vs_l2d_Zhang_et_al',
    # 'enhanced_baseline_vs_l2d_Zhang_et_al/r2_70_makespan_900jsp_eps02_explr_incent_01_10x10x10_0i_run_config.yml': 'enhanced_baseline_vs_l2d_Zhang_et_al',

    # 'evaluation_l2d_Zhang_et_al/bench_l2d_6x6x6_cpu_4_env_rl4jsp_gen_0i_run_config.yml': 'enhanced_baseline_vs_l2d_Zhang_et_al',
    # 'evaluation_l2d_Zhang_et_al/bench_l2d_8x8x8_cpu_4_env_rl4jsp_gen_0i_run_config.yml': 'evaluation_l2d_Zhang_et_al',
    # 'evaluation_l2d_Zhang_et_al/bench_l2d_10x10x10_cpu_4_env_rl4jsp_gen_0i_run_config.yml': 'evaluation_l2d_Zhang_et_al',

    # 'test/test_dqn_8x8x8_900jssp_eps02_0i_run_config.yml': 'test_k8s',
    # 'test/test_bench_l2d_10x10x10_cpu_4_env_rl4jsp_gen_0i_run_config.yml': 'test_k8s',
}
request_cpu = "1"
request_memory = "2500Mi"

# prevent jobs from been scheduled to certain node types
affinity_key = "nodetype"
affinity_logical_expr = "NotIn"  # "NotIn", "In"
affinity_value = "imamaster" # imamaster

# # load secrets
# secret_wandb_path = Path("../../Cluster_Configs/secret_wandb.yaml")
# with open(secret_wandb_path, "r") as cp:
#     secret_wandb = yaml.safe_load(cp)
# secret_git_path = Path.cwd().parents[1] / Path("Cluster_Configs/secret_git.yaml")
# with open(secret_git_path, "r") as cp:
#     secret_git = yaml.safe_load(cp)
#
# wandb_api_key = secret_wandb['stringData']['API_KEY']
# git_api_key = secret_git['stringData']['API_KEY']
# git_api_secret = secret_git['stringData']['API_SECRET']

if date_stamp is None:
    date_stamp = str(datetime.datetime.now())[:19].replace(' ', '_').replace('-', '.').replace(':',
                                                                                               '.')  # used to group related runs together
jinja = Environment(loader=FileSystemLoader(searchpath="../../Data/k8s_jobs/"), trim_blocks=True, lstrip_blocks=True)
template = jinja.get_template(template_file)
counter = 0
for config_file, exp_alias in config_files_dict.items():
    # check config files availability and naming conventions
    # get parameters dict
    run_config_file_path = Path.cwd() / Path('../../Data/run_config_files/{}'.format(config_file))
    run_config_pars = get_config_pars(run_config_file_path)
    job_group = exp_alias
    for random_seed in random_seed_lst:
        if random_seed is None:
            random_seed = ""
        conf_name = config_file[config_file.rfind("/") + 1: config_file.rfind("_run_config.yml")]
        data = {
            "JOBGROUP": job_group,
            "Job_Name": conf_name.replace('_', '-') + "-" + str(random_seed) + "-",
            "IMAGE_REFERENCE": docker_hub_image,
            "SCRIPT_TO_EXECUTE": script_to_execute,
            "CONFIG_FILE": config_file,
            "RANDOM_SEED": random_seed,
            "EXP_ALIAS": exp_alias,
            "WANDB_PROJECT": wandb_project,
            "DATES_TAMP": date_stamp,
            "GIT_PROJECT": git_project,
            "PR_FOLDER": pr_folder,
            "BRANCHNAME": branchname,
            # "GIT_API_KEY": git_api_key,
            # "WANDB_API_KEY": wandb_api_key,
            "WANDB_DOCKER_ENV_VAR": docker_hub_image,
            "REQEST_CPU": request_cpu,
            "REQEST_MEMORY": request_memory,
            "DATE_STAMP": date_stamp,
            "AffinityKey": affinity_key,
            "AffinityLogicalExpr": affinity_logical_expr,
            "AffinityValue": affinity_value,
        }

        k8s_job_definition = template.render(data)
        save_path = Path("../../Data/k8s_jobs/generated_jobs")
        save_path.mkdir(exist_ok=True)
        with open(save_path / Path(f'job-{conf_name}-{random_seed}.yaml'), "w+") as f:
            f.write(k8s_job_definition)
        print(f'file {save_path} is created')
        counter += 1

print(template.render(data))
print("total runs:", counter)
