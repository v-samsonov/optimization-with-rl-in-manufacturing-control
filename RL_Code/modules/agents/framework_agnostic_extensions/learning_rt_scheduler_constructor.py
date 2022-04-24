from RL_Code.modules.agents.framework_agnostic_extensions.schedules import ConstantSchedule, LinearSchedule, \
    PiecewiseSchedule, linear_interpolation


def parse_ln_rt_endpoints(endpoints_lst):
    transf_endpoints_lst = []
    for lst in endpoints_lst:
        lst[0] = int(float(lst[0]))
        lst[1] = float(lst[1])
        transf_endpoints_lst.append(tuple(lst))
    return transf_endpoints_lst


class LnRtSchedulerConstrucor():
    # Client
    def build_ln_rt_scheduler(self, learning_rate_val, **kwargs):
        builder, ln_rt_kwargs = self._get_builder(learning_rate_val, **kwargs)
        return builder(**ln_rt_kwargs)

    # Creator
    def _get_builder(self, learning_rate_val, **kwargs):
        if type(learning_rate_val) == float or learning_rate_val[0] == 'ConstantSchedule':
            return ConstantSchedule, {'value': learning_rate_val}
        else:
            assert type(
                learning_rate_val) == list  # learning rate var shell be a list of parameters for the constructor
            assert type(learning_rate_val[0]) == str  # first list entry shell be the type of scheduler
            if learning_rate_val[
                0] == 'LinearSchedule':  # schedule_timesteps: (int), final_p: (float), initial_p: (float)
                ln_rt_kwargs = {'schedule_timesteps': learning_rate_val[1], 'final_p': learning_rate_val[2],
                                'initial_p': learning_rate_val[3]}
                return LinearSchedule, ln_rt_kwargs
            if learning_rate_val[0] == 'PiecewiseSchedule':
                # endpoints – ([(int, int)]) list of pairs (time, value)
                # interpolation – (lambda (float, float, float): float)
                # outside_value – (float)
                ln_rt_kwargs = {'endpoints': parse_ln_rt_endpoints(learning_rate_val[1]),
                                'interpolation': linear_interpolation,
                                'outside_value': learning_rate_val[2]}
                return PiecewiseSchedule, ln_rt_kwargs

    # Product
    # None
