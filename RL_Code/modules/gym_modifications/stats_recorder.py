import json
import os
import time

import wandb
from gym import error
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np


class StatsRecorderMod(object):
    def __init__(self, directory, file_prefix, wandb_logs, autoreset=False, env_id=None, log_frequency=1):
        self.autoreset = autoreset
        self.env_id = env_id

        self.initial_reset_timestamp = None
        self.directory = directory
        self.file_prefix = file_prefix
        self.episode_lengths = []
        self.episode_rewards = []
        self.episode_types = []  # experimental addition
        self._type = 't'
        self.timestamps = []
        self.steps = None
        self.total_steps = 0
        self.rewards = None
        self.infos = []
        self.info = {}

        self.done = None
        self.closed = False
        self.wandb_logs = wandb_logs
        self.episode_number = 0
        self.log_frequency = log_frequency

        filename = '{}.stats.json'.format(self.file_prefix)
        self.path = os.path.join(self.directory, filename)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in ['t', 'e']:
            raise error.Error('Invalid episode type {}: must be t for training or e for evaluation', type)
        self._type = type

    def before_step(self, action):
        assert not self.closed

        if self.done:
            raise error.ResetNeeded(
                "Trying to step environment which is currently done. While the monitor is active for {}, you cannot step beyond the end of an episode. Call 'env.reset()' to start the next episode.".format(
                    self.env_id))
        elif self.steps is None:
            raise error.ResetNeeded(
                "Trying to step an environment before reset. While the monitor is active for {}, you must call 'env.reset()' before taking an initial step.".format(
                    self.env_id))

    def after_step(self, observation, reward, done, info):
        self.steps += 1
        self.total_steps += 1
        self.rewards += reward
        self.done = done
        self.info = info

        if done:
            self.save_complete()

        if done:
            if self.autoreset:
                self.before_reset()
                self.after_reset(observation)

    def before_reset(self):
        assert not self.closed
        if self.steps is not None:
            if self.done is not None and not self.done and self.steps > 0:
                raise error.Error(
                    "Tried to reset environment which is not done. While the monitor is active for {}, you cannot call reset() unless the episode is over.".format(
                        self.env_id))

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_reset(self, observation):
        self.steps = 0
        self.rewards = 0
        self.episode_number += 1
        # We write the type at the beginning of the episode. If a user
        # changes the type, it's more natural for it to apply next
        # time the user calls reset().
        self.episode_types.append(self._type)

    def save_complete(self):
        # logging data at the end of the episode
        if self.steps is not None:
            if self.wandb_logs:
                if (self._type == 'e') | (
                        len(self.timestamps) == 0):  # always log first episode and evaluation episodes
                    log_flag = True
                else:  # log every log_frequency-th episode during training
                    log_flag = (self.episode_number % self.log_frequency) == 0

                if (self._type == 'e') & (log_flag):
                    if len(self.timestamps) > 0:
                        wandb.log({'solution_time': time.time() - self.timestamps[-1]}, commit=False)
                    else:  # first reset
                        wandb.log({'solution_time': time.time() - self.initial_reset_timestamp}, commit=False)

                if log_flag:
                    if len(self.info) > 0:
                        for k, v in self.info.items():
                            wandb.log({k: v}, commit=False)
                    wandb.log({
                        'episode_reward': float(self.rewards),
                        'env_id': self.env_id,
                        'total_steps': self.total_steps,
                        'episode_number': self.episode_number
                    })

            self.info.pop('action_mask', None)
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(float(self.rewards))
            self.timestamps.append(time.time())

            if len(self.infos) > 0:
                for k, v in self.info.items():
                    self.infos[k].append(v)
            else:
                self.infos = {k: [v] for k, v in self.info.items()}

    def close(self):
        self.flush()
        self.closed = True

    def flush(self):
        if self.closed:
            return

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
                'info': self.infos
            }, f, default=json_encode_np)
