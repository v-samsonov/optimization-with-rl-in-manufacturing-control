from gym.wrappers.monitor import *

from RL_Code.modules.gym_modifications.stats_recorder import StatsRecorderMod


class Monitor_Mod(Monitor):
    def __init__(self, env, directory, wandb_logs, video_callable=None, force=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        self.wandb_logs = wandb_logs
        super(Monitor_Mod, self).__init__(env, directory, video_callable, force, resume,
                                          write_upon_reset, uid, mode)

    def _start(self, directory, video_callable=None, force=False, resume=False,
               write_upon_reset=False, uid=None, mode=None):
        """Start monitoring.

        Args:
            directory (str): A per-training run directory where to record stats.
            video_callable (Optional[function, False]): function that takes in the index of the episode and outputs a boolean, indicating whether we should record a video on this episode. The default (for video_callable is None) is to take perfect cubes, capped at 1000. False disables video recording.
            force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
            resume (bool): Retain the training data already in this directory, which will be merged with our new data
            write_upon_reset (bool): Write the manifest file on each reset. (This is currently a JSON file, so writing it is somewhat expensive.)
            uid (Optional[str]): A unique id used as part of the suffix for the file. By default, uses os.getpid().
            mode (['evaluation', 'training']): Whether this is an evaluation or training episode.
        """
        if self.env.spec is None:
            logger.warn(
                "Trying to monitor an environment which has no 'spec' set. This usually means you did not create it via 'gym.make', and is recommended only for advanced users.")
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            os.makedirs(directory, exist_ok=True)

        if video_callable is None:
            video_callable = capped_cubic_video_schedule
        elif video_callable == False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, None, or False for video_callable, not {}: {}'.format(
                type(video_callable), video_callable))
        self.video_callable = video_callable

        # Check on whether we need to clear anything
        if force:
            clear_monitor_files(directory)
        elif not resume:
            training_manifests = detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.

    You should use a unique directory for each training run, or use 'force=True' to automatically clear previous monitor files.'''.format(
                    directory, ', '.join(training_manifests[:5])))

        self._monitor_id = monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)
        # We use the 'openai-gym' prefix to determine if a file is
        # ours
        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}.{}'.format(self._monitor_id, uid if uid else os.getpid())

        self.stats_recorder = StatsRecorderMod(directory,
                                               '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix),
                                               self.wandb_logs, autoreset=self.env_semantics_autoreset, env_id=env_id)

        if not os.path.exists(directory): os.mkdir(directory)
        self.write_upon_reset = write_upon_reset

        if mode is not None:
            self._set_mode(mode)
