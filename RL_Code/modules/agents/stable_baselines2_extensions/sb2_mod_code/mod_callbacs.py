from stable_baselines.common.callbacks import CheckpointCallback
import os

class ModCheckpointCallback(CheckpointCallback):
    # def __init__(self, *args, **kwargs):
    #     super(ModCheckpointCallback, self).__init__()
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, '{}_{}_steps'.format(self.name_prefix, 0))
        self.model.save(path)
        if self.verbose > 1:
            print("Saving model checkpoint to {}".format(path))