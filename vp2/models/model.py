import abc
import os

from hydra.utils import to_absolute_path


class VideoPredictionModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, batch):
        raise NotImplementedError

    def format_model_epoch_filename(self, epoch):
        raise NotImplementedError

    def get_checkpoint_file(self, checkpoint_dir, model_epoch):
        checkpoint_dir = to_absolute_path(checkpoint_dir)
        if model_epoch is not None:
            if os.path.isfile(checkpoint_dir):
                # If it's already pointing to a file, remove the file name
                checkpoint_dir = os.path.dirname(checkpoint_dir)
            checkpoint_file = os.path.join(
                checkpoint_dir, self.format_model_epoch_filename(model_epoch)
            )
        else:
            checkpoint_file = checkpoint_dir
        return checkpoint_file

    def close(self):
        """
        Clean up, e.g., any multiprocessing, or other things
        """

        pass
