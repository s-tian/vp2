import torch

from vp2.mpc.utils import stack_dicts
from vp2.models.model import VideoPredictionModel


class EnsembleModel(VideoPredictionModel):
    def __init__(self, models, epoch=None):
        self.models = models

    def __call__(self, batch):
        preds = list()
        for model in self.models:
            pred = model(batch)
            preds.append(pred)
        return stack_dicts(preds)

    @property
    def num_context(self):
        return self.models[0].num_context

    @property
    def base_prediction_modality(self):
        return self.models[0].base_prediction_modality
