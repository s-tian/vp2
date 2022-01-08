import abc
import numpy as np
from vp2.mpc.utils import (
    ObservationList,
    write_moviepy_gif,
    generate_text_square,
)


class Optimizer(metaclass=abc.ABCMeta):
    def log_best_plans(self, filename, vis_preds, goal, scores, fps=5):
        # Make sure that the goal has the same time dimension as the predictions
        if len(goal) == 1:
            goal = goal.repeat(len(vis_preds[0]))

        vis_preds = [o.append(goal, axis=1) for o in vis_preds]
        vis_preds = ObservationList.from_observations_list(vis_preds, axis=2)

        vis_preds_image = vis_preds.to_image_list()

        # Create images containing text with the score
        score_image = np.concatenate(
            [
                generate_text_square(str(np.round(score, decimals=4).item()))
                for score in scores
            ],
            axis=-1,
        )
        # shape is [3, 64, 64*num_plans]
        score_image = np.moveaxis(score_image, 0, -1)
        # shape is [64, 64*num_plans, 3]
        score_image = np.tile(score_image[None], (len(vis_preds), 1, 1, 1))
        # Add the score images to the visualizations
        vis_preds_image = np.concatenate((vis_preds_image, score_image), axis=1)
        # Write the gif
        write_moviepy_gif(list(vis_preds_image), filename, fps=fps)
