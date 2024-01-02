import torch
import torch.nn as nn
from contextlib import ExitStack

from vp2.models.model import VideoPredictionModel
from vp2.mpc.utils import dict_to_float_tensor, dict_to_cuda

try:
    from svg_prime import utils
    import svg_prime.models.shallow_vgg as shallow_vgg_model
    import svg_prime.models.conv_lstm as lstm_models
except ModuleNotFoundError:
    raise ModuleNotFoundError(f"Failed to load SVG' model. This is installed separately from the VP2 benchmark. "
                              f"Please follow the package installation instructions in the VP2 README to clone/install.")


class SVGPrime(nn.Module):
    """
    SVG' model as a nn.Module, assuming that all modules are already created.
    See "from_config" to construct a new SVGPrime model with fresh weights.
    """

    def __init__(self, encoder, decoder, frame_predictor, posterior, prior):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.frame_predictor = frame_predictor
        self.posterior = posterior
        self.prior = prior

    def load_state_dict_from_components(self, state_dict):
        """
        :param state_dict: A dict with keys "encoder", "decoder", "frame_predictor", "posterior", "prior"
        :return: None
        """
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.frame_predictor.load_state_dict(state_dict["frame_predictor"])
        self.posterior.load_state_dict(state_dict["posterior"])
        self.prior.load_state_dict(state_dict["prior"])

    def init_hidden(self, batch_size):
        # Set batch sizes for hidden states
        self.frame_predictor.batch_size = batch_size
        self.posterior.batch_size = batch_size
        self.prior.batch_size = batch_size

        # Initialize hidden states
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

    def forward(self, x, actions):
        """
        :param x: input context images in shape [T, B, C, H, W]
        :param actions: input actions in shape [T, B, a_dim]
        :return: predicted video in shape [T, B, C, H, W]
        """

        batch_size = x.shape[1]
        self.init_hidden(batch_size=batch_size)
        gen_seq = list()
        x_in = x[0]
        num_context = len(x)
        for i in range(1, len(actions) + 1):
            h = self.encoder(x_in)
            if i < num_context:
                h, skip = h
            else:
                h, _ = h
            tiled_action = utils.tile_actions_into_image(actions[i - 1], h.shape[-2:])
            if i < num_context:
                h_target = self.encoder(x[i])
                h_target = h_target[0]
                z_t, _, _ = self.posterior(h_target)
                self.prior(h)
                self.frame_predictor(torch.cat([h, tiled_action, z_t], dim=1))
                x_in = x[i]
                gen_seq.append(x_in)
            else:
                z_t, _, logvar = self.prior(h)
                h = self.frame_predictor(torch.cat([h, tiled_action, z_t], dim=1))
                x_in = self.decoder([h, skip])
                gen_seq.append(x_in)
        return torch.stack(gen_seq)

    @classmethod
    def from_config(cls, opt):
        """
        :param opt: An object with attributes:
        z_dim: dimensionality of z_t
        g_dim: dimensionality of encoder output vector and decoder input vector
        a_dim: dimensionality of robot action
        posterior_rnn_layers: number of layers in RNN for posterior prediction
        prior_rnn_layers: number of layers in RNN for prior prediction
        predictor_rnn_layers: number of layers in RNN for frame prediction
        rnn_size: dimensionality of RNN hidden layer
        channels: number of channels in the image (default is 3)
        M: scaling factor for LSTMs
        K: scaling factor for encoder/decoder
        :return: SVGPrime instance with randomly initialized weights
        """
        encoder = shallow_vgg_model.encoder(opt.g_dim, opt.channels, expand=opt.K)
        decoder = shallow_vgg_model.decoder(opt.g_dim, opt.channels, expand=opt.K)
        encoder.apply(utils.init_weights)
        decoder.apply(utils.init_weights)
        frame_predictor = lstm_models.ConvLSTM(opt.g_dim+opt.z_dim+opt.a_dim, opt.g_dim, opt.rnn_size, (8, 8), opt.predictor_rnn_layers, opt.batch_size, expand=opt.M)
        posterior = lstm_models.ConvGaussianLSTM(opt.g_dim, opt.z_dim, opt.rnn_size, (8, 8), opt.posterior_rnn_layers, 32, expand=opt.M)
        prior = lstm_models.ConvGaussianLSTM(opt.g_dim, opt.z_dim, opt.rnn_size, (8, 8), opt.prior_rnn_layers, 32, expand=opt.M)
        frame_predictor.apply(utils.init_weights)
        posterior.apply(utils.init_weights)
        prior.apply(utils.init_weights)
        return SVGPrime(encoder=encoder,
                        decoder=decoder,
                        frame_predictor=frame_predictor,
                        posterior=posterior,
                        prior=prior)


class SVGPrimeInterface(VideoPredictionModel):
    def __init__(
        self,
        checkpoint_dir,
        n_past,
        planning_modalities,
        max_batch_size=200,
        epoch=None,
        device="cuda:0",
    ):
        self.checkpoint_file = self.get_checkpoint_file(checkpoint_dir, epoch)
        saved_model = torch.load(self.checkpoint_file)

        self.model = SVGPrime.from_config(saved_model["opt"])
        self.model.load_state_dict_from_components(saved_model)
        self.model.eval()
        self.model.cuda()
        self.planning_modalities = planning_modalities
        self.base_prediction_modality = "rgb"
        self.num_context = n_past
        self.max_batch_size = max_batch_size
        self.device = device

    def format_model_epoch_filename(self, epoch):
        return f"model_{epoch}_sd.pth"

    def prepare_batch(self, xs):
        keys = ["video", "actions"]
        batch = {k: x for k, x in xs.items() if k in keys}
        batch = dict_to_float_tensor(batch)
        batch = dict_to_cuda(batch, device=self.device)
        batch["video"] = torch.permute(batch["video"], (1, 0, 4, 2, 3))
        batch["actions"] = torch.permute(batch["actions"], (1, 0, 2))
        return batch["video"], batch["actions"]

    def __call__(self, batch, grad_enabled=False):
        all_preds = list()
        with torch.no_grad() if not grad_enabled else ExitStack():
            video, actions = self.prepare_batch(batch)
            for compute_batch_idx in range(
                0, batch["video"].shape[0], self.max_batch_size
            ):
                predictions = self.model(
                    video[
                        :, compute_batch_idx : compute_batch_idx + self.max_batch_size
                    ],
                    actions[
                        :, compute_batch_idx : compute_batch_idx + self.max_batch_size
                    ],
                )
                predictions = predictions.permute(1, 0, 3, 4, 2)
                all_preds.append(predictions)
        predictions = torch.cat(all_preds, dim=0)
        if not grad_enabled:
            predictions = predictions.cpu().numpy()
        return dict(rgb=predictions)

