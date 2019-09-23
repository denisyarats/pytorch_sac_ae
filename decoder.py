import torch
import torch.nn as nn

from encoder import OUT_DIM


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


class StateDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 1

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024),
            nn.ReLU(), nn.Linear(1024, obs_shape[0]), nn.ReLU()
        )

        self.outputs = dict()

    def forward(self, obs, detach=False):
        h = self.trunk(obs)
        if detach:
            h = h.detach()
        self.outputs['h'] = h
        return h

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        L.log_param('train_encoder/fc1', self.trunk[0], step)
        L.log_param('train_encoder/fc2', self.trunk[2], step)
        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)


_AVAILABLE_DECODERS = {'pixel': PixelDecoder, 'state': StateDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    if decoder_type == 'pixel':
        return _AVAILABLE_DECODERS[decoder_type](
            obs_shape, feature_dim, num_layers, num_filters
        )
    return _AVAILABLE_DECODERS[decoder_type](obs_shape, feature_dim)
