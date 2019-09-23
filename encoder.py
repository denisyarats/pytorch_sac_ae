import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(
        self,
        obs_shape,
        feature_dim,
        num_layers=2,
        num_filters=32,
        stochastic=False
    ):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.stochastic = stochastic

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        if self.stochastic:
            self.log_std_min = -10
            self.log_std_max = 2
            self.fc_log_std = nn.Linear(
                num_filters * out_dim * out_dim, self.feature_dim
            )

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)

        if self.stochastic:
            self.outputs['mu'] = out
            log_std = torch.tanh(self.fc_log_std(h))
            # normalize
            log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
            ) * (log_std + 1)
            out = self.reparameterize(out, log_std)
            self.outputs['log_std'] = log_std

        self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class StateEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = feature_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_shape[0], 256), nn.ReLU(),
            nn.Linear(256, feature_dim), nn.ReLU()
        )

        self.outputs = dict()

    def forward(self, obs, detach=False):
        h = self.trunk(obs)
        if detach:
            h = h.detach()
        self.outputs['h'] = h
        return h

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        L.log_param('train_encoder/fc1', self.trunk[0], step)
        L.log_param('train_encoder/fc2', self.trunk[2], step)
        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {
    'pixel': PixelEncoder,
    'state': StateEncoder,
    'identity': IdentityEncoder
}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stochastic
):
    assert encoder_type in _AVAILABLE_ENCODERS
    if encoder_type == 'pixel':
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters, stochastic
        )
    return _AVAILABLE_ENCODERS[encoder_type](obs_shape, feature_dim)
