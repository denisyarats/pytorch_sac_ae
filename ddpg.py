# Code is taken from https://github.com/sfujim/TD3 with slight modifications

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from encoder import make_encoder

LOG_FREQ = 10000


class Actor(nn.Module):
    def __init__(
        self, obs_shape, action_shape, encoder_type, encoder_feature_dim
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim
        )

        self.l1 = nn.Linear(self.encoder.feature_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_shape[0])

        self.outputs = dict()

    def forward(self, obs, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        h = F.relu(self.l1(obs))
        h = F.relu(self.l2(h))
        action = torch.tanh(self.l3(h))
        self.outputs['mu'] = action
        return action

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.l1, step)
        L.log_param('train_actor/fc2', self.l2, step)
        L.log_param('train_actor/fc3', self.l3, step)


class Critic(nn.Module):
    def __init__(
        self, obs_shape, action_shape, encoder_type, encoder_feature_dim
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim
        )

        self.l1 = nn.Linear(self.encoder.feature_dim + action_shape[0], 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.outputs = dict()

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        obs_action = torch.cat([obs, action], dim=1)
        h = F.relu(self.l1(obs_action))
        h = F.relu(self.l2(h))
        q = self.l3(h)
        self.outputs['q'] = q
        return q

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        L.log_param('train_critic/fc1', self.l1, step)
        L.log_param('train_critic/fc2', self.l2, step)
        L.log_param('train_critic/fc3', self.l3, step)


class DDPGAgent(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        discount=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        encoder_type='identity',
        encoder_feature_dim=50
    ):
        self.device = device
        self.discount = discount
        self.tau = tau

        # models
        self.actor = Actor(
            obs_shape, action_shape, encoder_type, encoder_feature_dim
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, encoder_type, encoder_feature_dim
        ).to(device)

        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.actor_target = Actor(
            obs_shape, action_shape, encoder_type, encoder_feature_dim
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(
            obs_shape, action_shape, encoder_type, encoder_feature_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        self.train()
        self.critic_target.train()
        self.actor_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            action = self.actor(obs)
            return action.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        return self.select_action(obs)

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            target_Q = self.critic_target(
                next_obs, self.actor_target(next_obs)
            )
            target_Q = reward + (not_done * self.discount * target_Q)

        current_Q = self.critic(obs, action)

        critic_loss = F.mse_loss(current_Q, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor(self, obs, L, step):
        action = self.actor(obs, detach_encoder=True)
        actor_Q = self.critic(obs, action, detach_encoder=True)
        actor_loss = -actor_Q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        self.update_actor(obs, L, step)

        utils.soft_update_params(self.critic, self.critic_target, self.tau)
        utils.soft_update_params(self.actor, self.actor_target, self.tau)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
