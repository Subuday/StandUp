import copy
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functorch import combine_state_for_ensemble

from common import math, init
from common.scale import RunningScale
from utils import get_device_from_parameters

@dataclass
class TDMPC2Config:
    checkpoint: str = "./dog-run.pt"
    seed: int = 1

    horizon: int = 3
    
    # Environment
    action_dim: int = 38

    # Inference.
    use_mpc: bool = True
    cem_iterations: int = 10
    max_std: float = 2.0
    min_std: float = 0.05
    n_gaussian_samples: int = 512
    n_pi_samples: int = 24
    n_elites: int = 64
    elite_weighting_temperature: float = 0.5

    eval_episodes: int = 100
    
    observation_dim: int = 223

    simplex_dim: int = 8

    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 2

    mlp_hidden_dim: int = 512

    latent_dim: int = 512

    num_bins: int = 101
    vmin: int = -10
    vmax: int = +10

    num_Qs: int = 5

    pi_log_std_min: float = -10
    pi_log_std_max: float = 2

    discount_factor_min: float = 0.95
    discount_factor_max: float = 0.995
    discount_factor_denom: int = 5

    episode_length: int = 500

    temporal_decay_coeff: float = 0.5

    consistency_coeff: float = 20
    reward_coeff: float = 0.1
    q_value_coeff: float = 0.1

    # Training
    training_steps: int = 5_000_000
    training_batch_size: int = 256
    buffer_seed_size: int = 2500
    buffer_capacity: int = 1_000_000
    grad_clip_norm: float = 20.0
    lr: float = 3e-4
    encoder_lr: float = 0.3
    tau: float = 0.01
    entropy_coeff: float = 1e-4

class TDMPC2Policy(nn.Module):
    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config
        self.model = TDMPC2TOLD(config = config)
        self.model_optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.config.lr * self.config.encoder_lr},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
		], lr=self.config.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr = self.config.lr, eps = 1e-5)
        self.scale = RunningScale(config.tau)
        self.reset()

    @staticmethod
    def _discount_factor(config: TDMPC2Config) -> Tensor:
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.
        """
        frac = config.episode_length / config.discount_factor_denom
        return min(max((frac - 1) / (frac), config.discount_factor_min), config.discount_factor_max)
    
    def reset(self):
        self._prev_mean: torch.Tensor | None = None

    def save(self, file_path: str):
        torch.save({"model": self.model.state_dict()}, file_path)

    def load(self, file_path: str):
        dict = torch.load(file_path, weights_only=True)["model"]
        new_keys = []

        for key in dict.keys():
            if "_target_Qs" in key:
                new_keys.append((key, key.replace("_target_Qs", "_Qs_target")))

        for old_key, new_key in new_keys:
            dict[new_key] = dict.pop(old_key)
        
        self.model.load_state_dict(dict)

    @torch.no_grad()
    def select_action(self, observation: Tensor) -> Tensor:
        z = self.model.encode(observation)
        assert self.config.use_mpc
        return self.plan(z)
    
    @torch.no_grad()
    def plan(self, z: Tensor):
        device = get_device_from_parameters(self)
        
        # Sample Nπ trajectories from the policy.
        pi_actions = torch.empty(
            self.config.horizon,
            self.config.n_pi_samples,
            self.config.action_dim,
            device=device,
        )
        if self.config.n_pi_samples > 0:
            _z = einops.repeat(z, "1 d -> n d", n = self.config.n_pi_samples)
            for t in range(self.config.horizon - 1):
                # Note: Adding a small amount of noise here doesn't hurt during inference and may even be
                # helpful for CEM.
                pi_actions[t] = self.model.pi(_z)[0]
                _z = self.model.latent_dynamics(_z, pi_actions[t])
            pi_actions[-1] = self.model.pi(_z)[0]

        # In the CEM loop we will need this for a call to estimate_value with the gaussian sampled
        # trajectories.
        z = einops.repeat(z, "1 d -> n d", n=self.config.n_gaussian_samples)

        # Model Predictive Path Integral (MPPI) with the cross-entropy method (CEM) as the optimization
        # algorithm.
        # The initial mean and standard deviation for the cross-entropy method (CEM).
        mean = torch.zeros(self.config.horizon, self.config.action_dim, device=device)
        # Maybe warm start CEM with the mean from the previous step.
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.config.max_std * torch.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # Randomly sample action trajectories for the gaussian distribution.
            std_normal_noise = torch.randn(
                self.config.horizon,
                self.config.n_gaussian_samples - self.config.n_pi_samples,
                self.config.action_dim,
                device = device,
            )
            gaussian_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise, -1, 1)

            # Compute elite actions.
            actions = torch.cat([pi_actions, gaussian_actions], dim = 1)
            value = self.estimate_value(z, actions).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.config.n_elites, dim=0).indices
            elite_value = value[elite_idxs]
            elite_actions = actions[:, elite_idxs]
            
            # Update gaussian PDF parameters to be the (weighted) mean and standard deviation of the elites.
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.config.elite_weighting_temperature * (elite_value - max_value))
            score /= score.sum(axis=0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim = 1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(
                torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)
            ).clamp_(self.config.min_std, self.config.max_std)

        self._prev_mean = mean
        
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        action, std = actions[0], std[0]
        if not self.training:
            action += std * torch.randn(self.config.action_dim, device = device)
        return action.clamp(-1, 1)
    
    @torch.no_grad()
    def estimate_value(self, z: Tensor, actions: Tensor) -> Tensor:
        """Estimates the value of a trajectory as per eqn 4 of the FOWM paper."""
        # Initialize return and running discount factor.
        G, running_discount = 0, 1
        # Iterate over the actions in the trajectory to simulate the trajectory using the latent dynamics
        # model. Keep track of return.
        for t in range(self.config.horizon):
            reward = self.model.reward(z, actions[t])
            reward = math.two_hot_inv(reward, self.config)
            z = self.model.latent_dynamics(z, actions[t])
            G += running_discount * reward
            running_discount *= TDMPC2Policy._discount_factor(self.config)
        pi = self.model.pi(z)[0]
        q = self.model.Qs(z, pi)
        q1, q2 = q[np.random.choice(self.config.num_Qs, 2, replace=False)]
        q1, q2 = math.two_hot_inv(q1, self.config), math.two_hot_inv(q2, self.config)
        q = (q1 + q2) / 2
        return G + running_discount * q
    
    def _set_Qs_requires_grad(self, requires_grad: bool):
        for p in self.model._Qs.parameters():
            p.requires_grad_(requires_grad)
    
    def _update_pi(self, zs):
        device = get_device_from_parameters(self)

        self.pi_optim.zero_grad(set_to_none=True)
        self._set_Qs_requires_grad(False)

        pis, log_pis, _, _ = self.model.pi(zs)
        qs = self.model.Qs(zs, pis)
        qs1, qs2 = qs[np.random.choice(self.config.num_Qs, 2, replace=False)]
        qs1, qs2 = math.two_hot_inv(qs1, self.config), math.two_hot_inv(qs2, self.config)
        qs = (qs1 + qs2) / 2
        
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.config.temporal_decay_coeff, torch.arange(len(qs), device = device))
        pi_loss = ((self.config.entropy_coeff * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.config.grad_clip_norm)
        self.pi_optim.step()
        self._set_Qs_requires_grad(True)

        return pi_loss

    def forward(self, batch: dict[str, Tensor]):
        device = get_device_from_parameters(self)

        observations = batch['observations']
        current_observation = observations[0]
        next_observations = observations[1:]
        actions = batch['actions']
        reward = batch['reward']

        horizon, batch_size = next_observations.shape[:2]
        
        # Compute various targets with stopgrad.
        with torch.no_grad():
            # Latent state consistency targets.
            z_targets = self.model.encode(next_observations)
            pi = self.model.pi(z_targets)[0]
            q_targets = self.model.Qs_target(z_targets, pi)
            q1_target, q2_target = q_targets[np.random.choice(self.config.num_Qs, 2, replace=False)]
            q1_target, q2_target = math.two_hot_inv(q1_target, self.config), math.two_hot_inv(q2_target, self.config)
            q_targets = reward + TDMPC2Policy._discount_factor(self.config) * torch.min(q1_target, q2_target)

        self.model_optim.zero_grad(set_to_none=True)

        # Run latent rollout using the latent dynamics model and policy model.
        # Note this has shape `horizon+1` because there are `horizon` actions and a current `z`. Each action
        # gives us a next `z`.
        z_preds = torch.empty(horizon + 1, batch_size, self.config.latent_dim, device=device)
        z_preds[0] = self.model.encode(current_observation)
        for t in range(horizon):
            z_preds[t + 1] = self.model.latent_dynamics(z_preds[t], actions[t])

        # Compute reward predictions based on the latent rollout.
        reward_preds = self.model.reward(z_preds[:-1], actions)

        # Compute Q predictions based on the latent rollout.
        q_preds = self.model.Qs(z_preds[:-1], actions)


        # Compute losses.
        # Exponentially decay the loss weight with respect to the timestep. Steps that are more distant in the
        # future have less impact on the loss. Note: unsqueeze will let us broadcast to (seq, batch).
        temporal_loss_coeffs = torch.pow(self.config.temporal_decay_coeff, torch.arange(horizon, device=device)).unsqueeze(-1)

        # Compute consistency loss as MSE loss between latents predicted from the rollout and latents
        # predicted from the (target model's) observation encoder.
        consistency_loss = (
            (
                temporal_loss_coeffs *
                F.mse_loss(z_preds[1:], z_targets, reduction="none").mean(dim=-1)
            )
            .sum(0)
            .mean()
        ) * 1 / horizon

        # Compute the reward loss as MSE loss between rewards predicted from the rollout and the dataset
        # rewards.
        reward_loss = (
            (
                temporal_loss_coeffs
                * math.soft_ce(reward_preds, reward, self.config).mean(dim=-1)
            )
            .sum(0)
            .mean()
        ) * 1 / horizon
        
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        q_value_loss = (
            (
                temporal_loss_coeffs.repeat(self.config.num_Qs, 1)
                * math.soft_ce(
                    q_preds, 
                    einops.repeat(q_targets, 'h w c -> q h w c', q = q_preds.shape[0]
                ), self.config).mean(dim=-1).view(self.config.num_Qs * horizon, -1)
            )
            .sum(0)
            .mean()
        ) * 1 / (horizon + self.config.num_Qs)

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.q_value_coeff * q_value_loss
        )

        # Update model
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.model_optim.step()

        # Update policy
        pi_loss = self._update_pi(z_preds.detach())

        # Update target Q-functions
        with torch.no_grad():
            for p, p_target in zip(self.model._Qs.parameters(), self.model._Qs_target.parameters()):
                p_target.data.lerp_(p.data, self.config.tau)

        return {
			"loss": loss.item(),
			"consistency_loss": consistency_loss.item(),
			"reward_loss": reward_loss.item(),
			"q_value_loss": q_value_loss.item(),
			"pi_loss": pi_loss.item(),
			"grad_norm": grad_norm.item(),
			"scale": self.scale.value,
        }


class TDMPC2TOLD(nn.Module):
    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config
        self._encoder = TDMPC2ObservationEncoder(config = config)
        self._dynamics = nn.Sequential(
            NormedLinear(config.latent_dim + config.action_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            NormedLinear(config.mlp_hidden_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            NormedLinear(config.mlp_hidden_dim, config.latent_dim, act = SimNorm(V = config.simplex_dim))
        )
        self._reward = nn.Sequential(
            NormedLinear(config.latent_dim + config.action_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            NormedLinear(config.mlp_hidden_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            nn.Linear(config.mlp_hidden_dim, config.num_bins)
        )
        self._pi = nn.Sequential(
            NormedLinear(config.latent_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            NormedLinear(config.mlp_hidden_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            nn.Linear(config.mlp_hidden_dim, 2 * config.action_dim, bias = True)
        )
        self._Qs = VectorizedModuleList([
            nn.Sequential(
                NormedLinear(config.latent_dim + config.action_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True), dropout = 0.01),
                NormedLinear(config.mlp_hidden_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
                nn.Linear(config.mlp_hidden_dim, config.num_bins, bias = True)
            )
            for _ in range(config.num_Qs)
        ])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])

        self._Qs_target = copy.deepcopy(self._Qs).requires_grad_(False)

    def train(self, mode = True):
        super().train(mode)
        self._Qs_target.train(False)
        return self

    def encode(self, observation: Tensor) -> Tensor:
        return self._encoder(observation)

    def latent_dynamics(self, z: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def reward(self, z: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([z, a], dim=-1)
        return self._reward(x)

    def Qs(self, z: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([z, a], dim=-1)
        return self._Qs(x)
    
    def Qs_target(self, z: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([z, a], dim=-1)
        return self._Qs_target(x)

    def pi(self, z: Tensor) -> Union[Tensor, Tensor, Tensor, Tensor]:
        mu, log_std = self._pi(z).chunk(2, dim=-1)

        device = get_device_from_parameters(self)
        pi_log_std_min = torch.tensor(self.config.pi_log_std_min, device=device)
        pi_log_std_max = torch.tensor(self.config.pi_log_std_max, device=device)
        pi_log_std_diff = pi_log_std_max - pi_log_std_min

        log_std = math.log_std(log_std, pi_log_std_min, pi_log_std_diff)
        eps = torch.randn_like(mu)
        log_pi = math.gaussian_logprob(eps, log_std)
        pi = mu + eps * torch.exp(log_std)
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return pi, log_pi, mu, log_std

class TDMPC2ObservationEncoder(nn.Module):
    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.state = nn.Sequential(*TDMPC2ObservationEncoder._create_state(config))

    def _create_state(config: TDMPC2Config) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(NormedLinear(config.observation_dim, config.encoder_hidden_dim, act = nn.Mish(inplace = True)))
        for _ in range(config.encoder_num_layers - 2):
            layers.append(NormedLinear(config.encoder_hidden_dim, config.encoder_hidden_dim, act = nn.Mish(inplace = True)))
        layers.append(NormedLinear(config.encoder_hidden_dim, config.latent_dim, act = SimNorm(V = config.simplex_dim)))
        return layers
    
    def forward(self, x: Tensor) -> Tensor:
        return self.state(x)

class NormedLinear(nn.Linear):
    def __init__(self, *args, act, dropout = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))
    
class SimNorm(nn.Module):
    def __init__(self, V: int):
        super().__init__()
        self.V = V

    def forward(self, x):
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.V)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)
    
class VectorizedModuleList(nn.Module):
    def __init__(self, modules: Optional[Iterable[nn.Module]], **kwargs):
        super().__init__()
        modules = nn.ModuleList(modules)
        fn, params, _ = combine_state_for_ensemble(modules)
        self.vmap = torch.vmap(fn, in_dims=(0, 0, None), randomness="different", **kwargs)
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])

    def forward(self, *args, **kwargs):
        return self.vmap([p for p in self.params], (), *args, **kwargs)