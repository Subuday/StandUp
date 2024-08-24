import copy
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from functorch import combine_state_for_ensemble

import math_utils
from utils import get_device_from_parameters

@dataclass
class TDMPC2Config:
    checkpoint: str = "./cup-spin.pt"
    
    observation_dim: int = 8
    action_dim: int = 2

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

    episode_length: int = 1000

    temporal_decay_coeff: float = 0.5

    consistency_coeff: float = 20
    reward_coeff: float = 0.1
    q_value_coeff: float = 0.1

class TDMPC2Policy(nn.Module):
    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config
        self.model = TDMPC2TOLD(config = config)

    @staticmethod
    def _discount_factor(config: TDMPC2Config) -> Tensor:
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.
        """
        frac = config.episode_length / config.discount_factor_denom
        return min(max((frac - 1) / (frac), config.discount_factor_min), config.discount_factor_max)
    

    def load(self, file_path: str):
        dict = torch.load(file_path, weights_only=True)["model"]
        new_keys = []

        for key in dict.keys():
            if "_target_Qs" in key:
                new_keys.append((key, key.replace("_target_Qs", "_Qs_target")))

        for old_key, new_key in new_keys:
            dict[new_key] = dict.pop(old_key)
        
        self.model.load_state_dict(dict)


    def forward(self, batch: dict[str, Tensor]):
        device = get_device_from_parameters(self)

        observations = batch['observations']
        current_observation = observations[0]
        next_observations = observations[1:]
        actions = batch['actions']
        reward = batch['reward']

        horizon, batch_size = next_observations.shape[:2]
        
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

        # Compute various targets with stopgrad.
        with torch.no_grad():
            # Latent state consistency targets.
            z_targets = self.model.encode(next_observations)
            pi = self.model.pi(z_targets)[0]
            q_targets = self.model.Qs_target(z_targets, pi)
            q1_target, q2_target = q_targets[np.random.choice(self.config.num_Qs, 2, replace=False)]
            q1_target, q2_target = math_utils.two_hot_inv(q1_target, self.config), math_utils.two_hot_inv(q2_target, self.config)
            q_targets = reward + TDMPC2Policy._discount_factor(self.config) * torch.min(q1_target, q2_target)

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
                * math_utils.ce_loss(reward_preds, math_utils.two_hot(reward, self.config)).mean(dim=-1)
            )
            .sum(0)
            .mean()
        ) * 1 / horizon
        
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        q_value_loss = (
            (
                temporal_loss_coeffs.repeat(self.config.num_Qs, 1)
                * math_utils.ce_loss(q_preds, math_utils.two_hot(q_targets, self.config).unsqueeze(0).expand(5, -1, -1, -1)).mean(dim=-1).view(self.config.num_Qs * horizon, -1)
            )
            .sum(0)
            .mean()
        ) * 1 / (horizon + self.config.num_Qs)

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.q_value_coeff * q_value_loss
        )


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
        self._Qs = VectorizedModuleList([
            nn.Sequential(
                NormedLinear(config.latent_dim + config.action_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True), dropout = 0.01),
                NormedLinear(config.mlp_hidden_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
                nn.Linear(config.mlp_hidden_dim, config.num_bins, bias = True)
            )
            for _ in range(config.num_Qs)
        ])
        self._Qs_target = copy.deepcopy(self._Qs).requires_grad_(False)
        self._pi = nn.Sequential(
            NormedLinear(config.latent_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            NormedLinear(config.mlp_hidden_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
            nn.Linear(config.mlp_hidden_dim, 2 * config.action_dim, bias = True)
        )

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

        log_std = math_utils.log_std(log_std, pi_log_std_min, pi_log_std_max)
        eps = torch.randn_like(mu)
        log_pi = math_utils.gaussian_logprob(eps, log_std)
        pi = mu + eps * torch.exp(log_std)
        mu, pi, log_pi = math_utils.squash(mu, pi, log_pi)
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