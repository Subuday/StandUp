import copy
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math_utils
from utils import get_device_from_parameters

@dataclass
class TDMPC2Config:
    input_dim: int = 51
    action_dim: int = 19

    simplex_dim: int = 8

    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 2

    mlp_hidden_dim: int = 256

    latent_dim: int = 512

    num_bins: int = 101

    num_Qs: int = 5

    pi_log_std_min: float = -10
    pi_log_std_max: float = 2

    temporal_decay_coeff: float = 0.5

class TDMPC2Policy(nn.Module):
    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config
        self.model = TDMPC2TOLD(config = config)

    def forward(self, batch: dict[str, Tensor]):
        device = get_device_from_parameters(self)

        observations = batch['observations']
        current_observation = observations[0]
        next_observations = observations[1:]
        actions = batch['actions']
        reward = batch['reward']

        horizon, batch_size = next_observations[:2]
        
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
            # TODO: Implement discount factor.
            discount = 1
            pi = self.model.pi(z_targets)
            # TODO: Implement Q target computation.
            q_targets = reward + discount * self.model.Qs(z_targets, pi)

        # Compute losses.
        # Exponentially decay the loss weight with respect to the timestep. Steps that are more distant in the
        # future have less impact on the loss. Note: unsqueeze will let us broadcast to (seq, batch).
        # temporal_loss_coeffs = torch.pow(
        #     self.config.temporal_decay_coeff, torch.arange(horizon, device=device)
        # ).unsqueeze(-1)

        # TODO: Check if need padding.
        # Compute the reward loss as MSE loss between rewards predicted from the rollout and the dataset
        # rewards.
        # reward_loss = (
        #     (
        #         temporal_loss_coeffs
        #         * F.mse_loss(reward_preds, reward, reduction="none")
        #     )
        #     .sum(0)
        #     .mean()
        # )

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
                NormedLinear(config.latent_dim + config.action_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
                NormedLinear(config.mlp_hidden_dim, config.mlp_hidden_dim, act = nn.Mish(inplace = True)),
                nn.Linear(config.mlp_hidden_dim, config.num_bins, bias = True)
            )
            for _ in range(config.num_Qs)
        ])
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
        self.layers = nn.Sequential(*TDMPC2ObservationEncoder._create_layers(config))

    def _create_layers(config: TDMPC2Config) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(NormedLinear(config.input_dim, config.encoder_hidden_dim, act = nn.Mish(inplace = True)))
        for _ in range(config.encoder_num_layers - 2):
            layers.append(NormedLinear(config.encoder_hidden_dim, config.encoder_hidden_dim, act = nn.Mish(inplace = True)))
        layers.append(NormedLinear(config.encoder_hidden_dim, config.latent_dim, act = SimNorm(V = config.simplex_dim)))
        return layers
    
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) 

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
    def __init__(self, modules: Optional[Iterable[nn.Module]]):
        super().__init__()
        self.base_model = copy.deepcopy(modules[0]).to("meta")
        self.modules = nn.ModuleList(modules)
        self.params, self.buffers = torch.func.stack_module_state(modules)
        self.vmap = torch.vmap(self._call_single_model, in_dims=(0, 0, None), randomness="different")

    def _call_single_model(self, params, buffers, *args):
        return torch.func.functional_call(self.base_model, (params, buffers), args)
    
    def forward(self, *args):
        return self.vmap(self.params, (self.buffers), *args)