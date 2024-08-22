from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from utils import get_device_from_parameters

@dataclass
class TDMPC2Config:
    latent_dim: int = 50
    temporal_decay_coeff: float = 0.5

class TDMPC2Policy(nn.Module):
    def __init__(self, config: TDMPC2Config):
        self.config = config
        self.model = TDMPC2TOLD()

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
    def __init__(self):
        self._encoder = TDMPC2ObservationEncoder()
        self._dynamics = nn.Sequential()
        self._reward = nn.Sequential()
        self._pi = nn.Sequential()
        self._Qs = nn.Sequential()

    def encode(self, observation: Tensor) -> Tensor:
        pass

    def latent_dynamics(self, z: Tensor, a: Tensor) -> Tensor:
        pass

    def reward(self, z: Tensor, a: Tensor) -> Tensor:
        pass

    def Qs(self, z: Tensor, a: Tensor) -> Tensor:
        pass

    def pi(self, z: Tensor) -> Tensor:
        pass

class TDMPC2ObservationEncoder(nn.Module):
    def __init__(self):
        pass
