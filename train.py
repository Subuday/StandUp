from datetime import datetime as dt
from common.logger import Logger

from tensordict import TensorDict
import torch

from buffer import Buffer
from env_dm_control import make_env
from tdmpc2 import TDMPC2Policy, TDMPC2Config
from utils import get_device_from_parameters, set_global_seed

def to_episode_info(env, observation, action = None, reward = None):
    """Creates a TensorDict for a new episode."""
    if isinstance(observation, dict):
        observation = TensorDict(observation, batch_size=(), device='cpu')
    else:
        observation = observation.unsqueeze(0).cpu()
    if action is None:
        action = torch.full_like(env.rand_act(), float('nan'))
    if reward is None:
        reward = torch.tensor(float('nan'))
    info = TensorDict(dict(
        obs=observation,
        action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
    ), batch_size=(1,))
    return info

def train(
    config,
    env,
    buffer,
    policy,
    logger: Logger,
):
    device = get_device_from_parameters(policy)

    step = 0
    collected_episode_info = []
    episode_done = True
    while True:
        if step == config.training_steps:
            break

        policy.eval()

        if episode_done:
            if len(collected_episode_info) > 0:
                buffer.add(torch.cat(collected_episode_info))
            
            policy.reset()
            observation = env.reset()
            collected_episode_info = [to_episode_info(env, observation)]

        if step > config.buffer_seed_size:
            action = policy.select_action(observation.unsqueeze(0).to(device)).cpu()
        else:
            action = env.rand_act()
        observation, reward, episode_done, _ = env.step(action)
        collected_episode_info.append(to_episode_info(env, observation, action, reward))

        if step < config.buffer_seed_size:
            step += 1
            continue

        if step == config.buffer_seed_size:
            iters = config.buffer_seed_size
        else:
            iters = 1
        
        policy.train()
        for _ in range(iters):
            sampled_observations, sampled_actions, sampled_reward, _ = buffer.sample()
            batch = {
                "observations": sampled_observations,
                "actions": sampled_actions,
                "reward": sampled_reward,
            }
            train_info = policy(batch)
            print("Step: ", step)
            print("Loss: ", train_info["loss"])
            print("Consistency loss: ", train_info["consistency_loss"])
            print("Reward loss: ", train_info["reward_loss"])
            print("Q value loss: ", train_info["q_value_loss"])

        step += 1
    
    logger.save_checkpoint(identifier = dt.now().strftime('%H-%M-%S'), policy = policy)

def main():
    config = TDMPC2Config()
    
    set_global_seed(config.seed)

    env = make_env(config)

    buffer = Buffer(config)

    policy = TDMPC2Policy(config).to("mps")

    logger = Logger(log_dir = "./out/")

    train(config, env, buffer, policy, logger)

if __name__ == "__main__":
    main()