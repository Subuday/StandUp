from datetime import datetime as dt

import torch

from env_dm_control import make_env
from tdmpc2 import TDMPC2Policy, TDMPC2Config
from utils import set_global_seed

def to_episode_info(observation):
    pass

def train(
    env,
    policy,
    config,
):
    policy.train()

    buffer = []

    step = 0
    collected_episode_info = []
    episode_done = True
    while True:
        if step == config.training_steps:
            break

        if episode_done:
            if len(collected_episode_info) > 0:
                buffer.append(collected_episode_info)
            
            policy.reset()
            observation = env.reset()
            collected_episode_info = [to_episode_info(observation)]

        if step > config.buffer_seed_size:
            action = env.rand_act()
            # action = policy.select_action(observation)
        else:
            action = env.rand_act()
        observation, reward, episode_done, info = env.step(action)
        collected_episode_info.append(to_episode_info(observation))
            
        print("Colllecting seed data... ", step)

        if step < config.buffer_seed_size:
            step += 1
            continue

        if step == config.buffer_seed_size:
            iters = config.buffer_seed_size
        else:
            iters = 1
        
        for _ in range(iters):
            pass

        if iters > 1:
            print("training with seed data... ", step)
        else:
            print("training with new data... ", step)

        step += 1



    batch = {
        "observations": torch.randn(4, 256, 51),
        "actions": torch.randn(3, 256, 19),
        "reward": torch.randn(3, 256, 1)
    } 

    # dict = torch.load("./cup-spin.pt")["metadata"]
    # print(dict)
    config = TDMPC2Config()
    policy = TDMPC2Policy(config = config)
    policy.load("./cup-spin.pt")
    # policy(batch)    

def main():
    out_dir = f"outputs/train/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}"
    
    config = TDMPC2Config()
    
    set_global_seed(config.seed)

    env = make_env(config)

    policy = TDMPC2Policy(config).to("mps")

    train(env, policy, config)

if __name__ == "__main__":
    main()