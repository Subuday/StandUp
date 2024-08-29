from datetime import datetime as dt

import torch

from env_dm_control import make_env
from tdmpc2 import TDMPC2Policy, TDMPC2Config
from utils import set_global_seed

def train(
    env,
    policy,
    training_steps
):
    policy.train()

    step = 0
    while True:
        if step == training_steps:
            break

        
    while step < training_steps:
        pass


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

    train(env, policy, config.training_steps)

if __name__ == "__main__":
    main()