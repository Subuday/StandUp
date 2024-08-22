import torch

from tdmpc2 import TDMPC2Policy, TDMPC2Config

def train():
    batch = {
        "observations": torch.randn(4, 256, 51),
        "actions": torch.randn(3, 256, 19),
        "reward": torch.randn(3, 256, 1)
    } 
    policy = TDMPC2Policy(config=TDMPC2Config()) 
    policy(batch)

if __name__ == "__main__":
    train()