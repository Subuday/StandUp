import os
from pathlib import Path


class Logger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents = True, exist_ok = True)
        self.checkpoints_dir = self.get_checkpoints_dir(log_dir)

    @classmethod
    def get_checkpoints_dir(cls, log_dir: str | Path) -> Path:
        return Path(log_dir) / "checkpoints"
    
    def save_checkpoint(
        self,
        identifier: str, 
        policy,
    ):
        self.checkpoints_dir.mkdir(parents = True, exist_ok = True)
        policy.save(os.path.join(self.checkpoints_dir, f"{identifier}.pt"))