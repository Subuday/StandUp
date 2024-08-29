from pathlib import Path
from datetime import datetime as dt
import imageio

from dm_control import suite
import torch
import numpy as np

from tdmpc2 import TDMPC2Policy, TDMPC2Config
from env_dm_control import make_env
from utils import get_device_from_parameters, set_global_seed


def rollout(
    env,
    policy,
    render_callback
):
    device = get_device_from_parameters(policy)

    # Reset the policy and environments.
    policy.reset()
    observation, reward, done = env.reset(), 0, False

    render_callback(env.render())

    all_actions_means = []
    all_actions = []
    all_rewards = []

    while not done:
        observation = observation.to(device).unsqueeze(0)
        
        with torch.inference_mode():
            action = policy.select_action(observation)
        
        # value = value.cpu().item()
        # action_mean = action_mean.cpu().item()
        # print(value)
        observation, reward, done, info = env.step(action)
        print(reward)
        
        render_callback(env.render())

        all_actions.append(action)
        all_rewards.append(reward)
        # all_actions_means.append(action_mean)

        
    print("Done!")

    # all_elite_actions = torch.stack(policy.all_elite_actions)
    # print(f"All elite actions mean: {all_elite_actions.mean()} - {all_elite_actions.std()}")
    # for dim in range(all_elite_actions.shape[3]):
    #     print(f"Elite action dim {dim} mean: {all_elite_actions[:,:, :, dim].mean()} - {all_elite_actions[:, :, :, dim].std()}")
    # print(all_elite_actions.shape)

    # all_observations = torch.stack(policy.all_observations)
    # for dim in range(all_observations.shape[1]):
    #     print(f"Dim {dim} - mean: {all_observations[:, dim].mean()} - {all_observations[:, dim].std()}")

    # print("Actions taken:")
    # print(all_actions[0].shape)
    res = {
        "actions": torch.stack(all_actions),
        "rewards": torch.stack(all_rewards)
    }
    # print(f"R: mean - {torch.mean(res['rewards'])}, std - {torch.std(res['rewards'])}")
    # print(f"A: mean - {torch.mean(res['actions'])}, std - {torch.std(res['actions'])}")
    # print(f"M: mean - {np.mean(all_actions_means)}, std - {np.std(all_actions_means)}")
    return res


def eval_policy(
    env,
    policy,
    n_episodes: int,
    videos_dir: Path
):
    policy.eval()

    # Callback for visualization.
    def render_frame(frame):
        ep_frames.append(frame)

    for i in range(n_episodes):
        ep_frames: list[np.ndarray] = []

        rollout_res = rollout(env, policy, render_frame)


        if len(ep_frames) > 0:
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = videos_dir / f"eval_episode_{i}.mp4"
            imageio.mimsave(video_path, ep_frames, fps = 15)

    print(f"R - {rollout_res['rewards'].sum(0)}")
    
def main():
    config = TDMPC2Config()

    out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}"

    set_global_seed(config.seed)

    env = make_env(config)

    policy = TDMPC2Policy(config).to("mps")
    policy.load(config.checkpoint)

    # for l in policy.model._encoder.state:
    #     print(f"Encoder - mean {l.weight.mean()} - {l.weight.std()}")
    # for p in policy.model._dynamics:
    #     print(f"Dynamics - mean {p.weight.mean()} - {p.weight.std()}")
    # for l in policy.model._reward:
    #     print(f"Reward - mean {l.weight.mean()} - {l.weight.std()}")
    # for p in policy.model._Qs.params:
    #     print(f"_Qs - mean {p.mean()} - {p.std()}") 
    # for p in policy.model._Qs_target.params:
    #     print(f"_Qs_target - mean {p.mean()} - {p.std()}")
    # for l in policy.model._pi:
    #     print(f"Pi - mean {l.weight.mean()} - {l.weight.std()}")
        

    with torch.no_grad():
        eval_policy(
            env,
            policy,
            1,
            videos_dir = Path(out_dir) / "videos",
        )
    env.close()


if __name__ == "__main__":
    main()