import pickle
from pathlib import Path
from sokoban_bazaar.dataset import get_trajectories
import os
import argparse
from collections import defaultdict
import copy
from sokoban_bazaar.solver import PDDL, symbolic_state

root_path = os.environ.get(
    "SOKOBAN_DATASET_ROOT_PATH", default=os.path.join(Path.home(), ".sokoban-datasets")
)
from sokoban_bazaar.dataset import SokobanCustomResetEnv
from tqdm import tqdm
import numpy as np


def generate_search_trajectories(
        episode_start_idx, max_episodes, num_rollouts, max_rollout_steps
):
    env_name = "gym_sokoban:Boxoban-Train-v0"
    dataset_name = "expert"
    save_dir = os.path.join(
        root_path, env_name, "tiny_rgb_array", dataset_name, "branched-trajectories"
    )
    os.makedirs(save_dir, exist_ok=True)
    episodes, _ = get_trajectories(env_name, dataset_name)
    env = SokobanCustomResetEnv(reset=False)
    for episode_idx, episode in enumerate(tqdm(
            episodes[episode_start_idx: episode_start_idx + max_episodes]
    )):
        rollouts = defaultdict(lambda: [])
        episode_score = 0

        for step_i in range(len(episode["timesteps"])):
            observation = episode["observations"][step_i]
            if step_i > 0:
                episode_score += episode["rewards"][step_i - 1]
            rollouts[step_i] = defaultdict(lambda: [])

            rollout_scores = []
            for rollout_i in range(num_rollouts):
                env.reset(tiny_rgb_state=copy.deepcopy(observation).transpose(1, 2, 0))
                for k in ["observations", "rewards", "actions",
                          "dones", "symbolic_state", "timesteps", "returns_to_go"]:
                    rollouts[step_i][k].append([])

                rollout_score = episode_score
                for rollout_step in range(max_rollout_steps):
                    step_action = env.action_space.sample()
                    step_obs, step_reward, done, info = env.step(step_action)
                    rollout_score += step_reward

                    obs_img = env.render(mode="tiny_rgb_array")
                    _, obs_info = symbolic_state(obs_img)

                    rollouts[step_i]["observations"][-1].append(obs_img.transpose(2, 0, 1))
                    rollouts[step_i]["rewards"][-1].append(step_reward)
                    rollouts[step_i]["actions"][-1].append(step_action)
                    rollouts[step_i]["dones"][-1].append(done)
                    rollouts[step_i]["timesteps"][-1].append(
                        rollout_step + 1 + episode["timesteps"][step_i]
                    )
                    rollouts[step_i]["symbolic_state"][-1].append(
                        obs_info["true_state"].flatten()
                    )

                    if done:
                        for _ in range(max_rollout_steps - rollout_step - 1):
                            rollouts[step_i]["observations"][-1].append(
                                rollouts[step_i]["observations"][-1][-1]
                            )
                            rollouts[step_i]["rewards"][-1].append(0)
                            rollouts[step_i]["actions"][-1].append(
                                rollouts[step_i]["actions"][-1][-1]
                            )
                            rollouts[step_i]["dones"][-1].append(True)
                            rollouts[step_i]["timesteps"][-1].append(
                                rollout_step + 1 + episode["timesteps"][step_i]
                            )
                            rollouts[step_i]["symbolic_state"][-1].append(
                                rollouts[step_i]["symbolic_state"][-1][-1]
                            )
                        break

                rollout_scores.append(rollout_score)

            _done_filter = np.array(rollouts[step_i]['dones'])[:, -1]
            rollout_scores = np.array(rollout_scores)
            rollout_scores[~_done_filter] += -18  # expected value of random policy
            returns_to_go = np.zeros((num_rollouts, step_i + max_rollout_steps))
            returns_to_go[:, 0] = rollout_scores
            for rtg_i in range(1, step_i + max_rollout_steps):
                if rtg_i <= step_i:
                    returns_to_go[:, rtg_i] = returns_to_go[:, rtg_i-1]-episode["rewards"][step_i]
                else:
                    returns_to_go[:, rtg_i] = returns_to_go[:, rtg_i-1]-rollouts[step_i]['rewards'][rtg_i-step_i-1]
            rollouts[step_i]['returns_to_go'] = returns_to_go
            rollouts[step_i] = dict(rollouts[step_i])
        rollouts = dict(rollouts)

        pickle.dump(
            {"episode": episode, "future-rollouts": rollouts},
            open(
                os.path.join(save_dir, f"episode_{episode_start_idx + episode_idx}.p"),
                "wb",
            ),
        )


def get_args():
    parser = argparse.ArgumentParser("Combinatorial Tasks with Decision Transformers ")
    parser.add_argument(
        "--job",
        default="train",
        type=str,
        choices=["train", "eval", "generate-offline-data"],
        help="job to be performed",
    )

    parser.add_argument(
        "--episode-start-idx",
        default=0,
        type=int,
        help="max number steps for data collection",
    )
    parser.add_argument(
        "--max-episodes",
        default=1000,
        type=int,
        help="max number steps for data collection",
    )
    parser.add_argument(
        "--num-rollouts",
        default=10,
        type=int,
        help="max number steps for data collection",
    )
    parser.add_argument(
        "--max-rollout-steps",
        default=10,
        type=int,
        help="max number steps for data collection",
    )

    # process arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    generate_search_trajectories(
        args.episode_start_idx,
        args.max_episodes,
        args.num_rollouts,
        args.max_rollout_steps,
    )
