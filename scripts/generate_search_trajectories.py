import pickle
from pathlib import Path
from sokoban_bazaar.dataset import get_trajectories
import os
import argparse
from collections import defaultdict
import copy

root_path = os.environ.get(
    "SOKOBAN_DATASET_ROOT_PATH", default=os.path.join(Path.home(), ".sokoban-datasets")
)
from sokoban_bazaar.dataset import SokobanCustomResetEnv


def generate_search_trajectories(
    episode_start_idx, max_episodes, num_rollouts, max_rollout_steps
):
    env_name = "gym_sokoban:Boxoban-Train"
    dataset_name = "expert"
    save_dir = os.path.join(
        root_path, env_name, "tiny_rgb_array", dataset_name, "branched-trajectories"
    )
    os.makedirs(save_dir, exist_ok=True)
    episodes, _ = get_trajectories("gym_sokoban:Boxoban-Train-v0", "expert")
    env = SokobanCustomResetEnv(reset=False)
    for episode_idx, episode in enumerate(
        episodes[episode_start_idx : episode_start_idx + max_episodes]
    ):
        rollouts = defaultdict(lambda: [])
        for step_i, observation in enumerate(episode["observations"]):
            rollouts[step_i] = []
            for rollout_i in range(num_rollouts):
                env.reset(tiny_rgb_state=copy.deepcopy(observation).transpose(1, 2, 0))
                rollouts[step_i].append(defaultdict(lambda: []))
                for step in range(max_rollout_steps):
                    step_action = env.action_space.sample()
                    step_obs, step_reward, _, done = env.step(step_action)

                    rollouts[step_i][-1]["observations"].append(step_obs)
                    rollouts[step_i][-1]["rewards"].append(step_reward)
                    rollouts[step_i][-1]["actions"].append(step_action)
                    rollouts[step_i][-1]["dones"].append(done)
                    if done:
                        break

                rollouts[step_i][-1] = dict(rollouts[step_i][-1])
        rollouts = dict(rollouts)

        pickle.dump(
            {"episode": episode, "future-rollouts": rollouts},
            open(
                os.path.join(save_dir, f"episode_{episode_start_idx+episode_idx}.p"),
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
