import gym
import argparse
import sokoban_bazaar
from sokoban_bazaar.dataset import generate_offline_dataset
from sokoban_bazaar.utils import domain_pddl_path
import os
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser("Combinatorial Tasks with Decision Transformers ")
    parser.add_argument(
        "--job",
        default="train",
        type=str,
        choices=["train", "eval", "generate-offline-data"],
        help="job to be performed",
    )

    # env-args
    env_args = parser.add_argument_group("data generation args")
    env_args.add_argument(
        "--env_name",
        default="gym_sokoban:Sokoban-small-v0",
        choices=[
            "gym_sokoban:Sokoban-small-v0",
            "gym_sokoban:Sokoban-small-v1",
            "gym_sokoban:Sokoban-v2",
            "Sokoban5x5-v0",
            "gym_sokoban:Sokoban-large-v0",
            "gym_sokoban:Sokoban-large-v1",
        ],
        help="name of the environment",
    )
    env_args.add_argument(
        "--env-observation-mode",
        default="tiny_rgb_array",
        choices=["rgb_array", "tiny_rgb_array"],
        help="render( image mode) for the environment",
    )

    # data-generation args
    data_generation_args = parser.add_argument_group("data generation args")
    data_generation_args.add_argument(
        "--max-episodes",
        default=1000,
        type=int,
        help="max number steps for data collection",
    )
    data_generation_args.add_argument(
        "--dataset-quality",
        default="expert",
        type=str,
        help="max number steps for data collection",
    )
    data_generation_args.add_argument(
        "--episode-start-idx",
        default=0,
        type=int,
        help="max number steps for data collection",
    )

    # process arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # ##################################################################################
    # Setup env
    # ##################################################################################
    env = gym.make(args.env_name)

    # ##################################################################################
    # Job: Generate Offline Data
    # ##################################################################################
    dataset_dir = os.path.join(Path.home(), ".sokoban-datasets", args.env_name,
                                 args.env_observation_mode, args.dataset_quality)
    os.makedirs(dataset_dir, exist_ok=True)
    generate_offline_dataset(
        env,
        domain_pddl_path(args.env_name),
        args.env_observation_mode,
        dataset_dir=dataset_dir,
        max_episodes=args.max_episodes,
        episode_start_idx=args.episode_start_idx
    )

    # log to file
    print(f"Data generated and stored at {args.env_dataset_dir}")
