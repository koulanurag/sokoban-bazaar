import gym
import argparse

import numpy as np

import sokoban_bazaar
from sokoban_bazaar.utils import domain_pddl_path
import os
from pathlib import Path
from tqdm import tqdm
from sokoban_bazaar.solver import symbolic_state
import pickle
from sokoban_bazaar.dataset import get_dataset


def get_args():
    parser = argparse.ArgumentParser("Combinatorial Tasks with Decision Transformers ")
    parser.add_argument(
        "--dataset-dir", default=os.path.join(Path.home(), ".sokoban-datasets")
    )

    # env-args
    env_args = parser.add_argument_group("data generation args")
    env_args.add_argument(
        "--env-name",
        default="gym_sokoban:Sokoban-small-v1",
        choices=[
            "Sokoban5x5-v0",
            "gym_sokoban:Sokoban-small-v0",
            "gym_sokoban:Sokoban-small-v1",
            "gym_sokoban:Sokoban-v2",
            "gym_sokoban:Sokoban-v1",
            "gym_sokoban:Sokoban-v0",
            "gym_sokoban:Sokoban-large-v0",
            "gym_sokoban:Sokoban-large-v1",
            "gym_sokoban:Sokoban-large-v2",
            "gym_sokoban:Sokoban-huge-v1",
        ],
        help="name of the environment",
    )
    # process arguments
    args = parser.parse_args()

    return args


def __main():
    args = get_args()
    episode_dataset, _ = get_dataset(args.env_name, "expert")

    # generate test-episodes
    test_episodes = 50
    env = gym.make(args.env_name)
    test_states = {}
    while len(test_states.keys()) < test_episodes:
        env.reset()

        observation = env.render(mode="tiny_rgb_array")
        sym_state, info = symbolic_state(observation)
        _key = tuple(sym_state.flatten())
        if _key not in test_states:
            test_states[_key] = {
                "state": sym_state,
                "info": info,
                "tiny_rgb_array": observation,
            }

    with open(
        os.path.join(args.dataset_dir, args.env_name, "test_states.p"), "wb"
    ) as test_file:
        pickle.dump(test_states, test_file)


if __name__ == "__main__":
    __main()
