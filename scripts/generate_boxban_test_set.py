import time

import gym
import argparse

import numpy as np

import sokoban_bazaar
from sokoban_bazaar.utils import domain_pddl_path
import os
from pathlib import Path
from tqdm import tqdm
from sokoban_bazaar.solver import PDDL, symbolic_state
import pickle
from sokoban_bazaar.dataset import get_dataset
import random


def get_args():
    parser = argparse.ArgumentParser("Combinatorial Tasks with Decision Transformers ")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(Path.home(), ".sokoban-datasets")
    )

    # env-args
    env_args = parser.add_argument_group("data generation args")
    env_args.add_argument(
        "--env-name",
        default="gym_sokoban:Boxoban-Train-v0",
        choices=[
            "gym_sokoban:Boxoban-Train-v0",
        ],
        help="name of the environment",
    )
    # process arguments
    args = parser.parse_args()

    return args


def __main():
    args = get_args()

    # generate test-episodes
    env = gym.make("gym_sokoban:Boxoban-Test-v0")
    test_states = []
    _domain_pddl_path = domain_pddl_path(args.env_name)
    for episode_idx in tqdm(range(1000)):
        env.reset(map_idx=episode_idx)
        observation = env.render(mode='tiny_rgb_array')
        sym_state, info = symbolic_state(observation)
        pddl = PDDL(
            sym_state,
            domain_pddl_path=_domain_pddl_path,
            problem_name=f"task-{random.randint(0, int(1e+10))}",
            domain_name="sokoban",
        )
        start_time = time.time()
        plan = pddl.search_plan()
        time_taken = (time.time() - start_time)

        test_states.append({'state': sym_state,
                            'info': info,
                            'tiny_rgb_array': observation,
                            'plan': plan,
                            'plan-time': time_taken,
                            'map_idx': episode_idx})

    test_states = sorted(test_states, key=lambda ele: ele['plan-time'])
    os.makedirs(os.path.join(args.dataset_dir,
                           args.env_name), exist_ok=True)
    with open(os.path.join(args.dataset_dir,
                           args.env_name,
                           'test_states.p'), 'wb') as test_file:
        pickle.dump(test_states, test_file)


if __name__ == '__main__':
    __main()
