import gym
import argparse
import sokoban_bazaar
from sokoban_bazaar.utils import domain_pddl_path
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from sokoban_bazaar.solver import PDDL, symbolic_state
import random
import pickle

def generate_offline_dataset(
        env,
        domain_pddl_path,
        dataset_quality,
        env_observation_mode,
        dataset_dir,
        max_episodes=1,
        episode_start_idx=0,
):
    total_step_count = 0
    done = True
    episode_count = 0

    # seed env
    env.action_space.seed(episode_start_idx)
    env.seed(episode_start_idx)

    with tqdm(total=max_episodes) as pbar:
        while episode_count < max_episodes:

            # ##########################################################################
            # Reset episode
            # ##########################################################################
            if done:
                episode_step_i = 0
                episode_info = defaultdict(lambda: [])
                env.reset()
                obs_img = env.render(mode=env_observation_mode)
                sym_state = symbolic_state(env.render(mode="tiny_rgb_array"))
                pddl = PDDL(
                    sym_state,
                    domain_pddl_path=domain_pddl_path,
                    problem_name=f"task-{random.randint(0, int(1e+10))}",
                    domain_name="sokoban",
                )
                plan = pddl.search_plan()

            # ##########################################################################
            # Step into environment
            # ##########################################################################
            if dataset_quality == 'expert':
                action = plan.pop(0)  # pop action to be executed
            elif dataset_quality == 'random':
                action = env.action_space.sample()
            else:
                raise ValueError()

            _, reward, done, info = env.step(action)
            next_obs_img = env.render(mode=env_observation_mode)
            total_step_count += 1
            episode_step_i += 1

            # Convert image from (height, width, channel)
            # to (channel, height, width) for network compatibility.
            # Save rendered images as rendering allows different image modes
            episode_info["observations"].append(obs_img.transpose(2, 0, 1))
            episode_info["actions"].append(action)
            episode_info["rewards"].append(reward)
            episode_info["dones"].append(done)
            episode_info["timesteps"].append(episode_step_i)

            # ##########################################################################
            # Save episode
            # ##########################################################################
            if done:
                # store episode returns-to-go
                episode_info["returns_to_go"] = []
                return_to_go = 0
                for step_reward in episode_info["rewards"][::-1]:
                    return_to_go += step_reward
                    episode_info["returns_to_go"].append(return_to_go)
                episode_info["returns_to_go"] = episode_info["returns_to_go"][::-1]

                # store last observation
                episode_info["observations"].append(next_obs_img.transpose(2, 0, 1))

                pickle.dump(
                    dict(episode_info),
                    open(os.path.join(
                        dataset_dir,
                        f"episode_{episode_start_idx + episode_count}.p"),
                        "wb"
                    ),
                )
                episode_count += 1

                # update progress bar
                pbar.update(1)
                pbar.set_description(
                    f"Episodes: {episode_count} " f"Steps: {total_step_count}"
                )

            obs_img = next_obs_img


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
        "--dataset-dir",
        default=os.path.join(Path.home(), ".sokoban-datasets")
    )

    # env-args
    env_args = parser.add_argument_group("data generation args")
    env_args.add_argument(
        "--env-name",
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


def __main():
    args = get_args()

    env = gym.make(args.env_name)
    dataset_dir = os.path.join(args.dataset_dir, args.env_name,
                               args.env_observation_mode, args.dataset_quality)
    os.makedirs(dataset_dir, exist_ok=True)
    generate_offline_dataset(
        env=env,
        domain_pddl_path=domain_pddl_path(args.env_name),
        dataset_quality=args.dataset_quality,
        env_observation_mode=args.env_observation_mode,
        dataset_dir=dataset_dir,
        max_episodes=args.max_episodes,
        episode_start_idx=args.episode_start_idx
    )

    # log to file
    print(f"Data generated and stored at {dataset_dir}")


if __name__ == '__main__':
    __main()
