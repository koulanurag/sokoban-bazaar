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
        "--env",
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
        "--episode-start-idx",
        default=0,
        type=int,
        help="max number steps for data collection",
    )

    # process arguments
    args = parser.parse_args()


if __name__ == '__main__':
    generate_offline_dataset(
        env,
        args.domain_pddl_file_path,
        args.env_observation_mode,
        args.env_dataset_dir,
        max_episodes=args.max_episodes,
        episode_start_idx=args.episode_start_idx
    )

    # log to file
    print(f"Data generated and stored at {args.env_dataset_dir}")
