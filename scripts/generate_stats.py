import pickle

from sokoban_bazaar.dataset import (
    get_dataset,
    pad_episodic_batch,
    _ENV_NAMES,
    _DATASET_NAMES,
    get_test_envs,
)
from markdownTable import markdownTable
import pandas as pd
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from tqdm import tqdm


def generate_dataset_stats():
    train_df = pd.DataFrame(columns=[])
    eval_df = pd.DataFrame(columns=[])
    for env_name in ["gym_sokoban:Boxoban-Train-v0"]:
        for dataset_name in ["expert"]:
            print(env_name, dataset_name)
            try:
                episode_dataset, _ = get_dataset(env_name, dataset_name)
                episode_score = {"max": -np.inf, "min": np.inf, "mean": 0}
                episode_length = {"max": -np.inf, "min": np.inf, "mean": 0}

                for file in tqdm(episode_dataset.episode_files):
                    try:
                        with open(file, "rb") as _file:
                            _episode_info = pickle.load(_file)

                            _len = len(_episode_info["returns_to_go"])
                            episode_length["max"] = max(episode_length["max"], _len)
                            episode_length["min"] = min(episode_length["min"], _len)
                            episode_length["mean"] += _len

                            episode_score["max"] = max(
                                episode_score["max"], _episode_info["returns_to_go"][0]
                            )
                            episode_score["min"] = min(
                                episode_score["min"], _episode_info["returns_to_go"][0]
                            )
                            episode_score["mean"] += _episode_info["returns_to_go"][0]

                    except:
                        print(file)

                episode_score["mean"] /= len(episode_dataset.episode_files)
                episode_length["mean"] /= len(episode_dataset.episode_files)

                new_row = pd.Series(
                    {
                        "env_name": env_name,
                        "dataset_name": dataset_name,
                        "episodes": len(episode_dataset),
                        **{f"{k}-score": v for k, v in episode_score.items()},
                        **{f"{k}-len": v for k, v in episode_length.items()},
                    }
                )
                train_df = pd.concat(
                    [train_df, new_row.to_frame().T], ignore_index=True
                )
            except Exception as e:
                print(e)

        new_row = pd.Series(
            {
                "env_name": env_name,
                "num-test-states": len(get_test_envs(env_name)),
            }
        )
        eval_df = pd.concat([eval_df, new_row.to_frame().T], ignore_index=True)

    return train_df, eval_df


if __name__ == "__main__":
    stats_df, eval_df = generate_dataset_stats()
    print(
        markdownTable(stats_df.to_dict(orient="records"))
        .setParams(row_sep="always")
        .getMarkdown()
    )
    print("\n\n")
    print(
        markdownTable(eval_df.to_dict(orient="records"))
        .setParams(row_sep="always")
        .getMarkdown()
    )
