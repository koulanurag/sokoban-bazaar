import pickle

from sokoban_bazaar.dataset import get_dataset, pad_batch, _ENV_NAMES, _DATASET_NAMES
from markdownTable import markdownTable
import pandas as pd
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np


def generate_dataset_stats():
    df = pd.DataFrame(columns=[])
    for env_name in _ENV_NAMES:
        for dataset_name in _DATASET_NAMES:
            try:
                episode_dataset = get_dataset(env_name, dataset_name)
                episode_dataloader = DataLoader(
                    episode_dataset,
                    batch_size=256,
                    sampler=SequentialSampler(range(len(episode_dataset))),
                    num_workers=4,
                    collate_fn=pad_batch,
                )

                episode_score = {'max': -np.inf, 'min': np.inf, 'mean': 0}
                episode_length = {'max': -np.inf, 'min': np.inf, 'mean': 0}
                for batch in episode_dataloader:
                    episode_score['max'] = max(episode_score['max'], batch['returns_to_go'][:, 0, 0].max().item())
                    episode_score['min'] = min(episode_score['min'], batch['returns_to_go'][:, 0, 0].min().item())
                    episode_score['mean'] += batch['returns_to_go'][:, 0, 0].mean().item()

                for file in episode_dataset.episode_files:
                    _len = len(pickle.load(open(file, 'rb'))['returns_to_go'])
                    episode_length['max'] = max(episode_length['max'], _len)
                    episode_length['min'] = min(episode_length['min'], _len)
                    episode_length['mean'] += _len

                episode_score['mean'] /= len(episode_dataloader)
                episode_length['mean'] /= len(episode_dataset.episode_files)

                new_row = pd.Series({
                    'env_name': env_name,
                    'dataset_name': dataset_name,
                    'episodes': len(episode_dataset),
                    **{f'{k}-score': v for k, v in episode_score.items()},
                    **{f'{k}-len': v for k, v in episode_length.items()}
                })
                df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            except Exception as e:
                print(e)

    return df


if __name__ == '__main__':
    stats_df = generate_dataset_stats()
    print(markdownTable(stats_df.to_dict(orient='records'))
          .setParams(row_sep='always')
          .getMarkdown())
