import pytest
from pytest_cases import parametrize, fixture_ref


@pytest.mark.parametrize('env_name,dataset_name',
                         [(_env_name, _dataset_name) for _dataset_name in
                          ['random',
                           'expert',
                           'medium',
                           'medium-expert',
                           'expert-random']
                          for _env_name in ["gym_sokoban:Sokoban-small-v0",
                                            "gym_sokoban:Sokoban-small-v1",
                                            "gym_sokoban:Sokoban-v2",
                                            "Sokoban5x5-v0",
                                            "gym_sokoban:Sokoban-large-v0",
                                            "gym_sokoban:Sokoban-large-v1"]])
def test_dataset(env_name, dataset_name):
    from sokoban_bazaar.dataset import get_dataset, pad_batch
    from torch.utils.data import DataLoader

    episode_dataset = get_dataset(env_name, dataset_name)
    episode_dataloader = DataLoader(
        episode_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=10,
        collate_fn=pad_batch,
    )
