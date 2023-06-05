import pytest
from sokoban_bazaar.dataset import get_dataset, pad_batch, _ENV_NAMES, _DATASET_NAMES
from pytest_cases import parametrize, fixture_ref


@pytest.mark.parametrize('env_name,dataset_name',
                         [(_env_name, _dataset_name) for _dataset_name in
                          _DATASET_NAMES
                          for _env_name in _ENV_NAMES])
def test_dataset(env_name, dataset_name):
    from torch.utils.data import DataLoader

    episode_dataset = get_dataset(env_name, dataset_name)
    episode_dataloader = DataLoader(
        episode_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_batch,
    )

    for batch in episode_dataloader:
        assert all(k in batch.keys() for k in ["observations",
                                               "actions",
                                               "rewards",
                                               "returns_to_go",
                                               "timesteps"])
        break
