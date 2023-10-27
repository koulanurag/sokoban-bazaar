import pytest
from sokoban_bazaar.dataset import _ENV_NAMES, _DATASET_NAMES


# @pytest.mark.parametrize(
#    "env_name,dataset_name",
#    [
#        (_env_name, _dataset_name)
#        for _dataset_name in _DATASET_NAMES
#        for _env_name in ["Sokoban5x5-v0"]
#    ],
# )
# def test_dataset(env_name, dataset_name):
#    from torch.utils.data import DataLoader
#    from sokoban_bazaar.dataset import pad_episodic_batch
#
#    episode_dataset, weighted_sampler = get_dataset(env_name, dataset_name)
#    episode_dataloader = DataLoader(
#        episode_dataset,
#        batch_size=256,
#        num_workers=0,
#        sampler=weighted_sampler,
#        collate_fn=pad_episodic_batch,
#    )
#
#    for batch in episode_dataloader:
#        assert all(
#            k in batch.keys()
#            for k in [
#                "observations",
#                "actions",
#                "rewards",
#                "returns_to_go",
#                "timesteps",
#            ]
#        )
#        break


@pytest.mark.parametrize(
    "env_name, dataset_name",
    [
        (_env_name, _dataset_name)
        for _dataset_name in _DATASET_NAMES
        for _env_name in _ENV_NAMES
    ],
)
def test_trajectories(env_name, dataset_name):
    from sokoban_bazaar.dataset import get_trajectories

    episodes, _ = get_trajectories(env_name, dataset_name)
    for episode in episodes:
        for k in [
            "observations",
            "actions",
            "rewards",
            "dones",
            "timesteps",
            "symbolic_state",
            "returns_to_go",
        ]:
            assert k in episode.keys(), f"{k} not found in episode keys "
        break


@pytest.mark.parametrize(
    "env_name", [_env_name for _env_name in ["gym_sokoban:Boxoban-Train-v0"]]
)
def test_test_envs(env_name):
    from sokoban_bazaar.dataset import get_test_envs

    for test_env, env_info in get_test_envs(env_name):
        test_env.step(test_env.action_space.sample())
        test_env.close()
