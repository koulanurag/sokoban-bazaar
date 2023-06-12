import pytest
from sokoban_bazaar.utils import set_state
import gym


@pytest.mark.parametrize('env_name', ["gym_sokoban:Sokoban-small-v0",
                                      "gym_sokoban:Sokoban-small-v1",
                                      "Sokoban5x5-v0",
                                      "gym_sokoban:Sokoban-v2"])
def test_set_state(env_name):
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    for _ in range(10):
        env.reset()
        eval_env = set_state(eval_env, env.render(mode='tiny_rgb_array'))

        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, _, done = env.step(action)
            eval_obs, _, _, eval_done = eval_env.step(action)

            assert done == eval_done
            assert (obs == eval_obs).all(), 'observation don\'t match'
