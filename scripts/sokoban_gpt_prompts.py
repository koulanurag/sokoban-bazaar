import gym
import numpy as np


def symbolic_state(obs):
    height, width, channels = obs.shape
    _symbolic_obs = [[None for _ in range(width)] for _ in range(height)]

    for row_i in range(height):
        for col_i in range(width):
            _object_color = tuple(obs[row_i, col_i, :])

            if _object_color == (0, 0, 0):
                _object_name = "#"
            elif _object_color == (243, 248, 238):
                _object_name = "_"
            elif _object_color == (254, 126, 125):
                _object_name = "X"
            elif _object_color == (142, 121, 56):
                _object_name = "@"
            elif _object_color == (254, 95, 56):
                _object_name = "box_on_target"
            elif _object_color == (160, 212, 56):
                _object_name = "$"
            elif _object_color == (219, 212, 56):
                _object_name = "player_on_target"
            else:
                raise ValueError("only 'tiny_rgb_array' are supported")

            _symbolic_obs[row_i][col_i] = _object_name

    return np.array(_symbolic_obs)


if __name__ == "__main__":
    env = gym.make("gym_sokoban:Sokoban-small-v0")
    env.reset()
    sym_state = symbolic_state(env.render(mode="tiny_rgb_array"))
    for row in sym_state:
        print("".join(row))

    solution = [
        1,
        5,
        5,
        3,
        3,
        2,
        6,
        4,
        8,
        1,
        1,
        1,
        4,
        8,
        8,
        1,
        1,
        1,
        3,
        3,
        3,
        2,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
    ]
    for action in solution:
        env.render()
        _, _, _, info = env.step(action)

    print(info)
