from gym.envs.registration import register

# Register custom sokoban environment
register(
    id="Sokoban5x5-v0",
    entry_point="gym_sokoban.envs:SokobanEnv",
    kwargs={"dim_room": (5, 5), "max_steps": 200, "num_boxes": 2},
)
