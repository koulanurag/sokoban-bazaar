"""
Solver for  gym-sokoban (https://github.com/mpSchrader/gym-sokoban)
"""
import gym
import numpy as np
import os
import random
from pyperplan.planner import HEURISTICS, search_plan, SEARCHES

OBJECT_NAME_TO_IDX = {
    "wall": 0,
    "floor": 1,
    "box_target": 2,
    "box_off_target": 3,
    "box_on_target": 4,
    "player_off_target": 5,
    "player_on_target": 6,
}

OBJECT_IDX_TO_NAME = {v: k for k, v in OBJECT_NAME_TO_IDX.items()}

OBJECT_NAME_TO_COLOR = {
    "wall": (0, 0, 0),
    "floor": (243, 248, 238),
    "box_target": (254, 126, 125),
    "box_off_target": (142, 121, 56),
    "box_on_target": (254, 95, 56),
    "player_off_target": (160, 212, 56),
    "player_on_target": (219, 212, 56),
}


def symbolic_state(obs):
    height, width, channels = obs.shape
    _symbolic_obs = [[None for _ in range(width)] for _ in range(height)]

    # Room fixed: represents all not movable parts of the room
    # Room structure: represents the current state of the room including movable parts
    info = {
        "room_fixed": np.zeros(shape=(height, width), dtype=int),
        "room_state": np.zeros(shape=(height, width), dtype=int),
        "true_state": np.zeros(shape=(height, width), dtype=int),
        "box_mapping": dict(),
    }

    for row_i in range(height):
        for col_i in range(width):
            _object_color = tuple(obs[row_i, col_i, :])

            if _object_color == (0, 0, 0):
                _object_name = "wall"
            elif _object_color == (243, 248, 238):
                _object_name = "floor"
            elif _object_color == (254, 126, 125):
                _object_name = "box_target"
            elif _object_color == (142, 121, 56):
                _object_name = "box_off_target"
            elif _object_color == (254, 95, 56):
                _object_name = "box_on_target"
            elif _object_color == (160, 212, 56):
                _object_name = "player_off_target"
            elif _object_color == (219, 212, 56):
                _object_name = "player_on_target"
            else:
                raise ValueError(
                    f"{_object_color}  not found \nonly 'tiny_rgb_array' are supported"
                )

            _symbolic_obs[row_i][col_i] = _object_name
            info["true_state"][row_i][col_i] = OBJECT_NAME_TO_IDX[_object_name]

            info["box_mapping"][(row_i, col_i)] = (row_i, col_i)

            info["room_fixed"][row_i][col_i] = OBJECT_NAME_TO_IDX[_object_name]
            if _object_name in [
                "player_off_target",
                "player_on_target",
                "box_off_target",
                "box_on_target",
            ]:
                info["room_fixed"][row_i][col_i] = OBJECT_NAME_TO_IDX["floor"]

            info["room_state"][row_i][col_i] = OBJECT_NAME_TO_IDX[_object_name]
            if _object_name in ["box_off_target"]:
                info["room_state"][row_i][col_i] = OBJECT_NAME_TO_IDX["box_on_target"]

    return np.array(_symbolic_obs), info


def symbolic_state_to_tiny_rgb(symbolic_state):
    assert symbolic_state.shape == (10, 10)
    img = np.zeros((10, 10, 3))
    height, width = 10, 10
    for row_i in range(height):
        for col_i in range(width):
            object_name = OBJECT_IDX_TO_NAME[symbolic_state[row_i][col_i]]
            object_color = OBJECT_NAME_TO_COLOR[object_name]
            img[row_i, col_i, :] = object_color

    return img


class PDDL:
    def __init__(self, symbolic_obs, problem_name, domain_name, domain_pddl_path):
        self.problem_name = problem_name
        self.domain_name = domain_name
        self.objects = self._process_objects(symbolic_obs)
        self.init_state = self._process_init_state(symbolic_obs)
        self.goal_state = self._generate_goal()
        self._domain_pddl_path = domain_pddl_path

    def save(self, file_path):
        with open(file_path, "w") as pddl_file:
            pddl_file.write(str(self))

    def __str__(self):
        _txt = f"(define(problem {self.problem_name}) \n"
        _txt += f"  (:domain {self.domain_name}) \n"

        # object definition
        _txt += f"  (:objects \n"
        for object_name, object_values in self.objects.items():
            for object_value in object_values:
                _txt += f"    {object_value} - {object_name}\n"
        _txt += f"  \n)\n"

        # initialization
        _txt += f"  (:init \n"
        for value in self.init_state:
            _txt += f"  ({value})\n"
        _txt += f"  )\n"

        # goal state
        _txt += f"  (:goal (and\n"
        for value in self.goal_state:
            _txt += f"  ({value})\n"
        _txt += f"  ))\n"

        # close
        _txt += f")\n"

        return _txt

    def _process_objects(self, symbolic_obs):
        box_counter, player_counter = 0, 0
        _objects = {
            "direction": ["dir-left", "dir-right", "dir-up", "dir-down"],
            "stone": [],
            "player": [],
            "location": [],
        }

        for row in range(symbolic_obs.shape[0]):
            for col in range(symbolic_obs.shape[1]):
                _objects["location"].append(f"pos-{col + 1}-{row + 1}")
                if symbolic_obs[row][col] == "player_off_target":
                    player_counter += 1
                    _objects["player"].append(f"player-0{player_counter}")
                elif symbolic_obs[row][col] == "box_target":
                    box_counter += 1
                    _objects["stone"].append(f"stone-0{box_counter}")

        return _objects

    def _process_init_state(self, symbolic_obs):
        goal, non_goal, clear = [], [], []
        player_location, box_location, move_directions = [], [], []

        player_counter = 0
        box_counter = 0

        for row in range(symbolic_obs.shape[0]):
            for col in range(symbolic_obs.shape[1]):
                # object location specification
                if symbolic_obs[row][col] == "box_target":
                    goal.append(f"IS-GOAL pos-{col + 1}-{row + 1}")
                else:
                    non_goal.append(f"IS-NONGOAL pos-{col + 1}-{row + 1}")

                    if symbolic_obs[row][col] == "player_off_target":
                        player_counter += 1
                        player_location.append(
                            f"at player-0{player_counter} pos-{col + 1}-{row + 1}"
                        )
                    elif symbolic_obs[row][col] == "box_off_target":
                        box_counter += 1
                        box_location.append(
                            f"at stone-0{box_counter} pos-{col + 1}-{row + 1}"
                        )

                # clear specification
                if (
                    symbolic_obs[row][col] != "wall"
                    and symbolic_obs[row][col] != "player_off_target"
                    and symbolic_obs[row][col] != "player_on_target"
                    and symbolic_obs[row][col] != "box_off_target"
                    and symbolic_obs[row][col] != "box_on_target"
                ):
                    clear.append(f"clear pos-{col + 1}-{row + 1}")

                # legal move specification
                if symbolic_obs[row][col] != "wall":
                    up_cell, down_cell = (row - 1, col), (row + 1, col)
                    right_cell, left_cell = (row, col + 1), (row, col - 1)
                    if self._valid_transition(
                        symbolic_obs, state=(row, col), next_state=up_cell
                    ):
                        move_directions.append(
                            f"MOVE-DIR pos-{col + 1}-{row + 1} pos-{up_cell[1] + 1}-{up_cell[0] + 1} dir-up"
                        )
                    if self._valid_transition(
                        symbolic_obs, state=(row, col), next_state=down_cell
                    ):
                        move_directions.append(
                            f"MOVE-DIR pos-{col + 1}-{row + 1} pos-{down_cell[1] + 1}-{down_cell[0] + 1} dir-down"
                        )
                    if self._valid_transition(
                        symbolic_obs, state=(row, col), next_state=right_cell
                    ):
                        move_directions.append(
                            f"MOVE-DIR pos-{col + 1}-{row + 1} pos-{right_cell[1] + 1}-{right_cell[0] + 1} dir-right"
                        )
                    if self._valid_transition(
                        symbolic_obs, state=(row, col), next_state=left_cell
                    ):
                        move_directions.append(
                            f"MOVE-DIR pos-{col + 1}-{row + 1} pos-{left_cell[1] + 1}-{left_cell[0] + 1} dir-left"
                        )

        move_directions = np.unique(move_directions).tolist()
        init_state = (
            goal + non_goal + move_directions + player_location + box_location + clear
        )
        return init_state

    def _generate_goal(self):
        return [f"at-goal {v}" for v in self.objects["stone"]]

    @staticmethod
    def _valid_transition(symbolic_obs, state, next_state):
        return not (  # check bounds of grid dimension
            state[0] < 0
            or state[1] < 0
            or next_state[0] < 0
            or next_state[1] < 0
            or state[0] >= symbolic_obs.shape[0]
            or state[1] >= symbolic_obs.shape[1]
            or next_state[0] >= symbolic_obs.shape[0]
            or next_state[1] >= symbolic_obs.shape[1]
            # check for wall
            or symbolic_obs[next_state[0], next_state[1]] == "wall"
            or symbolic_obs[state[0], state[1]] == "wall"
        )

    @property
    def _task_pddl_dir(self):
        if not (os.environ.get("TASK_PDDL_DIR") is None):
            return os.environ.get("TASK_PDDL_DIR")
        else:
            import tempfile

            return tempfile.gettempdir()

    def search_plan(self, search_name="astar", heuristic="hff"):
        assert search_name in ["astar", "wastar", "gbf", "bfs", "ehs", "ids", "sat"]

        task_pddl_path = os.path.join(
            self._task_pddl_dir, f"task_{random.randint(0, 1e+10)}.pddl"
        )
        self.save(task_pddl_path)

        # generate plan
        search_fn = SEARCHES[search_name]
        heuristic_class = HEURISTICS[heuristic]

        if search_name in ["bfs", "ids", "sat"]:
            heuristic_class = None

        solution = search_plan(
            domain_file=self._domain_pddl_path,
            problem_file=task_pddl_path,
            search=search_fn,
            heuristic_class=heuristic_class,
            use_preferred_ops=heuristic == "hffpo",
        )

        return [self.translate_action(action) for action in solution]

    @staticmethod
    def translate_action(operator_action):
        action_name = operator_action.name.split(" ")[-1]
        if "up" in action_name:
            action = 1
        elif "down" in action_name:
            action = 2
        elif "left" in action_name:
            action = 3
        elif "right" in action_name:
            action = 4
        else:
            raise ValueError()
        return action


if __name__ == "__main__":
    import argparse
    from gym.envs.registration import register
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opr", default="train", choices=["train", "generate-offline-data"]
    )
    parser.add_argument(
        "--sokoban_domain_pddl_path",
        default=os.path.join(os.getcwd(), "assets", "sokoban_domain.pddl"),
    )
    parser.add_argument(
        "--generated_problem_pddl_dir",
        default=os.path.join(os.getcwd(), "generated-problems"),
    )
    parser.add_argument("--num-episodes", default=10, type=int)
    # process arguments
    args = parser.parse_args()
    os.makedirs(args.generated_problem_pddl_dir, exist_ok=True)

    register(
        id="Sokoban5x5-v0",
        entry_point="gym_sokoban.envs:SokobanEnv",
        kwargs={"dim_room": (5, 5), "max_steps": 200, "num_boxes": 2},
    )

    env = gym.make("Sokoban5x5-v0")
    success_counts = 0
    for seed in tqdm(range(args.num_episodes)):
        # seeding
        random.seed(seed)
        np.random.seed(seed)
        env.seed(seed)

        # task pddl formulation
        env.reset()
        _symbolic_state, _ = symbolic_state(env.render("tiny_rgb_array"))
        pddl = PDDL(
            _symbolic_state,
            domain_pddl_path=os.path.join(os.getcwd(), "assets", "sokoban_domain.pddl"),
            problem_name=f"task-{seed}",
            domain_name="sokoban",
        )
        task_pddl_path = os.path.join(
            args.generated_problem_pddl_dir, f"task-{seed}.pddl"
        )
        pddl.save(task_pddl_path)

        # search and execute plan
        plan = pddl.search_plan()
        if not (plan is None):
            for step_i, action in enumerate(plan):
                env.render(mode="human")
                _, reward, done, info = env.step(action)
            env.render(mode="human")

            if reward >= 10.0:
                print(f"Episode: #{seed + 1}: plan successfully executed")
                success_counts += 1
            else:
                print(f"Episode: #{seed + 1}: generated plan failed")
        else:
            print(f"Episode: #{seed + 1}: no solution found")

    print(f"Success Rate: {success_counts / args.num_episodes}")
    env.close()
