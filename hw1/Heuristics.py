import numpy as np
from MazeProblem import MazeState, MazeProblem, compute_robot_direction
from Robot import UniformCostSearchRobot
from GraphSearch import NodesCollection


def tail_manhattan_heuristic(state: MazeState):
    # TODO (EX 7.2), implement heuristic, delete exception

    # manhatten(x,y) = sum |x_i - y_i| for i in 1...dim(x)
    return np.sum(np.abs(state.tail - state.maze_problem.tail_goal)) * state.maze_problem.forward_cost


def center_manhattan_heuristic(state: MazeState):
    # TODO (EX 9.2), implement heuristic, delete exception

    # calculate centers of robot and goal
    center_pos = (state.head + state.tail) // 2 + 1
    center_goal = (state.maze_problem.head_goal + state.maze_problem.tail_goal) // 2 + 1

    # manhatten(x,y) = sum |x_i - y_i| for i in 1...dim(x)
    return np.sum(np.abs(center_pos - center_goal)) * state.maze_problem.forward_cost


class ShorterRobotHeuristic:
    def __init__(self, maze_problem: MazeProblem, k):
        assert k % 2 == 0, "odd must be even"
        assert maze_problem.length - k >= 3, f"it is not possible to shorten a {maze_problem.length}-length robot by " \
                                             f"{k} units because robot length has to at least 3"
        self.k = k
        ################################################################################################################
        # TODO (EX. 13.2): replace all three dots, delete exception
        shorter_robot_head_goal, shorter_robot_tail_goal = self._compute_shorter_head_and_tails(
            maze_problem.head_goal, maze_problem.tail_goal  # use initial state head/tail as new goal
        )
        self.new_maze_problem = MazeProblem(maze_map=maze_problem.maze_map,
                                            initial_head=shorter_robot_tail_goal,  # swap directions
                                            initial_tail=shorter_robot_head_goal,
                                            head_goal=shorter_robot_head_goal,  # doesn't matter, don't change
                                            tail_goal=shorter_robot_tail_goal)  # doesn't matter, don't change
        self.node_dists = UniformCostSearchRobot().solve(self.new_maze_problem, compute_all_dists=True)
        ################################################################################################################

        assert isinstance(self.node_dists, NodesCollection)

    def _compute_shorter_head_and_tails(self, head, tail):
        # TODO (EX. 13.1): complete code here, delete exception
        robot_dir = compute_robot_direction(head, tail)
        new_head = head - robot_dir * self.k // 2  # pull head away from robot dir
        new_tail = tail + robot_dir * self.k // 2  # push tail towards robot dir
        return new_head, new_tail

    def __call__(self, state: MazeState):
        # TODO (EX. 13.3): replace each three dots, delete exception
        shorter_head_location, shorter_tail_location = self._compute_shorter_head_and_tails(state.head, state.tail)
        new_state = MazeState(self.new_maze_problem, head=shorter_tail_location, tail=shorter_head_location)
        if new_state in self.node_dists:
            node = self.node_dists.get_node(new_state)
            return node.g_value
        else:
            return float('inf')  # what should we return in this case, so that the heuristic would be as informative as possible
                        # but still admissible
