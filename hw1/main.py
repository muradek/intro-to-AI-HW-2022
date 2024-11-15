from Robot import *
from MazeProblem import *
from Animation import Animation
from Heuristics import *
from Utilities import *
from Experiments import *


if __name__ == "__main__":
    # test_robot(BreadthFirstSearchRobot, [0, 1, 2, 3, 4, 5])
    # solve_and_display(BreadthFirstSearchRobot, 5)

    # test_robot(UniformCostSearchRobot, [0, 1, 2, 3, 4, 5])
    # solve_and_display(UniformCostSearchRobot, 3)

    # test_robot(WAStartRobot, [0, 1, 2, 3, 4, 5], heuristic=tail_manhattan_heuristic)
    # solve_and_display(WAStartRobot, 5, heuristic=tail_manhattan_heuristic)
    # test_robot(WAStartRobot, [99], heuristic=tail_manhattan_heuristic)
    # test_robot(WAStartRobot, [99], heuristic=center_manhattan_heuristic)
    # solve_and_display(WAStartRobot, 99, heuristic=tail_manhattan_heuristic)
    # solve_and_display(WAStartRobot, 99, heuristic=center_manhattan_heuristic)

    # test_robot(WAStartRobot, [0, 1, 2, 3, 4, 5], heuristic=center_manhattan_heuristic)
    # solve_and_display(WAStartRobot, 5, heuristic=center_manhattan_heuristic)

    # w_experiment(0)
    # w_experiment(1)
    # w_experiment(2)

    # test_robot(UniformCostSearchRobot, [155, 153])
    # solve_and_display(UniformCostSearchRobot, 155)
    # solve_and_display(UniformCostSearchRobot, 153)

    # test_robot(WAStartRobot, [0, 1, 2, 3, 4, 5], heuristic=ShorterRobotHeuristic, k=2)
    # solve_and_display(WAStartRobot, 3, heuristic=ShorterRobotHeuristic, k=8)
    for k in range(2, 10, 2):
        test_robot(WAStartRobot, [3, 4], heuristic=ShorterRobotHeuristic, k=k)

    # test_robot(WAStartRobot, [0, 1, 2, 3, 4, 5], heuristic=ShorterRobotHeuristic, k=0)
    # test_robot(WAStartRobot, [0, 1, 2, 3, 4, 5], heuristic=ShorterRobotHeuristic, k=2)
    # test_robot(WAStartRobot, [2, 3, 4, 5], heuristic=ShorterRobotHeuristic, k=4)
    # test_robot(WAStartRobot, [2, 3, 4, 5], heuristic=ShorterRobotHeuristic, k=6)
    # test_robot(WAStartRobot, [2, 3, 4, 5], heuristic=ShorterRobotHeuristic, k=8)
    # test_robot(WAStartRobot, [3], heuristic=ShorterRobotHeuristic, k=10)

    # for i in [1, 10, 100, 1000, 10000]:
    #     print(f'running n={i} with center_manhatten_heuristic:')
    #     test_robot(WAStartRobot, [f'n={i}'], heuristic=center_manhattan_heuristic)
    #     print()
    #
    #     print(f'running n={i} with ShorterRobotHeuristic, k=2:')
    #     test_robot(WAStartRobot, [f'n={i}'], heuristic=ShorterRobotHeuristic, k=2)
    #     print()

    # shorter_robot_heuristic_experiment(2)
    # shorter_robot_heuristic_experiment(3)
    # shorter_robot_heuristic_experiment(4)
    # shorter_robot_heuristic_experiment(5)
    # solve_and_display(WAStartRobot, 5, heuristic=ShorterRobotHeuristic, k=0)


    # solve_and_display(WAStartRobot, 1611, heuristic=center_manhattan_heuristic)
    # solve_and_display(WAStartRobot, 1612, heuristic=center_manhattan_heuristic)