"""Search Algos: MiniMax, AlphaBeta
"""
from itertools import count
from collections import defaultdict
from utils import *
import time
import numpy as np
ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf # !!!!!

class SearchAlgos:
    def __init__(self, utility, succ, perform_move=None, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to,
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.goal = goal

    def search(self, state, depth, maximizing_player, cutoff_time=None) -> tuple:
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player, cutoff_time=None):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if cutoff_time is not None and cutoff_time <= time.time():
            raise TimeoutError

        # check end of state or search depth
        if self.goal(state) or depth == 0:
            return self.utility(state), None

        # set extreme func determined by player type (min / max)
        minimax_val, extreme_func = (-np.inf, max) if maximizing_player else (np.inf, min)

        # iterate successors to find best move in sub-tree
        best_move = None
        for suc, suc_move in self.succ(state):
            suc_minimax_value = self.search(suc, depth - 1, not maximizing_player, cutoff_time=cutoff_time)[0]

            minimax_val, best_move = extreme_func(
                [
                    (minimax_val, best_move),  # cur node minimax val and move
                    (suc_minimax_value, suc_move)  # next node minimax val and move
                ],
                key=lambda x: x[0]  # extreme determined by minimax val
            )

        return minimax_val, best_move


class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT, cutoff_time=None):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if cutoff_time is not None and cutoff_time <= time.time():
            raise TimeoutError

        # check end of state or search depth
        if self.goal(state) or depth == 0:
            return self.utility(state), None

        # set extreme func determined by player type (min / max)
        minimax_val, extreme_func = (-np.inf, max) if maximizing_player else (np.inf, min)

        # iterate successors to find best move in sub-tree
        best_move = None
        for suc, suc_move in self.succ(state):
            assert suc.check_valid()
            suc_minimax_value = self.search(suc, depth - 1, not maximizing_player, alpha=alpha, beta=beta,
                                            cutoff_time=cutoff_time)[0]

            minimax_val, best_move = extreme_func(
                [
                    (minimax_val, best_move),  # cur node minimax val and move
                    (suc_minimax_value, suc_move)  # next node minimax val and move
                ],
                key=lambda x: x[0]  # extreme determined by minimax val
            )

            # do pruning according to player type
            if maximizing_player:
                if minimax_val >= beta:
                    break  # prune maxplayer
                alpha = max(alpha, minimax_val)
            else:
                if minimax_val <= alpha:
                    break  # prune minplayer
                beta = min(beta, minimax_val)

        return minimax_val, best_move


class AlphaBetaTable(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT, table=None,
               cutoff_time=None):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        if cutoff_time is not None and cutoff_time <= time.time():
            raise TimeoutError

        if table is None:
            table = {}  # instantiate in first recursive iteration

        # if stored depth is larger, previous was found higher in the tree and is more informative.
        if state in table:
            tabled_depth, tabled_minimax_value, tabled_move = table[state]
            if tabled_depth > depth:
                return tabled_minimax_value, tabled_move

        # check end of state or search depth
        if self.goal(state) or depth == 0:
            return self.utility(state), None

        # set extreme func determined by player type (min / max)
        minimax_val, extreme_func = (-np.inf, max) if maximizing_player else (np.inf, min)

        # iterate successors to find best move in sub-tree
        best_move = None
        for suc, suc_move in self.succ(state):
            assert suc.check_valid()
            suc_minimax_value = self.search(suc, depth - 1, not maximizing_player, alpha=alpha, beta=beta, table=table,
                                            cutoff_time=cutoff_time)[0]

            # add newly calculated value to table if worthy
            if suc.killed_soldier():  # worthiness determined by killing a soldier
                table[suc] = (depth, suc_minimax_value, suc_move)

            minimax_val, best_move = extreme_func(
                [
                    (minimax_val, best_move),  # cur node minimax val and move
                    (suc_minimax_value, suc_move)  # next node minimax val and move
                ],
                key=lambda x: x[0]  # extreme determined by minimax val
            )

            # do pruning according to player type
            if maximizing_player:
                if minimax_val >= beta:
                    break  # prune maxplayer
                alpha = max(alpha, minimax_val)
            else:
                if minimax_val <= alpha:
                    break  # prune minplayer
                beta = min(beta, minimax_val)

        return minimax_val, best_move


def iterative_deepening_game_search(cutoff_time, state: State, algo: SearchAlgos):
    """
    performs iterative deepening search using the given algorithm.
    :param cutoff_time: When to stop calculating
    :param state: The current state of the game
    :param algo: a search algorithm
    :return: a move to perform
    """
    depth = 0
    try:
        for depth in count(1):
            _, state.player.cur_move = algo.search(state, depth, maximizing_player=True, cutoff_time=cutoff_time)
    except TimeoutError:
        print(f'{depth=}')
        pass

    return state.player.cur_move