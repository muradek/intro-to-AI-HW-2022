"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""

from SearchAlgos import AlphaBeta
from utils import *
from players.HeavyABPlayer import MAX_DEPTH as HEAVY_MAX_DEPTH
from players.AbstractPlayer import AbstractPlayer


EXPERIMENT_i = 3

class Player(AbstractPlayer):
    def _set_player_specific_move(self, cutoff_time):
        algo = AlphaBeta(light_utility_function, successor, goal=is_terminal_state)
        _, self.cur_move = algo.search(self.get_state(), EXPERIMENT_i - 1 + HEAVY_MAX_DEPTH, maximizing_player=True)
        return self.cur_move
