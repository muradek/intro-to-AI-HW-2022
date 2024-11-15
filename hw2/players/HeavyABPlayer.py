"""
MiniMax Player with AlphaBeta pruning with heavy heuristic
"""
from SearchAlgos import AlphaBeta
from utils import *
from players.AbstractPlayer import AbstractPlayer


MAX_DEPTH = 3


class Player(AbstractPlayer):
    def _set_player_specific_move(self, cutoff_time):
        algo = AlphaBeta(heavy_utility_function, successor, goal=is_terminal_state)
        _, self.cur_move = algo.search(self.get_state(), MAX_DEPTH, maximizing_player=True)
        return self.cur_move
