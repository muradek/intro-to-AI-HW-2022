"""
MiniMax Player with AlphaBeta pruning
"""
from players.AbstractPlayer import AbstractPlayer
from utils import *
from SearchAlgos import AlphaBeta, iterative_deepening_game_search

class Player(AbstractPlayer):
    def _set_player_specific_move(self, cutoff_time):
        algo = AlphaBeta(medium_utility_function, successor, goal=is_terminal_state)
        iterative_deepening_game_search(cutoff_time, self.get_state(), algo)
        return self.cur_move