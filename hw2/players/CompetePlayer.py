"""
MiniMax Player with AlphaBeta pruning and global time
"""
from SearchAlgos import AlphaBetaTable, iterative_deepening_game_search
from players.GlobalTimeABPlayer import Player as GlobalTimeABPlayer
from utils import *

MAX_NUM_OF_TURNS = 25
LEFTOVER_TIME_LIMIT = 0.5


class Player(GlobalTimeABPlayer):
    def _set_player_specific_move(self, cutoff_time):
        # same as global time AB player, but with table to remember state values
        algo = AlphaBetaTable(heavy_utility_function, successor, goal=is_terminal_state)
        iterative_deepening_game_search(cutoff_time, self.get_state(), algo)
        return self.cur_move
