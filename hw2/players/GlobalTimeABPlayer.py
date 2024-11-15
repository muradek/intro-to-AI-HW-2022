"""
MiniMax Player with AlphaBeta pruning and global time
"""
from players.AlphabetaPlayer import Player as AlphabetaPlayer
from SearchAlgos import AlphaBeta, iterative_deepening_game_search
from utils import *
import matplotlib.pyplot as plt

MAX_NUM_OF_TURNS = 25
LEFTOVER_TIME_LIMIT = 0.5

class Player(AlphabetaPlayer):
    def __init__(self, game_time):
        AlphabetaPlayer.__init__(self, game_time) # keep the inheritance of the parent's (AbstractPlayer) __init__()

        # set time limit per turn as a linear function of the current turn
        linspace = np.linspace(1, 0, MAX_NUM_OF_TURNS, endpoint=False)  # sample linearly from 1 to 0 (linear decay)
        time_limit_frac = linspace / np.sum(linspace)  # normalize to sum to 1
        self.time_limit_per_turn = game_time * time_limit_frac  # num turn limits sum to approximately game time limit

        self.player_turn = 0  # index for time limits
        self.out_of_limits = False  # boolean for easy check

    def make_move(self, time_limit):
        # ignore given time limit. use pre-calculated if possible
        if not self.out_of_limits:
            time_limit = self.time_limit_per_turn[self.player_turn]
        else:  # if done with pre-calculated, use default short time until the end
            time_limit = LEFTOVER_TIME_LIMIT

        print(f'{time_limit=}')

        return super(Player, self).make_move(time_limit - TIDY_UP_TIME)

    def set_player_move(self, move):
        # set move as usual
        super(Player, self).set_player_move(move)

        # update player taken turns
        self.player_turn += 1
        if self.player_turn >= self.time_limit_per_turn.size:
            self.out_of_limits = True
