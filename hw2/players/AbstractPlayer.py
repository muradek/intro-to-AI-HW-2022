"""Abstract class of player.
Your players classes must inherit from this.
"""
import time

import utils
import numpy as np
import random

class AbstractPlayer:
    """Your player must inherit from this class.
    Your player class name must be 'Player', as in the given examples (SimplePlayer, LivePlayer).
    Use like this:
    from players.AbstractPlayer import AbstractPlayer
    class Player(AbstractPlayer):
    """
    def __init__(self, game_time):
        """
        Player initialization.
        """
        self.game_time = game_time
        self.board = np.zeros(utils.NUM_POS_ON_BOARD)
        self.directions = utils.get_directions

        # custom stuff
        self.my_pos = np.full(utils.MAX_NUM_SOLDIERS, utils.NO_SOLDIER)
        self.rival_pos = np.full(utils.MAX_NUM_SOLDIERS, utils.NO_SOLDIER)
        self.turn = 0
        self.cur_move = None

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array of the board.
        No output is expected.
        """
        self.board = board

    def make_move(self, time_limit):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, (pos, soldier, dead_opponent_pos)
        """
        # immediately get current time
        start_time = time.time()

        # set random initial move for player
        if self.turn < utils.STAGE_1_NUM_OF_TURNS:
            self.cur_move = self._stage_1_random_move()
        else:
            self.cur_move = self._stage_2_random_move()

        # update move with player algorithm
        self._set_player_specific_move(start_time + time_limit - utils.TIDY_UP_TIME)

        # set move in player's game tracker
        self.set_player_move(self.cur_move)

        return self.cur_move

    def _set_player_specific_move(self, cutoff_time):
        """make a player specific move"""
        raise NotImplementedError

    def check_valid_player_state(self):
        my_soldier_idx = self.my_pos[(self.my_pos != utils.NO_SOLDIER) & (self.my_pos != utils.DEAD_SOLDIER)]
        rival_soldier_idx = self.rival_pos[(self.rival_pos != utils.NO_SOLDIER) & (self.rival_pos != utils.DEAD_SOLDIER)]
        other_idx = list(set(range(utils.NUM_POS_ON_BOARD)) - (set(my_soldier_idx) | set(rival_soldier_idx)))
        return (
                np.all(self.board[my_soldier_idx] == utils.MY_ID_IN_BOARD) and
                np.all(self.board[rival_soldier_idx] == utils.RIVAL_ID_IN_BOARD) and
                np.all(self.board[other_idx] == utils.EMPTY_CELL_IN_BOARD)
        )

    def set_move(self, move, active_player_board_id, active_player_pos, passive_player_pos):
        """adapted from SimplePlayer's `set_rival_move` function"""
        assert move is not None, 'in goal state. no move needed!'

        active_pos, active_soldier, passive_killed = move

        if self.turn < utils.STAGE_1_NUM_OF_TURNS:  # stage 1
            self.board[active_pos] = active_player_board_id
            active_player_pos[active_soldier] = active_pos
        else:  # stage 2
            active_prev_pos = active_player_pos[active_soldier]
            self.board[active_prev_pos] = utils.EMPTY_CELL_IN_BOARD
            self.board[active_pos] = active_player_board_id
            active_player_pos[active_soldier] = active_pos

        if passive_killed != utils.NO_SOLDIER:  # soldier killed
            self.board[passive_killed] = utils.EMPTY_CELL_IN_BOARD
            dead_soldier = int(np.where(passive_player_pos == passive_killed)[0][0])
            passive_player_pos[dead_soldier] = utils.DEAD_SOLDIER

        # update turn
        self.turn += 1

        assert self.check_valid_player_state(), f'invalid new player state'

    def set_player_move(self, move):
        self.set_move(move, utils.MY_ID_IN_BOARD, self.my_pos, self.rival_pos)

    def set_rival_move(self, move):
        self.set_move(move, utils.RIVAL_ID_IN_BOARD, self.rival_pos, self.my_pos)

    def get_state(self, my_turn: bool = True):
        return utils.State(self, self.board.copy(), self.my_pos.copy(), self.rival_pos.copy(), self.turn, my_turn)

    def is_player(self, player, pos1, pos2, board=None):
        """
        Function to check if 2 positions have the player on them
        :param player: 1/2
        :param pos1: position
        :param pos2: position
        :return: boolean value
        """
        if board is None:
            board = self.board
        if board[pos1] == player and board[pos2] == player:
            return True
        else:
            return False

    def check_next_mill(self, position, player, board=None):
        """
        Function to check if a player can make a mill in the next move.
        :param position: curren position
        :param board: np.array
        :param player: 1/2
        :return:
        """
        if board is None:
            board = self.board
        mill = [
            (self.is_player(player, 1, 2, board) or self.is_player(player, 3, 5, board)),
            (self.is_player(player, 0, 2, board) or self.is_player(player, 9, 17, board)),
            (self.is_player(player, 0, 1, board) or self.is_player(player, 4, 7, board)),
            (self.is_player(player, 0, 5, board) or self.is_player(player, 11, 19, board)),
            (self.is_player(player, 2, 7, board) or self.is_player(player, 12, 20, board)),
            (self.is_player(player, 0, 3, board) or self.is_player(player, 6, 7, board)),
            (self.is_player(player, 5, 7, board) or self.is_player(player, 14, 22, board)),
            (self.is_player(player, 2, 4, board) or self.is_player(player, 5, 6, board)),
            (self.is_player(player, 9, 10, board) or self.is_player(player, 11, 13, board)),
            (self.is_player(player, 8, 10, board) or self.is_player(player, 1, 17, board)),
            (self.is_player(player, 8, 9, board) or self.is_player(player, 12, 15, board)),
            (self.is_player(player, 3, 19, board) or self.is_player(player, 8, 13, board)),
            (self.is_player(player, 20, 4, board) or self.is_player(player, 10, 15, board)),
            (self.is_player(player, 8, 11, board) or self.is_player(player, 14, 15, board)),
            (self.is_player(player, 13, 15, board) or self.is_player(player, 6, 22, board)),
            (self.is_player(player, 13, 14, board) or self.is_player(player, 10, 12, board)),
            (self.is_player(player, 17, 18, board) or self.is_player(player, 19, 21, board)),
            (self.is_player(player, 1, 9, board) or self.is_player(player, 16, 18, board)),
            (self.is_player(player, 16, 17, board) or self.is_player(player, 20, 23, board)),
            (self.is_player(player, 16, 21, board) or self.is_player(player, 3, 11, board)),
            (self.is_player(player, 12, 4, board) or self.is_player(player, 18, 23, board)),
            (self.is_player(player, 16, 19, board) or self.is_player(player, 22, 23, board)),
            (self.is_player(player, 6, 14, board) or self.is_player(player, 21, 23, board)),
            (self.is_player(player, 18, 20, board) or self.is_player(player, 21, 22, board))
        ]

        return mill[position]

    def is_mill(self, position, board=None):
        if board is None:
            board = self.board
        """
        Return True if a player has a mill on the given position
        :param position: 0-23
        :return:
        """
        if position < 0 or position > 23:
            return False
        p = int(board[position])

        # The player on that position
        if p != 0:
            # If there is some player on that position
            return self.check_next_mill(position, p, board)
        else:
            return False

    ###########################
    # Taken from RandomPlayer #
    ###########################

    def _choose_rival_cell_to_kill(self):
        rival_cell = int(np.random.choice(np.where(self.board == 2)[0], 1)[0]) # choose random cell
        return rival_cell

    def _stage_1_choose_cell_and_soldier_to_move(self):
        cell = int(np.random.choice(np.where(self.board == 0)[0], 1)[0]) # choose random cell
        soldier_that_moved = int(np.random.choice(np.where(self.my_pos == -1)[0], 1)[0]) # choose random soldier
        return cell, soldier_that_moved

    def _stage_1_random_move(self) -> tuple:
        cell, soldier_that_moved = self._stage_1_choose_cell_and_soldier_to_move()
        rival_cell = utils.NO_SOLDIER if not self.is_mill(cell) else self._choose_rival_cell_to_kill()
        return cell, soldier_that_moved, rival_cell

    def _stage_2_random_move(self) -> tuple:
        direction, soldier_that_moved = -1, -1

        while True:
            # choose random soldier_on_board to move
            soldier_cell = int(np.random.choice(np.where(self.board == 1)[0], 1)[0])
            # check if can move
            direction_list = self.directions(int(soldier_cell))
            valid_direction = [direction for direction in direction_list if self.board[direction] == 0]
            if len(valid_direction) == 0:
                continue
            direction = random.choice(valid_direction)
            soldier_that_moved = int(np.where(self.my_pos == soldier_cell)[0][0])

            # check if made mill
            rival_cell = utils.NO_SOLDIER if not self.is_mill(direction) else self._choose_rival_cell_to_kill()

            return direction, soldier_that_moved, rival_cell

        assert direction == -1, 'No moves'
