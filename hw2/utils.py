import operator
import time

import numpy as np
import os
from collections import namedtuple
# from SearchAlgos import SearchAlgos
# from itertools import count
from copy import copy

MINIMAL_NUM_SOLDIERS = 3

ALPHA_VALUE_INIT = -np.inf
BETA_VALUE_INIT = np.inf

# board values
EMPTY_CELL_IN_BOARD = 0
MY_ID_IN_BOARD = 1
RIVAL_ID_IN_BOARD = 2

# soldier position values
NO_SOLDIER = -1
DEAD_SOLDIER = -2
MAX_NUM_SOLDIERS = 9

# game constants
STAGE_1_NUM_OF_TURNS = 18
NUM_POS_ON_BOARD = 24
MILL_ROWS = [
    [0, 1, 2],
    [8, 9, 10],
    [16, 17, 18],
    [3, 11, 19],
    [20, 12, 4],
    [21, 22, 23],
    [13, 14, 15],
    [5, 6, 7]
]
MILL_COLS = [
    [0, 3, 5],
    [8, 11, 13],
    [16, 19, 21],
    [1, 9, 17],
    [22, 14, 6],
    [18, 20, 23],
    [10, 12, 15],
    [2, 4, 7]
]
DOUBLE_MILL = [
    list(mill_row_set | mill_col_set)  # union of row and column indices
    for mill_row_set in [set(row) for row in MILL_ROWS]  # rows to sets for simple union operation
    for mill_col_set in [set(col) for col in MILL_COLS]  # cols to sets for simple union operation
    if len(mill_row_set & mill_col_set) > 0  # only if they have an intersection
]

# time handling constants
TIDY_UP_TIME = 0.2


# State and Move structs
Move = namedtuple('Move', 'soldier_new_pos, soldier_id, rival_soldier_to_kill')
class State:
    def __init__(self, player,
                 board: np.ndarray,
                 my_pos: np.ndarray,
                 rival_pos: np.ndarray,
                 turn_num: int,
                 my_turn: bool,
                 last_move: Move = None):
        self.player = player
        self.board = board
        self.my_pos = my_pos
        self.rival_pos = rival_pos
        self.turn_num = turn_num
        self.my_turn = my_turn
        self.last_move = last_move

    def is_stage_1(self):
        return self.turn_num < STAGE_1_NUM_OF_TURNS

    def killed_soldier(self):
        if self.last_move is None:
            return False
        return self.last_move.rival_soldier_to_kill != NO_SOLDIER

    def check_valid(self):
        my_soldier_idx = self.my_pos[(self.my_pos != NO_SOLDIER) & (self.my_pos != DEAD_SOLDIER)]
        rival_soldier_idx = self.rival_pos[(self.rival_pos != NO_SOLDIER) & (self.rival_pos != DEAD_SOLDIER)]
        other_idx = list(set(range(NUM_POS_ON_BOARD)) - (set(my_soldier_idx) | set(rival_soldier_idx)))
        return (
                np.all(self.board[my_soldier_idx] == MY_ID_IN_BOARD) and
                np.all(self.board[rival_soldier_idx] == RIVAL_ID_IN_BOARD) and
                np.all(self.board[other_idx] == EMPTY_CELL_IN_BOARD)
        )

    def update_state(self, move: Move):
        active_pos, active_soldier, passive_killed = move

        # new output state
        new_state = copy(self)
        new_state.turn_num += 1  # for next turn
        new_state.my_turn = not self.my_turn  # swap who's turn it is
        new_state.last_move = move  # change last move to the updating move

        # find active player info
        active_pos_list = new_state.my_pos if self.my_turn else new_state.rival_pos
        active_id = MY_ID_IN_BOARD if self.my_turn else RIVAL_ID_IN_BOARD

        if self.is_stage_1():
            assert active_pos_list[active_soldier] == NO_SOLDIER
            assert new_state.board[active_pos] == EMPTY_CELL_IN_BOARD

            active_pos_list[active_soldier] = active_pos
            new_state.board[active_pos] = active_id
        else:
            assert active_pos_list[active_soldier] != NO_SOLDIER and active_pos_list[active_soldier] != DEAD_SOLDIER
            prev_pos = active_pos_list[active_soldier]
            assert new_state.board[prev_pos] == active_id
            assert new_state.board[active_pos] == EMPTY_CELL_IN_BOARD

            active_pos_list[active_soldier] = active_pos
            new_state.board[prev_pos] = EMPTY_CELL_IN_BOARD
            new_state.board[active_pos] = active_id

        if passive_killed != NO_SOLDIER:
            passive_pos_list = new_state.rival_pos if self.my_turn else new_state.my_pos
            passive_id = RIVAL_ID_IN_BOARD if self.my_turn else MY_ID_IN_BOARD
            assert np.any(passive_pos_list == passive_killed)
            assert new_state.board[passive_killed] == passive_id

            passive_pos_list[passive_pos_list == passive_killed] = DEAD_SOLDIER
            new_state.board[passive_killed] = EMPTY_CELL_IN_BOARD

        return new_state

    def __eq__(self, other):
        return (
            np.all(self.board == other.board) and
            self.is_stage_1() == other.is_stage_1() and
            self.my_turn == other.my_turn and
            self.killed_soldier() == other.killed_soldier()
        )

    def __hash__(self):
        return hash(
            # all board values in order
            tuple(val for val in self.board) +
            (
                self.is_stage_1(),
                self.my_turn,
                self.killed_soldier()
            )
        )

    def __copy__(self):
        return self.__class__(self.player, self.board.copy(), self.my_pos.copy(), self.rival_pos.copy(),
                              self.turn_num, self.my_turn, self.last_move)

    def __repr__(self):
        return f'State({self.board=}, {self.my_pos=}, {self.rival_pos=}, {self.turn_num=}, {self.my_turn=}, ' \
               f'{self.last_move=})'


def successor(state: State):
    # make copy of state so that values are unchanged
    new_state = State(state.player,
                      state.board.copy(),
                      state.my_pos.copy(),
                      state.rival_pos.copy(),
                      state.turn_num + 1,  # next state is in next turn
                      not state.my_turn)  # flip turn

    # according to who's turn it is, set variables to use and manipulate
    active_player_board_id = MY_ID_IN_BOARD if state.my_turn else RIVAL_ID_IN_BOARD
    passive_player_board_id = RIVAL_ID_IN_BOARD if state.my_turn else MY_ID_IN_BOARD
    active_player_pos = new_state.my_pos if state.my_turn else new_state.rival_pos
    passive_player_pos = new_state.rival_pos if state.my_turn else new_state.my_pos

    if state.is_stage_1():
        succ_gen = __stage_1_succ
    else:
        succ_gen = __stage_2_succ

    yield from succ_gen(new_state,
                        active_player_board_id,
                        passive_player_board_id,
                        active_player_pos,
                        passive_player_pos)


def __stage_1_succ(state: State, active_player_board_id, passive_player_board_id, active_player_pos,
                   passive_player_pos):

    free_pos = np.where(state.board == EMPTY_CELL_IN_BOARD)[0].astype(np.int)
    next_soldier = int(np.where(active_player_pos == NO_SOLDIER)[0][0])

    for pos in free_pos:
        # change state according to player turn
        state.board[pos] = active_player_board_id
        active_player_pos[next_soldier] = pos

        if state.player.is_mill(pos, state.board):
            yield from __handle_kill_soldier(next_soldier, pos, passive_player_board_id, passive_player_pos, state)
        else:
            move = Move(pos, next_soldier, NO_SOLDIER)
            state.last_move = move
            yield state, move
            state.last_move = None

        # undo change state according to player turn
        state.board[pos] = EMPTY_CELL_IN_BOARD
        active_player_pos[next_soldier] = NO_SOLDIER


def __stage_2_succ(state: State, active_player_board_id, passive_player_board_id, active_player_pos,
                   passive_player_pos):

    # try to move every soldier
    player_turn_soldier_idx = np.where(active_player_pos != NO_SOLDIER)[0]
    for cur_soldier in player_turn_soldier_idx:
        # get cur soldier pos
        prev_pos = active_player_pos[cur_soldier]

        if prev_pos in [NO_SOLDIER, DEAD_SOLDIER]:
            continue

        # find legal moves for soldier
        legal_dirs = [d for d in get_directions(prev_pos) if state.board[d] == EMPTY_CELL_IN_BOARD]

        # soldier is picked up. empty in board
        state.board[prev_pos] = EMPTY_CELL_IN_BOARD

        # iterate legal moves and execute
        for new_pos in legal_dirs:
            # put soldier in new pos
            assert not np.any(active_player_pos == new_pos), (f'move {cur_soldier} to {new_pos} but {active_player_pos} and {state.board}')

            active_player_pos[cur_soldier] = new_pos
            state.board[new_pos] = active_player_board_id

            if state.player.is_mill(new_pos, state.board):
                yield from __handle_kill_soldier(cur_soldier, new_pos, passive_player_board_id, passive_player_pos,
                                                 state)
            else:
                move = Move(new_pos, cur_soldier, NO_SOLDIER)
                state.last_move = move
                yield state, move
                state.last_move = None

            # undo put soldier in new pos
            # NOTE no need to change active player pos, new pos will be given soon and reset after loop
            state.board[new_pos] = EMPTY_CELL_IN_BOARD

        # soldier is returned to previous position
        active_player_pos[cur_soldier] = prev_pos
        state.board[prev_pos] = active_player_board_id


def __handle_kill_soldier(cur_soldier, new_pos, passive_player_board_id, passive_player_pos, state):
    killable_soldiers = np.where((passive_player_pos != NO_SOLDIER) & (passive_player_pos != DEAD_SOLDIER))[0]
    if killable_soldiers.size == 0:
        yield state, Move(new_pos, cur_soldier, NO_SOLDIER)
    else:
        for soldier_to_kill in killable_soldiers:
            # change state according to rival kill soldier
            rival_prev_pos = passive_player_pos[soldier_to_kill]
            if rival_prev_pos in [NO_SOLDIER, DEAD_SOLDIER]:
                continue

            state.board[rival_prev_pos] = EMPTY_CELL_IN_BOARD
            passive_player_pos[soldier_to_kill] = NO_SOLDIER

            move = Move(new_pos, cur_soldier, rival_prev_pos)
            state.last_move = move
            yield state, move
            state.last_move = None

            # undo change state according to rival kill soldier
            state.board[rival_prev_pos] = passive_player_board_id
            passive_player_pos[soldier_to_kill] = rival_prev_pos


def is_terminal_state(state: State):
    return _player_cant_move(state) or _player_soldiers_dead(state)


def _pos_feasible_on_board(pos, board):
    """
    Taken from Game class
    """
    #      on board           free cell
    return (0 <= pos < NUM_POS_ON_BOARD) and board[pos] == EMPTY_CELL_IN_BOARD


def _player_cant_move(state: State) -> bool:
    if state.is_stage_1():
        # can always move in stage 1
        return False

    player_pos = state.my_pos if state.my_turn else state.rival_pos

    # iterate soldiers and check move
    for pos in player_pos:
        # non existent soldiers can't move.
        if pos in [NO_SOLDIER, DEAD_SOLDIER]:
            continue

        # check if there are any legal moves for this soldier
        if any(_pos_feasible_on_board(next_pos, state.board) for next_pos in get_directions(pos)):
            return False  # soldier has legal moves. player can still move

    return True  # no legal move found. player can't move


def _player_soldiers_dead(state: State) -> bool:
    """
    Taken from Game class
    """
    player_pos = state.my_pos if state.my_turn else state.rival_pos
    alive = np.where(player_pos != DEAD_SOLDIER)[0]
    if len(alive) < MINIMAL_NUM_SOLDIERS:
        return True
    return False


##############
# Heuristics #
##############

def closed_mill(state: State):
    if state.last_move is not None:
        # if rival turn, then we closed the morris, otherwise the rival closed the morris
        score = 1 if not state.my_turn else -1

        # return morris score if morris was actually closed
        return score * (state.last_move.rival_soldier_to_kill != NO_SOLDIER)

    # 0 in all other cases
    return 0


def num_mills(board: np.ndarray, player_id: int):
    mill_count = 0
    for possible_mill_idx in MILL_ROWS + MILL_COLS:
        mill_vals = board[possible_mill_idx]
        if np.all(mill_vals == player_id):
            mill_count += 1

    return mill_count


def num_blocked_soldiers(board: np.ndarray, player_pos):
    blocked_count = 0
    for pos in player_pos:
        if pos not in [NO_SOLDIER, DEAD_SOLDIER] and not any(_pos_feasible_on_board(next_, board)
                                                             for next_ in get_directions(pos)):
            blocked_count += 1

    return blocked_count


def num_soldiers(player_pos):
    return np.sum((player_pos != NO_SOLDIER) & (player_pos != DEAD_SOLDIER))


def __get_all_incomplete_mills(board: np.ndarray, player_id: int):
    incomplete_mills = []
    for possible_mill_idx in MILL_ROWS + MILL_COLS:
        mill_vals = board[possible_mill_idx]
        if np.sum(mill_vals == player_id) == 2 and np.sum(mill_vals == EMPTY_CELL_IN_BOARD) == 1:
            incomplete_mills.append(possible_mill_idx)

    return incomplete_mills


def num_incomplete_mills(board: np.ndarray, player_id: int):
    return len(__get_all_incomplete_mills(board, player_id))


def num_incomplete_double_mills(board: np.ndarray, player_id: int):
    # same as counting every two intersecting incomplete mills
    incomplete_double_mills_set = set()
    incomplete_mill_sets = [set(inc_mill) for inc_mill in __get_all_incomplete_mills(board, player_id)]
    for i in range(len(incomplete_mill_sets) - 1):
        for j in range(i + 1, len(incomplete_mill_sets)):
            intersection = incomplete_mill_sets[i] & incomplete_mill_sets[j]

            if len(intersection) > 0:
                incomplete_double_mills_set.add(tuple(intersection))

    return len(incomplete_double_mills_set)


def num_double_mills(board: np.ndarray, player_id: int):
    double_mill_count = 0
    for possible_double_mill_idx in DOUBLE_MILL:
        double_mill_vals = board[possible_double_mill_idx]
        if np.all(double_mill_vals == player_id):
            double_mill_count += 1

    return double_mill_count


def win_config(state: State):
    if is_terminal_state(state):
        if state.my_turn:
            return -1  # our turn on terminal state. rival wins
        else:
            return 1  # rival turn on terminal state. we win

    return 0  # all other states assume no knowledge


#####################
# Utility Functions #
#####################

def light_utility_function(state: State):
    sold = num_soldiers(state.my_pos) - num_soldiers(state.rival_pos)
    blk = num_blocked_soldiers(state.board, state.rival_pos) - num_blocked_soldiers(state.board, state.my_pos)
    win = win_config(state)

    return sold + blk + 1000*win


def medium_utility_function(state: State):
    sold = num_soldiers(state.my_pos) - num_soldiers(state.rival_pos)
    blk = num_blocked_soldiers(state.board, state.rival_pos) - num_blocked_soldiers(state.board, state.my_pos)
    inc = num_incomplete_mills(state.board, MY_ID_IN_BOARD) - num_incomplete_mills(state.board, RIVAL_ID_IN_BOARD)
    comp = num_mills(state.board, MY_ID_IN_BOARD) - num_mills(state.board, RIVAL_ID_IN_BOARD)
    win = win_config(state)

    return sold + blk + inc + comp + 1000*win


def heavy_utility_function(state: State):
    closed = closed_mill(state)
    mills = num_mills(state.board, MY_ID_IN_BOARD) - num_mills(state.board, RIVAL_ID_IN_BOARD)
    block = num_blocked_soldiers(state.board, state.rival_pos) - num_blocked_soldiers(state.board, state.my_pos)
    soldiers = num_soldiers(state.my_pos) - num_soldiers(state.rival_pos)

    if state.is_stage_1():
        inc_mills = num_incomplete_mills(state.board, MY_ID_IN_BOARD) - num_incomplete_mills(state.board,
                                                                                             RIVAL_ID_IN_BOARD)
        inc_double = num_incomplete_double_mills(state.board, MY_ID_IN_BOARD) - num_incomplete_double_mills(state.board,
                                                                                                            RIVAL_ID_IN_BOARD)

        # hyperparameters from:
        # https://kartikkukreja.wordpress.com/2014/03/17/heuristicevaluation-function-for-nine-mens-morris/
        return 18 * closed + 26 * mills + 1 * block + 9 * soldiers + 10 * inc_mills + 7 * inc_double
    else:
        double_mills = num_double_mills(state.board, MY_ID_IN_BOARD) - num_double_mills(state.board, RIVAL_ID_IN_BOARD)
        win = win_config(state)

        # hyperparameters from:
        # https://kartikkukreja.wordpress.com/2014/03/17/heuristicevaluation-function-for-nine-mens-morris/
        return 14 * closed + 43 * mills + 10 * block + 11 * soldiers + 8 * double_mills + 1086 * win


################
# Pre-Existing #
################


def get_directions(position):
    """Returns all the possible directions of a player in the game as a list.
    """
    assert 0 <= position <= 23, "illegal move" + f' {position}'
    adjacent = [
        [1, 3],
        [0, 2, 9],
        [1, 4],
        [0, 5, 11],
        [2, 7, 12],
        [3, 6],
        [5, 7, 14],
        [4, 6],
        [9, 11],
        [1, 8, 10, 17],
        [9, 12],
        [3, 8, 13, 19],
        [4, 10, 15, 20],
        [11, 14],
        [6, 13, 15, 22],
        [12, 14],
        [17, 19],
        [9, 16, 18],
        [17, 20],
        [11, 16, 21],
        [12, 18, 23],
        [19, 22],
        [21, 23, 14],
        [20, 22]
    ]
    return adjacent[position]

# def tup_add(t1, t2):
#     """
#     returns the sum of two tuples as tuple.
#     """
#     return tuple(map(operator.add, t1, t2))

def printBoard(board):
    print(int(board[0]),"(00)-----------------------",int(board[1]),"(01)-----------------------",int(board[2]),"(02)")
    print("|                             |                             |")
    print("|                             |                             |")
    print("|                             |                             |")
    print("|       ",int(board[8]),"(08)--------------",int(board[9]),"(09)--------------",int(board[10]),"(10)   |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       |        ",int(board[16]),"(16)-----",int(board[17]),"(17)-----",int(board[18]),"(18)   |        |")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print(int(board[3]),"(03)-",int(board[11]),"(11)---",int(board[19]),"(19)                 ",
          int(board[20]),"(20)-",int(board[12]),"(12)---",int(board[4]),"(04)")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print("|       |         |                       |        |        |")
    print("|       |        ",int(board[21]),"(21)-----",int(board[22]),"(22)-----",int(board[23]),"(23)   |        |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       |                     |                    |        |")
    print("|       ",int(board[13]),"(13)--------------",int(board[14]),"(14)--------------",int(board[15]),"(15)   |")
    print("|                             |                             |")
    print("|                             |                             |")
    print("|                             |                             |")
    print(int(board[5]),"(05)-----------------------",int(board[6]),"(06)-----------------------",int(board[7]),"(07)")
    print("\n")

