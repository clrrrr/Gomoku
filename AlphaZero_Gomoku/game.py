# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2
        self.current_player = None
        self.availables = None
        self.forbidden = None
        self.last_move = None
        # self.directions = [[-1, -1],[-1, 0], [-1, 1],
        #                    [0, -1], [0, 1],
        #                    [1, -1], [1, 0], [1, 1]]
        self.directions = [[-1, 1],[0, 1],[1, 0],[1, 1]]

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.forbidden = set()
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def get_valid_moves(self):
        if len(self.states) % 2 == 0:  # black to go
            return list(set(self.availables) - self.forbidden)
        else:
            return self.availables

    def count_pieces(self, square_state, start, direction, length):
        """
        :param square_state: 3-d array, slice 0:black, 1:white
        :param start: [x, y]
        :param direction: [dx, dy]
        :param length: int
        :return: [n_black, n_space] indicating number of black pieces/spaces
        along a consecutive path from start position
        do not judge if the visit of array is legal
        """
        n_black = 0
        n_space = 0
        for i in range(0, length):
            cur_x = start[0] + i * direction[0]
            cur_y = start[1] + i * direction[1]
            if square_state[0][cur_x][cur_y]:
                n_black += 1
            else:
                if not square_state[1][cur_x][cur_y]:
                    n_space += 1
        return [n_black, n_space]

    def is_legal(self, position):
        x, y = position
        if x < 0 or y < 0:
            return False
        if x >= self.width or y >= self.height:
            return False
        return True

    def count_six(self, potential, square_state):
        # count number of directions which has consecutive grids with length >=6
        n_six = 0
        for direction in self.directions:
            for i in range(-5, 1):
                start_x = potential[0] + i * direction[0]
                start_y = potential[1] + i * direction[1]
                end_x = start_x + 5 * direction[0]
                end_y = start_y + 5 * direction[1]
                if self.is_legal((start_x, start_y)) and self.is_legal((end_x, end_y)):
                    n_black, _ = self.count_pieces(square_state, (start_x, start_y), direction, 6)
                    if n_black == 5:
                        n_six += 1
                        break
        return n_six

    def has_five(self, potential, square_state):
        # positions to win
        for direction in self.directions:
            for i in range(-4, 1):
                start_x = potential[0] + i * direction[0]
                start_y = potential[1] + i * direction[1]
                end_x = start_x + 4 * direction[0]
                end_y = start_y + 4 * direction[1]
                if self.is_legal((start_x, start_y)) and self.is_legal((end_x, end_y)):
                    n_black, _ = self.count_pieces(square_state, (start_x, start_y), direction, 5)
                    if n_black == 4:
                        back_x = start_x - direction[0]
                        back_y = start_y - direction[1]
                        forward_x = end_x + direction[0]
                        forward_y = end_y + direction[1]
                        can_not_extend = True
                        if self.is_legal((back_x, back_y)):
                            if square_state[0][back_x][back_y]:
                                can_not_extend = False
                        if self.is_legal((forward_x, forward_y)):
                            if square_state[0][forward_x][forward_y]:
                                can_not_extend = False
                        if can_not_extend:
                            return True
        return False

    def count_three(self, potential, square_state):
        # count number of free 3's(which leads to bidirectional free 4's with one step)
        n_three = 0
        for direction in self.directions:
            for i in range(-4, 1):
                back_x = potential[0] + i * direction[0]
                back_y = potential[1] + i * direction[1]
                forward_x = back_x + 5 * direction[0]
                forward_y = back_y + 5 * direction[1]
                if not (self.is_legal((back_x, back_y)) and self.is_legal((forward_x, forward_y))):
                    continue
                if square_state[0][back_x][back_y] or square_state[1][back_x][back_y]:
                    continue
                if square_state[0][forward_x][forward_y] or square_state[1][forward_x][forward_y]:
                    continue
                start_x = back_x + direction[0]
                start_y = back_y + direction[1]
                n_black, n_space = self.count_pieces(square_state, (start_x, start_y), direction, 4)
                if n_black == 2 and n_space == 2:
                    n_three += 1
                    break
        return n_three

    def count_four(self, potential, square_state):
        # count number of 4's
        n_four = 0
        for direction in self.directions:
            for i in range(-4, 1):
                start_x = potential[0] + i * direction[0]
                start_y = potential[1] + i * direction[1]
                end_x = start_x + 4 * direction[0]
                end_y = start_y + 4 * direction[1]
                if self.is_legal((start_x, start_y)) and self.is_legal((end_x, end_y)):
                    n_black, n_space = self.count_pieces(square_state, (start_x, start_y), direction, 5)
                    if n_black == 3 and n_space == 2:
                        n_four += 1
                        break
        return n_four

    def update_forbidden(self):
        """
        recheck forbidden grids for the black after white's turn
        i.e. current_player=black

        my memo:
        only checking new forbidden places caused by the latest move is wrong
        since some of the old forbidden grids can no longer be forbidden after some white stones being placed
        so for the sake of convenience, here in my implementation I simply recheck all grids
        :return: void
        """

        square_state = np.zeros((2, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_black = moves[players == self.current_player]
            move_white = moves[players != self.current_player]

            # 0: black state
            # 1: white state
            square_state[0][move_black // self.width,
                            move_black % self.height] = 1.0
            square_state[1][move_white // self.width,
                            move_white % self.height] = 1.0

        self.forbidden = set()
        for move in self.availables:
            # the potential(to be forbidden) grid is a space grid
            potential = self.move_to_location(move)

            # long consecutive sequence
            if self.count_six(potential, square_state) > 0:
                self.forbidden.add(move)
                continue

            # A win has higher priority than the following forbidden rules
            if self.has_five(potential, square_state):
                continue

            # 3-3's
            if self.count_three(potential, square_state) >= 2:
                self.forbidden.add(move)
                continue

            # 4-4's
            if self.count_four(potential, square_state) >= 2:
                self.forbidden.add(move)

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

        # print(move)

        # update forbidden places
        if len(self.states) % 2 == 0:  # update after white's operation!!
            # recheck forbidden places
            self.update_forbidden()

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        if len(self.get_valid_moves()) == 0 and len(self.availables) > 0:
            return True, states[moved[-1]]

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
