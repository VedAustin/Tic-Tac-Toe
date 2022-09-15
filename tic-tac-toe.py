"""
Python implementation: CPython
Python version       : 3.10.6
numpy                : 1.23.2

Compiler    : Clang 13.0.0 (clang-1300.0.29.30)
OS          : Darwin
Release     : 21.6.0
Machine     : x86_64
Processor   : i386
CPU cores   : 8
Architecture: 64bit

***********************
This is a variation of the classic game of tic-tac-toe, where 2 players take turns putting X's and O's in a 3x3 grid,
and the first to get 3 in a row (horizontally, vertically, or diagonally) wins. 
In this version, there are 9 instances of the game being played simultaneously. 
Each turn, a player may make a single move in any one of the 9 game instances. 
(Note that the game instances can thus become unbalanced in the number of moves if a player does not respond directly
but instead plays in another instance.) The game instances are themselves arranged in a 3x3 grid, constituting a meta-game. 
Each time a player wins a game instance, they take the corresponding square in the meta-game. 
If a game instance results in a tie, that game instance is replaced with a fresh instance with no moves played. 
The game is over when a player wins the meta-game. 
***********************
"""

import numpy as np
import random
import logging
from collections import Counter
from enum import Enum
from dataclasses import dataclass
from typing import Optional

logging.getLogger().setLevel(logging.ERROR)


class Status(Enum):
    OFFENSE = 0
    DEFENSE = 1
    RANDOM = 2
    FULL = 3


@dataclass
class Player:
    name: str
    property_value: int


def create_board() -> np.ndarray:
    """Create a 3 x 3 board initialised to zero value

    Returns:
        np.ndarray: zero filled 3 x 3 board
    """
    return np.zeros((3, 3), dtype=int)


def find_available_cells(instance_board: np.ndarray) -> np.ndarray:
    """Identify cells that are still unfilled

    Args:
        instance_board (np.ndarray): instance of tic tac toe board

    Returns:
        np.ndarray: indices of all unfilled cells (if any)
    """
    return np.argwhere(instance_board == 0)


def pick_cell_randomly(instance_board: np.ndarray) -> tuple | None:
    """Pick a random cell from the board

    Args:
        instance_board (np.ndarray): instance of tic tac toe board

    Returns:
        tuple | None: if a cell is available, return indices of that cell or 
        None if no cell is available
    """
    cells_available = find_available_cells(instance_board)
    cells_available_max = len(cells_available)
    if cells_available_max > 0:
        logging.info("cells available - assigning value")
        random_cell = random.randint(0, cells_available_max-1)
        return tuple(cells_available[random_cell])
    else:
        logging.warning("cells unavailable")
        return None


def assign_cell(instance_board: np.ndarray, cell_indices: tuple, player: Player) -> None:
    """If an empty cell has been chosen, it is assigned to the player

    Args:
        instance_board (np.ndarray): instance of tic tac toe board
        cell_indices (tuple): indices of the chosen cell
        Player: player making the move
    """
    instance_board[cell_indices] = player.property_value


def check_winner(instance_board: np.ndarray, player: Player) -> bool:
    """Check if the board has a winner, given the player

    Args:
        instance_board (np.ndarray): instance of tic tac toe board
        Player: check if this player won the board

    Returns:
        bool: True if the player is the winner otherwise False
    """
    player_val = player.property_value
    row_check = np.all(instance_board[0, :] == player_val) | np.all(
        instance_board[1, :] == player_val) | np.all(instance_board[2, :] == player_val)
    column_check = np.all(instance_board[:, 0] == player_val) | np.all(
        instance_board[:, 1] == player_val) | np.all(instance_board[:, 2] == player_val)
    diagonal_check = np.all(np.array([instance_board[i, j] for i, j in zip(range(3), range(2, -1, -1))]) == player_val) | \
        np.all(np.array([instance_board[i, i]
               for i in range(3)]) == player_val)
    return bool(row_check or column_check or diagonal_check)


def random_move(instance_board: np.ndarray, player: Player) -> bool:
    """Randomly assign a cell within a board to a player

    Args:
        instance_board (np.ndarray): instance of tic tac toe board
        Player: player making the move

    Returns:
        bool: True if it was successful, False otherwise
    """
    cells_picked = pick_cell_randomly(instance_board)
    if cells_picked:
        assign_cell(instance_board, cells_picked, player)
        return True
    return False


def defensive_move(instance_board: np.ndarray, player: Player) -> bool:
    """Prevent the opponent (if possible) when making the winning move by claiming the cell

    Args:
        instance_board (np.ndarray): instance of tic tac toe board
        Player: player making the move

    Returns:
        bool: True if it was successful, False otherwise
    """
    board_copy = instance_board.copy()
    opponent_player = P2 if player.property_value == 1 else P1
    cells_picked = pick_cell_randomly(board_copy)
    while cells_picked:
        assign_cell(board_copy, cells_picked, opponent_player)
        if check_winner(board_copy, opponent_player):
            assign_cell(instance_board, cells_picked, player)
            return True
        else:
            attempts_tracker = Player(name="Tracker", property_value=101)
            assign_cell(board_copy, cells_picked, attempts_tracker)
            cells_picked = pick_cell_randomly(board_copy)
    return False


def offensive_move(instance_board: np.ndarray, player: Player) -> bool:
    """Making the winning move if an opportunity exists

    Args:
        instance_board (np.ndarray): instance of tic tac toe board
        Player: player making the move

    Returns:
        bool: True if it was successful, False otherwise
    """
    board_copy = instance_board.copy()
    cells_picked = pick_cell_randomly(board_copy)
    while cells_picked:
        assign_cell(board_copy, cells_picked, player)
        if check_winner(board_copy, player):
            assign_cell(instance_board, cells_picked, player)
            return True
        else:
            attempts_tracker = Player(name="Tracker", property_value=101)
            assign_cell(board_copy, cells_picked, attempts_tracker)
            cells_picked = pick_cell_randomly(board_copy)
    return False


def fill_cell(instance_board: np.ndarray, player: Player) -> Status:
    """Fill the cell with a winning move, defensive move or a random move

    Args:
        instance_board (np.ndarray): instance of tic tac toe board
        Player: player making the move

    Returns:
        Status: Return the type of move completed
    """
    if offensive_move(instance_board, player) is False:
        logging.info("winning move not available")
        if defensive_move(instance_board, player) is False:
            logging.info(
                "opponent did not have a cell that needed to be blocked")
            if random_move(instance_board, player) is False:
                logging.info("No cells available")
                return Status.FULL
            else:
                logging.info("Picked a random cell on the board")
                return Status.RANDOM
        else:
            logging.info("Successfully blocked a winning move by the opponent")
            return Status.DEFENSE
    else:
        logging.warning("Found a winning move")
        return Status.OFFENSE


def choose_player(current_player: Optional[Player] = None) -> Player:
    """Choose a random player (when starting the game) or their opponent (when taking turns)

    Args:
        current_player (Optional[int], optional): player who made the latest move. 
        Defaults to None when the game starts

    Returns:
        Player: return the player chosen
    """
    if current_player:
        if current_player.property_value == 1:
            return P2
        else:
            return P1
    else:
        players = [P1, P2]
        return random.choice(players)


def choose_board(meta_board: np.ndarray, meta_board_results: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Selecting a random instance of the board from the meta board

    Args:
        meta_board (np.ndarray): 9 instances of a ti tac toe board in one big board
        meta_board_results (np.ndarray): outcome of individual instance boards: [1, 0, -1]

    Returns:
        tuple[np.ndarray, tuple[int, int]]: instance of the smaller board + positional indices of that board
    """
    x, y = random.choice([0, 1, 2]), random.choice([0, 1, 2])
    if is_board_complete(meta_board_results):
        meta_board, meta_board_results = reset_board(
            meta_board, meta_board_results, (x, y))
    current_choice = meta_board_results[x, y]
    if current_choice == 0:
        return meta_board[x, y], (x, y)
    return choose_board(meta_board, meta_board_results)


def is_board_complete(instance_board: np.ndarray) -> bool:
    """Check to see if an instance board has every cell filled

    Args:
        instance_board (np.ndarray): instance of tic tac toe board

    Returns:
        bool: True if it is, False otherwise
    """
    return bool(np.all(instance_board != 0))


def update_meta_results(meta_board_results: np.ndarray, board_pos: tuple[int, int], player: Player) -> np.ndarray:
    """If an instance of a board has been won, update the results of the meta board

    Args:
        meta_board_results (np.ndarray): results of the meta board
        board_pos (tuple[int, int]): indices of the instance board in the meta board
        Player: player of interest

    Returns:
        np.ndarray: updated results of the meta board
    """
    meta_board_results[board_pos] = player.property_value
    return meta_board_results


def reset_board(meta_board: np.ndarray, meta_board_results: np.ndarray, board_pos: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Replace an instance of an existing board with a new board and also update its value to zero in the meta board

    Args:
        meta_board (np.ndarray): Board containing all the other instance boards
        meta_board_results (np.ndarray): results of the meta board
        board_pos (tuple[int, int]): positional indices of the instance board

    Returns:
        tuple[np.ndarray, np.ndarray]: updated meta board and the results of the meta board
    """
    meta_board[board_pos] = create_board()
    meta_board_results[board_pos] = 0
    return meta_board, meta_board_results


def play_meta_game_bots() -> Player:
    """Play a game between 2 bots

    Returns:
        Player: Winning player
    """
    meta_board = np.array([create_board()]*9).reshape((3, 3, 3, 3))
    meta_board_results = np.zeros((3, 3), dtype=int)

    current_player = choose_player()
    current_board, current_board_pos = choose_board(
        meta_board, meta_board_results)
    status = fill_cell(current_board, current_player)

    while not (check_winner(meta_board_results, P1)) and not (check_winner(meta_board_results, P2)):
        if status == Status.FULL:
            meta_board, meta_board_results = reset_board(
                meta_board, meta_board_results, current_board_pos)
        else:
            if status == Status.OFFENSE:
                meta_board_results = update_meta_results(
                    meta_board_results, current_board_pos, current_player)

        current_player = choose_player(current_player)
        current_board, current_board_pos = choose_board(
            meta_board, meta_board_results)
        status = fill_cell(current_board, current_player)

    if check_winner(meta_board_results, P2):
        return P2
    else:
        return P1


def play_meta_game() -> Player:
    """Play a game between a human and bot

    Returns:
        Player: Winning player
    """
    meta_board = np.array([create_board()]*9).reshape((3, 3, 3, 3))
    meta_board_results = np.zeros((3, 3), dtype=int)

    human_player = P1
    bot_player = P2

    while not (check_winner(meta_board_results, human_player)) and not (check_winner(meta_board_results, bot_player)):
        print("--------Meta Board (current)--------")
        print(meta_board)

        print("--------Meta Board Results(current)--------")
        print(meta_board_results)

        instance_board_coord = input("Select any instance board -> x,y: ")
        board_pos = (int(instance_board_coord[0]), int(
            instance_board_coord[2]))
        instance_board = meta_board[board_pos]
        print("--------Instance Board selected--------")
        print(instance_board)

        cell_coord_raw = input("Select the cell -> x,y: ")
        cell_coord = (int(cell_coord_raw[0]), int(cell_coord_raw[2]))
        instance_board[cell_coord] = human_player.property_value
        print("--------Instance Board updated--------")
        print(instance_board)

        # Check if human made a winning move
        if check_winner(instance_board, human_player):
            meta_board_results = update_meta_results(
                meta_board_results, board_pos, human_player)

        # Bot's turn to play
        current_board, current_board_pos = choose_board(
            meta_board, meta_board_results)
        status = fill_cell(current_board, bot_player)

        if status == Status.FULL:
            meta_board, meta_board_results = reset_board(
                meta_board, meta_board_results, current_board_pos)
        else:
            if status == Status.OFFENSE:
                meta_board_results = update_meta_results(
                    meta_board_results, current_board_pos, bot_player)

    if check_winner(meta_board_results, bot_player):
        return bot_player
    else:
        return human_player


if __name__ == '__main__':
    player_type = bool(int(input("Human[0] or Bot[1]: ")))
    P1 = Player(name='Bot1', property_value=1)
    P2 = Player(name='Bot2', property_value=-1)
    # Bot vs Bot
    if player_type:
        # print(Counter([play_meta_game_bots() for _ in range(10)]))
        winner = play_meta_game_bots()
    else:
        # Human vs Bot
        P1.name = 'Human'
        P2.name = 'Bot'
        winner = play_meta_game()

    print(f"Winner: {winner.name}")
