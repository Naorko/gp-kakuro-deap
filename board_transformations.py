import random

import numpy as np

from board import Board, Col, Row, Cell
from board_translator import get_boards

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~transformations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from board_utils import get_idx_ass, row_has_dups, col_has_dups, remove_dups_from_ass, get_poss_ass

INVALID_IDX = -1


# all board transformation

def put_mandatory_ass(board: Board) -> Board:
    cols_opt = board.cols_opt
    rows_opt = board.rows_opt
    cols_map = board.cols_map
    for col_idx, col_map in enumerate(cols_map):
        col_opt = cols_opt[col_idx]
        for row_idx, row_cell in col_map:
            row_opt = rows_opt[row_idx]
            unioned_col_opt = set.union(*[set(opt) for opt in col_opt])
            unioned_row_opt = set.union(*[set(opt) for opt in row_opt])
            intersect_opt = unioned_col_opt.intersection(unioned_row_opt)
            if len(intersect_opt) == 1:
                board.assignment[row_idx][row_cell] = intersect_opt.pop()
    return board

# cells transformation
def cell_add_max_ass(board: Board, cell: Cell) -> Board:
    ass = Board.EMPTY_CELL
    poss_ass = get_poss_ass(board, cell)
    if poss_ass:
        ass = max(poss_ass)
    board.assignment[cell.row_i][cell.row_j] = ass
    cell.ass = ass
    cell.opt_ass.remove(ass)


def cell_add_min_ass(board: Board, cell: Cell) -> Board:
    ass = Board.EMPTY_CELL
    poss_ass = get_poss_ass(board, cell)
    if poss_ass:
        ass = min(poss_ass)
    board.assignment[cell.row_i][cell.row_j] = ass
    cell.ass = ass
    cell.opt_ass.remove(ass)


# cells analyze

def get_largest_opt(board: Board) -> Cell:
    max_opt_cell = None
    max_opt = -1
    for (row_i,row_j), cell in board.cells.items():
        if len(cell.opt_ass) > max_opt:
            max_opt_cell = cell
            max_opt = len(cell.opt_ass)
    return max_opt_cell

def get_smallest_opt(board: Board) -> Cell:
    min_opt_cell = None
    min_opt = 10
    for (row_i,row_j), cell in board.cells.items():
        if len(cell.opt_ass) < min_opt:
            min_opt_cell = cell
            min_opt = len(cell.opt_ass)
    return min_opt_cell

def get_lowest_ass_row(board: Board) -> Cell:
    row_idx = None
    min_empty_cell = 10
    for row_i, row in enumerate(board.assignment):
        row_empty_cells = len([cell for cell in row if cell == Board.EMPTY_CELL])
        if min_empty_cell > row_empty_cells:
            row_idx = row_i
            min_empty_cell = row_empty_cells

    empty_cells_idxs = [i for i, val in enumerate(board.assignment[row_idx]) if val == Board.EMPTY_CELL]
    row_j = 0
    if empty_cells_idxs:
        row_j = random.choice(empty_cells_idxs)

    return board.cells[(row_idx,row_j)]

def get_largest_ass_row(board: Board) -> Cell:
    row_idx = None
    min_empty_cell = -1
    for row_i, row in enumerate(board.assignment):
        row_empty_cells = len([cell for cell in row if cell == Board.EMPTY_CELL])
        if min_empty_cell < row_empty_cells:
            row_idx = row_i
            min_empty_cell = row_empty_cells

    empty_cells_idxs = [i for i, val in enumerate(board.assignment[row_idx]) if val == Board.EMPTY_CELL]
    row_j = 0
    if empty_cells_idxs:
        row_j = random.choice(empty_cells_idxs)
    return board.cells[(row_idx,row_j)]



# rows transformation

def row_add_ass(board: Board, row_idx: Row) -> Board:
    if row_idx == INVALID_IDX:
        return board
    row_ass = set(board.assignment[row_idx])
    if board.EMPTY_CELL not in row_ass:
        return board

    row_ass.remove(board.EMPTY_CELL)
    rows_opt = [opt for opt in board.rows_opt[row_idx] if row_ass.issubset(set(opt))]
    if not rows_opt:  # No valid optional assignment
        return board
    opt_ass = set(random.choice(rows_opt))
    diff_opt_ass = list(opt_ass - row_ass)
    random.shuffle(diff_opt_ass)
    for idx, ass in enumerate(board.assignment[row_idx]):
        if ass == board.EMPTY_CELL:
            board.assignment[row_idx][idx] = diff_opt_ass.pop()
    return board


def row_delete_ass(board: Board, row_idx: Row) -> Board:
    if row_idx != INVALID_IDX:
        board.assignment[row_idx] = [Board.EMPTY_CELL] * board.rows_size[row_idx]
    return board


def row_delete_dup(board: Board, row_idx: Row) -> Board:
    if row_idx != INVALID_IDX:
        new_ass = remove_dups_from_ass(board.assignment[row_idx])
        board.assignment[row_idx] = new_ass

    return board


def row_delete_noopt(board: Board, row_idx: Row) -> Board:
    if row_idx != INVALID_IDX:
        opts = set(sum(board.rows_opt[row_idx], ()))
        ass = board.assignment[row_idx]
        new_ass = [val if val in opts else Board.EMPTY_CELL for val in ass]
        board.assignment[row_idx] = new_ass

    return board


# cols transformations
def col_add_ass(board: Board, col_idx: Col) -> Board:
    if col_idx == INVALID_IDX:
        return board

    col_ass = board.get_col_ass(col_idx)
    col_ass_set = set(col_ass)

    if Board.EMPTY_CELL not in col_ass:
        return board

    col_ass_set.remove(Board.EMPTY_CELL)
    cols_opt = [opt for opt in board.cols_opt[col_idx] if col_ass_set.issubset(set(opt))]
    if not cols_opt:  # No valid optional assignment
        return board

    opt_ass = set(random.choice(cols_opt))
    diff_opt_ass = list(opt_ass - col_ass_set)
    random.shuffle(diff_opt_ass)
    for col_cell, ass in enumerate(col_ass):
        if ass == Board.EMPTY_CELL:
            row_idx, row_cell = get_idx_ass(board, col_idx, col_cell)
            board.assignment[row_idx][row_cell] = diff_opt_ass.pop()
    return board


def col_delete_ass(board: Board, col_idx: Col) -> Board:
    if col_idx != INVALID_IDX:
        new_ass = [Board.EMPTY_CELL] * len(board.cols_map[col_idx])
        board.set_col_ass(col_idx, new_ass)

    return board


def col_delete_dup(board: Board, col_idx: Col) -> Board:
    if col_idx != INVALID_IDX:
        new_ass = remove_dups_from_ass(board.get_col_ass(col_idx))
        board.set_col_ass(col_idx, new_ass)

    return board


def col_delete_noopt(board: Board, col_idx: Col) -> Board:
    if col_idx != INVALID_IDX:
        opts = set(sum(board.cols_opt[col_idx], ()))
        ass = board.get_col_ass(col_idx)
        new_ass = [val if val in opts else Board.EMPTY_CELL for val in ass]
        board.set_col_ass(col_idx, new_ass)

    return board


# analyze nodes


def get_empty_cell_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if Board.EMPTY_CELL in row_ass:
            return row_idx

    return Row(INVALID_IDX)


def get_has_dup_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        if row_has_dups(board, row_idx):
            return row_idx

    return Row(INVALID_IDX)


def get_invalid_sum_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if sum(row_ass) != board.rows_sum[row_idx]:
            return row_idx

    return Row(INVALID_IDX)


def get_empty_cell_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        col_ass = board.get_col_ass(col_idx)
        if Board.EMPTY_CELL in col_ass:
            return col_idx

    return Col(INVALID_IDX)


def get_has_dup_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        if col_has_dups(board, col_idx):
            return col_idx

    return Col(INVALID_IDX)


def get_invalid_sum_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        col_ass = board.get_col_ass(col_idx)
        if sum(col_ass) != board.cols_sum[col_idx]:
            return col_idx

    return Col(INVALID_IDX)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~first experiment~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_invalid_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if Board.EMPTY_CELL in row_ass or sum(row_ass) != board.rows_sum[row_idx] or row_has_dups(board, row_idx):
            return row_idx

    return Row(INVALID_IDX)


def get_invalid_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        col_ass = board.get_col_ass(col_idx)
        if Board.EMPTY_CELL in col_ass or sum(col_ass) != board.cols_sum[col_idx] or col_has_dups(board, col_idx):
            return col_idx

    return Col(INVALID_IDX)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~first experiment~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def board_is_ok(board: Board) -> bool:
    return (np.array([get_empty_cell_col(board), get_has_dup_col(board), get_invalid_sum_col(board),
                      get_empty_cell_row(board), get_has_dup_row(board),
                      get_invalid_sum_row(board)]) == INVALID_IDX).all()


# def calc_sol1(Board):
#     return row_add_ass(row_add_ass(put_mandatory_ass(Board), get_invalid_row(row_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(col_add_ass(Board, get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))))), get_invalid_row(row_add_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)))), -1), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(col_add_ass(Board, -1), get_invalid_row(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)))), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_row(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))))))))), get_invalid_row(row_add_ass(row_add_ass(row_add_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_row(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))))), get_invalid_row(row_add_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(row_add_ass(row_add_ass(put_mandatory_ass(col_add_ass(Board, -1)), -1), get_invalid_row(Board)))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(put_mandatory_ass(Board)))))), get_invalid_row(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))))))), get_invalid_row(col_add_ass(col_add_ass(row_add_ass(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(col_add_ass(Board, get_invalid_col(Board))))))), get_invalid_col(col_add_ass(Board, get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))))))

def calc_sol2(Board):
    return row_add_ass(row_add_ass(row_add_ass(row_add_ass(row_delete_noopt(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, -1), get_invalid_sum_row(col_add_ass(row_add_ass(col_delete_dup(col_add_ass(Board, -1), get_invalid_sum_col(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_col(Board)))))), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_empty_cell_col(Board)))))), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, get_empty_cell_col(Board)), -1), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, -1), -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))))), get_invalid_sum_row(row_add_ass(row_delete_noopt(col_add_ass(col_delete_dup(Board, -1), get_invalid_sum_col(Board)), -1), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, -1), -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))))))))), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, get_empty_cell_col(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), -1))))), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, get_invalid_sum_col(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), -1))))), get_invalid_sum_row(Board)), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(row_add_ass(row_add_ass(row_delete_noopt(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, -1), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_col(Board)))))), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_empty_cell_col(Board)))))), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, -1), -1), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, -1), -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))))), get_invalid_sum_row(row_add_ass(row_delete_noopt(col_add_ass(col_delete_dup(Board, -1), get_invalid_sum_col(Board)), -1), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, -1), -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))))))))), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, get_empty_cell_col(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), -1))))), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, -1), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), -1))))), get_invalid_sum_row(row_add_ass(row_add_ass(col_delete_noopt(Board, -1), get_has_dup_row(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_col(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))))))))), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, -1), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_col(Board)))))), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(Board, get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_row(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_empty_cell_col(Board)))))), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, get_empty_cell_col(Board)), -1), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, -1), -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))))), get_invalid_sum_row(row_add_ass(row_delete_noopt(col_add_ass(col_delete_dup(Board, -1), get_invalid_sum_col(Board)), -1), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, -1), -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))))))))), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(row_add_ass(col_delete_noopt(col_add_ass(Board, -1), get_empty_cell_col(Board)), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_col(Board)))))), get_invalid_sum_row(row_add_ass(row_delete_noopt(col_delete_dup(Board, get_empty_cell_col(Board)), -1), get_invalid_sum_row(Board))))))))), get_invalid_sum_row(row_add_ass(row_delete_noopt(col_add_ass(col_delete_dup(Board, -1), get_invalid_sum_col(Board)), -1), get_invalid_sum_row(row_add_ass(row_add_ass(row_delete_noopt(col_delete_dup(Board, get_empty_cell_col(Board)), -1), get_invalid_sum_row(Board)), get_invalid_sum_row(row_add_ass(col_delete_dup(Board, -1), get_invalid_sum_row(row_add_ass(row_add_ass(col_delete_dup(Board, -1), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, -1), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_col(Board)))), get_invalid_sum_row(col_add_ass(row_add_ass(row_add_ass(row_delete_noopt(Board, get_empty_cell_row(Board)), get_invalid_sum_row(Board)), get_invalid_sum_row(put_mandatory_ass(Board))), get_invalid_sum_col(Board))))))))))))))


def calc_sol_while(Board):
    i = 0
    while not board_is_ok(Board) and i <= 10:
        Board = row_add_ass(row_add_ass(col_add_ass(put_mandatory_ass(Board), get_empty_cell_col(col_delete_dup(col_delete_noopt(row_delete_ass(Board, -1), get_invalid_sum_col(Board)), get_invalid_sum_col(Board)))), get_empty_cell_row(col_delete_dup(col_add_ass(Board, -1), get_has_dup_col(Board)))), get_invalid_sum_row(col_delete_ass(Board, -1)))
        i+=1

    return Board


if __name__ == '__main__':
    board = get_boards()[0]
    # print(board.assignment, board.eval_fitness_on_board(0.2, 0.2, 0.2, 0.2, 0.2))
    assigned_board = calc_sol_while(board)
    # assigned_board = board
    # assigned_board.assignment = [[1,5,8],[9,8,3,7],[8,9],[3,1],[2,4,9,6],[8,9,6]]
    print(np.mean([calc_sol_while(b).eval_fitness_on_board(0.2, 0.2, 0.2, 0.2, 0.2) for b in get_boards()]))
    print(assigned_board.assignment, assigned_board.eval_fitness_on_board(0.2, 0.2, 0.2, 0.2, 0.2))
    cells = []
    for cell in board.cells:
        cells.append(cell.opt_ass)
    print(cells)
