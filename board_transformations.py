import random
from board import Board, Col, Row
from board_translator import get_boards

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~transformations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from board_utils import get_idx_ass, get_col_ass, row_has_dups, col_has_dups

INVALID_IDX = -1


# rows transformation

def row_add_ass(board: Board, row_idx: Row) -> Board:
    if row_idx == INVALID_IDX:
        return board
    row_ass = set(board.assignment[row_idx])
    if board.EMPTY_CELL not in row_ass:
        return board

    row_ass.remove(board.EMPTY_CELL)
    rows_opt = [opt for opt in board.rows_opt[row_idx] if row_ass.issubset(set(opt))]
    opt_ass = set(random.choice(rows_opt))
    diff_opt_ass = list(opt_ass - row_ass)
    random.shuffle(diff_opt_ass)
    for idx, ass in enumerate(board.assignment[row_idx]):
        if ass == board.EMPTY_CELL:
            board.assignment[row_idx][idx] = diff_opt_ass.pop()
    return board


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


def random_opt_row_ass(board: Board) -> Board:
    rand_row = random.randint(len(board.rows_size))
    row_ass = random.choice(board.rows_opt[rand_row])
    board.assignment[rand_row] = row_ass
    return board


# cols transformations

def col_trans(board: Board, col_idx: Col) -> Board:
    if col_idx == INVALID_IDX:
        return board
    col_ass = get_col_ass(board, col_idx)
    col_ass_set = set(col_ass)
    if Board.EMPTY_CELL not in col_ass:
        return board

    col_ass_set.remove(Board.EMPTY_CELL)
    cols_opt = [opt for opt in board.cols_opt[col_idx] if col_ass_set.issubset(set(opt))]
    opt_ass = set(random.choice(cols_opt))
    diff_opt_ass = list(opt_ass - col_ass_set)
    random.shuffle(diff_opt_ass)
    for col_cell, ass in enumerate(col_ass):
        if ass == Board.EMPTY_CELL:
            row_idx, row_cell = get_idx_ass(board, col_idx, col_cell)
            board.assignment[row_idx][row_cell] = diff_opt_ass.pop()
    return board


# analyze nodes

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
        col_ass = get_col_ass(board, col_idx)
        if Board.EMPTY_CELL in col_ass or sum(col_ass) != board.cols_sum[col_idx] or col_has_dups(board, col_idx):
            return col_idx

    return Col(INVALID_IDX)


def board_is_ok(board: Board) -> bool:
    col_ok, row_ok = get_invalid_col(board), get_invalid_row(board)
    return col_ok == INVALID_IDX and row_ok == INVALID_IDX


if __name__ == '__main__':
    board = get_boards()[0]
    print(board.assignment)
    for i in range(6):
        put_mandatory_ass(board)
        print('after mand: ', board.assignment)
        col_trans(board, 0)
        print('after row_and_ass: ', board.assignment)
