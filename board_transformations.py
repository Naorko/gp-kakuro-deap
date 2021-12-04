import random
import functools
import numpy as np
from board import Board, Col, Row


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~transformations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# rows_transformation
from board_translator import get_boards


def row_add_ass(board: Board, row_idx: Row):
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


# all boards transformation
def put_mandatory_ass(board):
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




def get_idx_ass(board, col_idx, col_cell):
    col_map = board.cols_map[col_idx]
    col_cell = col_map[col_cell]
    return col_cell

# cols_trsnaformation
def col_trans(board, col_idx: Col):
    col_ass = get_col_ass(board, col_idx)
    col_ass_set = set(col_ass)
    if board.EMPTY_CELL not in col_ass:
        return board

    col_ass_set.remove(board.EMPTY_CELL)
    cols_opt = [opt for opt in board.cols_opt[col_idx] if col_ass_set.issubset(set(opt))]
    opt_ass = set(random.choice(cols_opt))
    diff_opt_ass = list(opt_ass - col_ass_set)
    random.shuffle(diff_opt_ass)
    for col_cell, ass in enumerate(col_ass):
        if ass == board.EMPTY_CELL:
            row_idx, row_cell = get_idx_ass(board, col_idx, col_cell)
            board.assignment[row_idx][row_cell] = diff_opt_ass.pop()
    return board


# connections nodes
def analyze_row(board):
    return row


def analyze_col(board):
    return col


def init_ind(board):
    pass


def delete_row_ass(board):
    pass


def rows_assignment(board):
    pass


def column_assignment(board):
    pass


def random_opt_row_ass(board):
    rand_row = random.randint(len(board.rows_size))
    row_ass = random.choice(board.rows_opt[rand_row])
    board.assignment[rand_row] = row_ass
    return board


def smart_col_sum_fix(board):
    pass


def smart_col_dup_fix(board):
    rand_col = random.randint(len(board.cols_sum))
    for assign in board.cols_map[rand_col]:
        pass



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~utils~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_cross_cols(board, row_idx):
    return [cols[0] for cols in board.rows_map[row_idx]]


def get_rand_opt_row_ass(board, row_idx):
    return list(board.rows_opt[row_idx,])


def row_is_not_assigned(row):
    return -1 in row


def get_col_ass(board, col_num):
    return [board.assignment[i][j] for i, j in board.cols_map[col_num]]


def ass_dups_idx(assignment):
    dup_indexes = [[] for _ in range(10)]
    for idx, ass in enumerate(assignment):
        if ass != board.EMPTY_CELL:
            dup_indexes[ass].append(idx)
    return [dups for dups in dup_indexes if len(dups) > 1]

def col_dups_idx(board, col_num):
    col_ass = get_col_ass(board, col_num)
    return ass_dups_idx(col_ass)

def row_dups_idx(board, row_num):
    row_ass = board.assignment[row_num]
    return ass_dups_idx(row_ass)


def col_has_dups(board, col_num):
    return len(list(filter(lambda arr: len(arr) > 0, col_dups_idx(board, col_num)))) > 0

def row_has_dups(board, row_num):
    return len(list(filter(lambda arr: len(arr) > 0, row_dups_idx(board, row_num)))) > 0


def col_is_greater_sum(board, col_num):
    col_ass = get_col_ass(board, col_num)
    return sum([ass for ass in col_ass if ass != -1]) > board.cols_sum[col_num]


def col_is_smaller_sum(board, col_num):
    # TODO:consider to add smart smaller calculator
    col_ass = get_col_ass(board, col_num)
    if -1 in col_ass:
        return False
    return sum(col_ass) < board.cols_sum[col_num]

Row = int
def get_invalid_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if board.EMPTY_CELL in row_ass or sum(row_ass) != board.rows_sum[row_idx] or row_has_dups(board, row_idx):
            return row_idx

    return -1

def get_opt_assignment(board, row_idx):
    for opt in board.rows_opt[row_idx]:
        yield opt


def is_not_full_assigned(board):
    return -1 in np.array(board.assignment).flatten()


def board_is_ok(board):
    for col_num in range(len(board.cols_sum)):
        if col_is_smaller_sum(board, col_num) or col_is_greater_sum(board, col_num) or has_dups(board, col_num):
            return False
    return True


if __name__ == '__main__':
    board = get_boards()[0]
    print(board.assignment)
    for i in range(6):
        put_mandatory_ass(board)
        print('after mand: ',board.assignment)
        col_trans(board, 0)
        print('after row_and_ass: ',board.assignment)
