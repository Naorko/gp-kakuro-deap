import random
import functools


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~transformations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def get_col_ass(board, col_num):
    return [board.assignment[i, j] for i, j in board.cols_map[col_num]]


def col_dups_idx(board, col_num):
    col_ass = get_col_ass(board, col_num)
    dup_indexes = [[] * 9]
    for idx, ass in enumerate(col_ass):
        dup_indexes[ass].append(idx)

    return [dups for dups in dup_indexes if len(dups) > 1]


def has_dups(board, col_num):
    return len(list(filter(lambda arr: len(arr) > 0, col_dups_idx(board, col_num)))) > 0


def col_is_equal_sum(board, col_num):
    col_ass = get_col_ass(board, col_num)
    return sum(col_ass) == board.cols_sum[col_num]
