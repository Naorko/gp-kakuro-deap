import random
import functools
import board

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


def assign_with_cond(board):
    for row_idx, row in enumerate(board.assignment):
        if row_is_not_assigned(row):
            board.assignment[row_idx] = board.rows_opt[row_idx, random.randint(len(board.rows_opt[row_idx]))]
            if True in [col_has_dups(board, col_num) for col_num in get_cross_cols(board, row_idx)]:
                pass # akol avod naor, ani modia al prisha. toda.



def while_no_dups_node(func,condition):
    while not(False in [col_has_dups(,]):
        func()

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
def get_invalid_row(board: board.Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if board.EMPTY_CELL in row_ass or sum(row_ass) != board.rows_sum[row_idx] or row_has_dups(board, row_idx):
            return row_idx

    return -1
