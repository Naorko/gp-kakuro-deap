from board import Board


def get_idx_ass(board, col_idx, col_cell):
    col_map = board.cols_map[col_idx]
    col_cell = col_map[col_cell]
    return col_cell


def get_cross_cols(board, row_idx):
    return [cols[0] for cols in board.rows_map[row_idx]]


def get_rand_opt_row_ass(board, row_idx):
    return list(board.rows_opt[row_idx,])


def row_is_not_assigned(row):
    return Board.EMPTY_CELL in row


def get_col_ass(board, col_num):
    return [board.assignment[i][j] for i, j in board.cols_map[col_num]]


def ass_dups_idx(assignment):
    dup_indexes = [[] for _ in range(10)]
    for idx, ass in enumerate(assignment):
        if ass != Board.EMPTY_CELL:
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
    col_ass = get_col_ass(board, col_num)
    if -1 in col_ass:
        return False
    return sum(col_ass) < board.cols_sum[col_num]


def get_opt_assignment(board, row_idx):
    for opt in board.rows_opt[row_idx]:
        yield opt
