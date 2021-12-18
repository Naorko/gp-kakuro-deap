import numpy as np
import pandas as pd

EMPTY_CELL = -1

def get_idx_ass(board, col_idx, col_cell):
    col_map = board.cols_map[col_idx]
    col_cell = col_map[col_cell]
    return col_cell


def get_cross_cols(board, row_idx):
    return [cols[0] for cols in board.rows_map[row_idx]]


def get_rand_opt_row_ass(board, row_idx):
    return list(board.rows_opt[row_idx,])


def row_is_not_assigned(row):
    return EMPTY_CELL in row


def ass_dups_idx(assignment):
    dup_indexes = [[] for _ in range(10)]
    for idx, ass in enumerate(assignment):
        if ass != EMPTY_CELL:
            dup_indexes[ass].append(idx)
    return [dups for dups in dup_indexes if len(dups) > 1]


def col_dups_idx(board, col_num):
    col_ass = board.get_col_ass(col_num)
    return ass_dups_idx(col_ass)


def row_dups_idx(board, row_num):
    row_ass = board.assignment[row_num]
    return ass_dups_idx(row_ass)


def col_has_dups(board, col_num):
    return len(list(filter(lambda arr: len(arr) > 0, col_dups_idx(board, col_num)))) > 0


def row_has_dups(board, row_num):
    return len(list(filter(lambda arr: len(arr) > 0, row_dups_idx(board, row_num)))) > 0


def col_is_greater_sum(board, col_num):
    col_ass = board.get_col_ass(col_num)
    return sum([ass for ass in col_ass if ass != -1]) > board.cols_sum[col_num]


def col_is_smaller_sum(board, col_num):
    col_ass = board.get_col_ass(col_num)
    if -1 in col_ass:
        return False
    return sum(col_ass) < board.cols_sum[col_num]


def get_opt_assignment(board, row_idx):
    for opt in board.rows_opt[row_idx]:
        yield opt

#
# def get_ass_for_cell(board, row, row_i):
#     row_ass = board.assignment[row_i]
#     for col_map in board.cols_map:
#         for map in col_map:
#             if map[0]==row_i
#     col_ass = board.get_col_ass()


def remove_dups_from_ass(ass):
    num_acc = [0] * 9
    for val in ass:
        if val != EMPTY_CELL:
            num_acc[val - 1] += 1

    return [val if val == EMPTY_CELL or num_acc[val - 1] <= 1 else EMPTY_CELL for val in ass]


def get_opt_matrix_lst(opts):
    smart_opt_lst = []
    for val_opt in opts:
        opt_length = len(val_opt)
        opt_matrix = np.zeros((opt_length, 9), dtype=np.ubyte)
        for row_idx, opt in enumerate(val_opt):
            for val in opt:
                opt_matrix[row_idx][val-1] = 1

        opt_df = pd.DataFrame(opt_matrix, columns=[f'v_{v}' for v in range(1, 10)])
        smart_opt_lst.append(opt_df)

    return smart_opt_lst


def get_poss_opt_by_ass(opt_df, ass):
    ass = [val for val in ass if val != EMPTY_CELL]
    if ass:
        query = ' and '.join([f'v_{a} == 1' for a in ass])
        df = opt_df.query(query)
    else:
        df = opt_df

    return [(i+1, val) for i, val in enumerate(df.sum()) if val > 0 and i+1 not in ass]


def get_poss_ass(board, cell):
    if cell.ass != EMPTY_CELL:
        return []

    row_poss = get_poss_opt_by_ass(board.row_smart_opt[cell.row_i], board.assignment[cell.row_i])
    col_poss = get_poss_opt_by_ass(board.col_smart_opt[cell.col_i], board.get_col_ass(cell.col_i))
    poss_ass = [(i, j+y) for i, j in row_poss for x, y in col_poss if i == x]

    return poss_ass


def update_cells_by_rows_cols(board, rows=set(), cols=set()):
    cells = set()
    for row in rows:
        cells |= {(row, j) for j in range(board.rows_size[row])}

    for col in cols:
        cells |= set(board.cols_map[col])

    for cell_k in cells:
        cell = board.cells[cell_k]
        cell.opt_ass = get_poss_ass(board, cell)
