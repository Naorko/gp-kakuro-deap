import itertools
import json
import numpy as np
import os

from board import Board

WALL = 'x/x'
EMPTY_CELL = 'x'
BOARDS_FILE_PATH = os.path.join('data', 'data_generated_easy.ecl')

BOARDI = """data(generated,1,-,6,6,
[](
[](x/x, x/x, 35/x, 17/x, 16/x, x/x),
   [](x/x, 17/14, x, x, x, x/x),
   [](x/27, x, x, x, x, 4/x),
   [](x/17, x, x, 17/4, x, x),
   [](x/x, x/21, x, x, x, x),
   [](x/x, x/23, x, x, x, x/x)
))"""


def load_board_str(board_string):
    board_size = tuple(int(b) for b in board_string.split(',')[3:5])
    board = ','.join(board_string.split(',')[5:]).replace(' ', '').replace('\n', '')[:-1]
    board = board.replace('[](', '[').replace(')', ']')  # Change parenthesis
    bx = '[[' + '],['.join([','.join('"' + x + '"' for x in row.split(',')) for row in
                            ('],' + board[1:-1] + ',[').split('],[')[1:-1]]) + ']]'

    return board_size, np.matrix(json.loads(bx))


def label_rows(board):
    board = board.copy()
    n, m = board.shape
    row_idx = -1
    col_idx = 0
    in_row = False
    rows_size = []
    for i in range(n):
        for j in range(m):
            if board[i, j] == EMPTY_CELL:
                if not in_row:
                    in_row = True
                    row_idx += 1
                    col_idx = 0

                board[i, j] = f'{row_idx}~{col_idx}'
                col_idx += 1

            else:
                if in_row:
                    rows_size.append(col_idx)
                    in_row = False

    if in_row:
        rows_size.append(col_idx)

    return board, rows_size


def eval_cell(cell_str: str):
    if cell_str in [WALL, EMPTY_CELL]:
        return cell_str

    if '~' in cell_str:
        idxs = cell_str.split('~')
        return tuple(int(idx) for idx in idxs)

    cell_arr = cell_str.split('/')
    if cell_arr[0] == 'x':
        return cell_arr[1],

    if cell_arr[1] == 'x':
        return cell_arr[0],

    return tuple(cell_arr)


def extract_sum(board, is_transposed=False):
    sum_idx = 0 if is_transposed else -1
    n, m = board.shape

    sums = []
    for i in range(n):
        for j in range(m - 1):
            cell_val = eval_cell(board[i, j])
            next_cell_val = eval_cell(board[i, j + 1])
            if next_cell_val == EMPTY_CELL and isinstance(cell_val, tuple):
                sums.append(int(cell_val[sum_idx]))

    return sums


def extract_cols(board, labeled_board):
    board, labeled_board = board.T, labeled_board.T
    n, m = board.shape

    cols, cur_col = [], []
    for i in range(n):
        for j in range(m):
            if board[i, j] == EMPTY_CELL:
                cell_idx = eval_cell(labeled_board[i, j])
                cur_col.append(cell_idx)
            else:
                if cur_col:
                    cols.append(cur_col)
                    cur_col = []
    if cur_col:
        cols.append(cur_col)

    return cols


def extract_row_map(row, cols_map):
    row_map = []
    for col_idx, col in enumerate(cols_map):
        for allele_idx, mapping in enumerate(col):
            if mapping[0] == row:
                row_map.append((col_idx, allele_idx))
    return row_map


def extract_board_params(board_size, board) -> Board:
    labeled_board, rows_size = label_rows(board)
    rows_sum = extract_sum(board)
    rows_opt = [get_parts(row_sum, row_size) for row_sum, row_size in zip(rows_sum, rows_size)]
    cols_sum = extract_sum(board.T, is_transposed=True)
    cols_map = extract_cols(board, labeled_board)
    cols_opt = [get_parts(col_sum, col_size) for col_sum, col_size in zip(cols_sum, [len(col) for col in cols_map])]
    rows_map = [extract_row_map(row, cols_map) for row in range(len(rows_size))]
    # Should now cross-optimize rows_opt and cols_opt
    board = Board(board_size, rows_size, rows_sum, rows_opt, cols_sum, cols_map, cols_opt, rows_map)

    return board


def get_parts(row_sum, row_size):
    possible_nums = range(1, 10)
    all_perm = itertools.combinations(possible_nums, row_size)
    parts = []
    for perm in all_perm:
        if sum(perm) == row_sum:
            parts.append(perm)
    return parts


def get_boards():
    with open(BOARDS_FILE_PATH, 'r') as board_file:
        boards_str = board_file.read()
        boards = boards_str.split('.')
        boards = [extract_board_params(*load_board_str(b)) for b in boards]
        return boards


if __name__ == '__main__':
    boards = get_boards()
    for i, b in enumerate(boards):
        if len(b.cols_sum) != len(b.cols_map) or len(b.cols_sum) != len(b.cols_opt) or len(b.cols_opt) != len(b.cols_map):
            print(f'Error in board {i}')
            # for board_attr, attr_val in b.__dict__.items():
            #     print(f'{board_attr}:', len(attr_val))
    print(f'There are {len(boards)} in the boards file')
    print()
    board = boards[34]
    for board_attr, attr_val in board.__dict__.items():
        print(f'{board_attr}:', attr_val)
