import numpy as np

from board_utils import get_opt_matrix_lst, EMPTY_CELL, get_poss_ass


class Row(int):
    pass


class Col(int):
    pass


class Cell(object):

    def __init__(self, row_i=None, row_j=None, ass=None, opt_ass=None, col_i=None):
        self.row_i = row_i
        self.row_j = row_j
        self.ass = ass
        self.opt_ass = opt_ass
        self.col_i = col_i


class Board(object):

    def __init__(self, board_size=None, rows_size=None, rows_sum=None, rows_opt=None, cols_sum=None, cols_map=None,
                 cols_opt=None, rows_map=None):
        self.board_size = board_size
        self.rows_size = rows_size
        self.rows_sum = rows_sum
        self.rows_opt = rows_opt
        self.cols_sum = cols_sum
        self.cols_map = cols_map
        self.cols_opt = cols_opt
        self.rows_map = rows_map
        self.cols_size = [len(col_map) for col_map in self.cols_map]
        self.size = sum(rows_size)
        self.assignment = [[EMPTY_CELL] * row_size for row_size in rows_size] if rows_size else None
        self.row_smart_opt = get_opt_matrix_lst(self.rows_opt)
        self.col_smart_opt = get_opt_matrix_lst(self.cols_opt)
        self.cells = self.init_cells()
        self.last_ass = []

    def init_cells(self):
        cells = dict()
        for row_i, row in enumerate(self.assignment):
            for row_j, ass in enumerate(row):
                col_i = self.get_col_by_cell(row_i, row_j)
                cell = Cell(row_i, row_j, ass, None, col_i)
                opt_ass = get_poss_ass(self, cell)
                cell.opt_ass = opt_ass
                cells[(row_i, row_j)] = cell
        return cells

    def set_assignment(self, cell: Cell, ass: int):
        self.assignment[cell.row_i][cell.row_j] = ass
        cell.ass = ass

    def get_col_by_cell(self, row_i, row_j):
        for col_i, col_map in enumerate(self.cols_map):
            if (row_i, row_j) in col_map:
                return col_i

    def get_col_ass(self, col_idx):
        return [self.assignment[i][j] for i, j in self.cols_map[col_idx]]

    def set_col_ass(self, col_idx, col_ass):
        for col_i, (i, j) in enumerate(self.cols_map[col_idx]):
            self.assignment[i][j] = col_ass[col_i]

    def eval_fitness_on_board(self, rows_sum_weight, rows_dup_weight, cols_sum_weight, cols_dup_weight,
                              unassignment_weight):
        # arithmetic sequence sum
        def dup_penalty(n, d=1):
            # return n-1 if n else 0
            return (d * n * (n - 1)) / 2

        def nthroot(a, n=6):
            return np.power(a, (1 / n))

        rows_sum_penalty = 0
        rows_dup_penalty = 0
        cols_sum_penalty = 0
        cols_dup_penalty = 0

        # rows penalty
        for row_i, row_size in enumerate(self.rows_size):
            board_row_sum = sum([ass for ass in self.assignment[row_i] if ass != EMPTY_CELL])
            row_sum_penalty = abs(board_row_sum - self.rows_sum[row_i])
            row_sum_penalty /= max((self.rows_size[row_i] * 9 - self.rows_sum[row_i]), self.rows_sum[row_i])
            row_sum_penalty = nthroot(row_sum_penalty)
            row_sum_penalty = (self.rows_size[row_i] / self.size) * row_sum_penalty
            rows_sum_penalty += row_sum_penalty

            num_occurrences = [0] * 11
            for row_val in [ass for ass in self.assignment[row_i]]:
                num_occurrences[row_val - 1] += 1
            row_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            row_dup_penalty /= dup_penalty(len(self.assignment[row_i]))
            row_dup_penalty = nthroot(row_dup_penalty)
            row_dup_penalty = (self.rows_size[row_i] / self.size) * row_dup_penalty
            rows_dup_penalty += row_dup_penalty


        # cols penalty
        cols = [[self.assignment[row_i][cell_i] for row_i, cell_i in col_idx] for col_idx in self.cols_map]
        for col_i, col_vals in enumerate(cols):
            board_col_sum = sum([ass for ass in col_vals if ass != EMPTY_CELL])
            col_sum_penalty = abs(board_col_sum - self.cols_sum[col_i])
            col_sum_penalty /= max((len(self.cols_map[col_i]) * 9 - self.cols_sum[col_i]), self.cols_sum[col_i])
            col_sum_penalty = nthroot(col_sum_penalty)
            col_sum_penalty = (self.cols_size[col_i] / self.size) * col_sum_penalty
            cols_sum_penalty += col_sum_penalty

            num_occurrences = [0] * 11
            for col_val in [ass for ass in col_vals]:
                num_occurrences[col_val - 1] += 1
            col_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            col_dup_penalty /= dup_penalty(len(col_vals))
            col_dup_penalty = nthroot(col_dup_penalty)
            col_dup_penalty = (self.cols_size[col_i] / self.size) * col_dup_penalty
            cols_dup_penalty += col_dup_penalty

        # unassignment cells penalty
        unassignment_penalty = sum([1 if ass == EMPTY_CELL else 0 for ass in sum(self.assignment, [])])
        unassignment_penalty /= self.size
        unassignment_penalty = nthroot(unassignment_penalty)

        total_penalty = rows_sum_weight * rows_sum_penalty + rows_dup_weight * rows_dup_penalty \
                        + cols_sum_weight * cols_sum_penalty + cols_dup_weight * cols_dup_penalty \
                        + unassignment_weight * unassignment_penalty
        return total_penalty
