from typing import NewType


class Board(object):
    EMPTY_CELL = -1

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
        self.assignment = [[Board.EMPTY_CELL] * row_size for row_size in rows_size] if rows_size else None
        self.cells = self.init_cells()

    def init_cells(self):
        cells = dict
        for row_i, row in enumerate(self.assignment):
            for row_j, ass in enumerate(row):
                row_opts = self.rows_opt[row_i]
                col_i = self.get_col_by_cell(row_i, row_j)
                cols_opts = self.cols_opt[col_i]
                unioned_col_opt = set.union(*[set(opt) for opt in cols_opts])
                unioned_row_opt = set.union(*[set(opt) for opt in row_opts])
                intersect_opt = unioned_col_opt.intersection(unioned_row_opt)
                cells[(row_i, row_j)] = Cell(row_i, row_j, ass, intersect_opt, col_i)
        return cells

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
            return (d * n * (n - 1)) / 2

        rows_sum_penalty = 0
        rows_dup_penalty = 0
        cols_sum_penalty = 0
        cols_dup_penalty = 0
        # rows penalty
        for row_i, row_size in enumerate(self.rows_size):
            board_row_sum = sum([ass for ass in self.assignment[row_i] if ass != Board.EMPTY_CELL])
            row_sum_penalty = abs(board_row_sum - self.rows_sum[row_i])
            row_sum_penalty /= max((self.rows_size[row_i] * 9 - self.rows_sum[row_i]), self.rows_sum[row_i])
            rows_sum_penalty += (self.rows_size[row_i] / self.size) * row_sum_penalty

            num_occurrences = [0] * 11
            for row_val in [ass for ass in self.assignment[row_i]]:
                num_occurrences[row_val - 1] += 1
            row_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            row_dup_penalty /= dup_penalty(len(self.assignment[row_i]))
            rows_dup_penalty += (self.rows_size[row_i] / self.size) * row_dup_penalty

        # rows_dup_penalty /= len(self.rows_size)
        # rows_sum_penalty /= len(self.rows_size)

        # cols penalty
        cols = [[self.assignment[row_i][cell_i] for row_i, cell_i in col_idx] for col_idx in self.cols_map]
        for col_i, col_vals in enumerate(cols):
            board_col_sum = sum([ass for ass in col_vals if ass != Board.EMPTY_CELL])
            col_penalty = abs(board_col_sum - self.cols_sum[col_i])
            col_penalty /= max((len(self.cols_map[col_i]) * 9 - self.cols_sum[col_i]), self.cols_sum[col_i])
            cols_sum_penalty += (self.cols_size[col_i] / self.size) * col_penalty

            num_occurrences = [0] * 11
            for col_val in [ass for ass in col_vals]:
                num_occurrences[col_val - 1] += 1
            col_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            col_dup_penalty /= dup_penalty(len(col_vals))
            cols_dup_penalty += (self.cols_size[col_i] / self.size) * col_dup_penalty

        # cols_sum_penalty /= len(cols)
        # cols_dup_penalty /= len(cols)

        # unassignment cells penalty
        unassignment_penalty = sum([1 if ass == Board.EMPTY_CELL else 0 for ass in sum(self.assignment, [])])
        unassignment_penalty /= self.size

        total_penalty = rows_sum_weight * rows_sum_penalty + rows_dup_weight * rows_dup_penalty \
                        + cols_sum_weight * cols_sum_penalty + cols_dup_weight * cols_dup_penalty \
                        + unassignment_weight * unassignment_penalty
        return total_penalty


# class Attr(int):
#     pass


class Row(int):
    pass


class Col(int):
    pass


class Cell():

    def __init__(self, row_i, row_j, ass, opt_ass, col_i):
        self.row_i = row_i
        self.row_j = row_j
        self.ass = ass
        self.opt_ass = opt_ass
        self.col_i = col_i
