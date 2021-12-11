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

        self.assignment = [[Board.EMPTY_CELL] * row_size for row_size in rows_size] if rows_size else None

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
            rows_sum_penalty += row_sum_penalty

            num_occurrences = [0] * 11
            for row_val in [ass for ass in self.assignment[row_i]]:
                num_occurrences[row_val - 1] += 1
            row_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            row_dup_penalty /= dup_penalty(len(self.assignment[row_i]))
            rows_dup_penalty += row_dup_penalty

        rows_dup_penalty /= len(self.rows_size)
        rows_sum_penalty /= len(self.rows_size)

        # cols penalty
        cols = [[self.assignment[row_i][cell_i] for row_i, cell_i in col_idx] for col_idx in self.cols_map]
        for col_i, col_vals in enumerate(cols):
            board_col_sum = sum([ass for ass in col_vals if ass != Board.EMPTY_CELL])
            col_penalty = abs(board_col_sum - self.cols_sum[col_i])
            col_penalty /= max((len(self.cols_map[col_i]) * 9 - self.cols_sum[col_i]), self.cols_sum[col_i])
            cols_sum_penalty += col_penalty

            num_occurrences = [0] * 11
            for col_val in [ass for ass in col_vals]:
                num_occurrences[col_val - 1] += 1
            col_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            col_dup_penalty /= dup_penalty(len(col_vals))
            cols_dup_penalty += col_dup_penalty

        cols_sum_penalty /= len(cols)
        cols_dup_penalty /= len(cols)

        # unassignment cells penalty
        num_of_cells = sum(self.rows_size)
        unassignment_penalty = sum([1 if ass == Board.EMPTY_CELL else 0 for ass in sum(self.assignment, [])])
        unassignment_penalty /= num_of_cells

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
