class Board(object):
    EMPTY_CELL = -1

    def __init__(self, rows_size, rows_sum, rows_opt, cols_sum, cols_map, cols_opt, rows_map):
        self.rows_size = rows_size
        self.rows_sum = rows_sum
        self.rows_opt = rows_opt
        self.cols_sum = cols_sum
        self.cols_map = cols_map
        self.cols_opt = cols_opt
        self.rows_map = rows_map

        self.assignment = [[Board.EMPTY_CELL] * row_size for row_size in rows_size]


class Row(int):
    pass


class Col(int):
    pass
