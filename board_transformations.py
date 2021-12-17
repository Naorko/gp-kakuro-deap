import random

import numpy as np

from board import Board, Col, Row, Cell
from board_translator import get_boards

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~transformations~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from board_utils import get_idx_ass, row_has_dups, col_has_dups, remove_dups_from_ass, get_poss_ass, EMPTY_CELL

INVALID_IDX = -1


def id(x):
    return x


# all board transformation


def put_mandatory_ass(board: Board) -> Board:
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


def put_mandatory_ass_cell(board: Board) -> Board:
    for cell in board.cells.values():
        poss_lst = get_poss_ass(board, cell)
        if len(poss_lst) == 1:
            (ass, _), = poss_lst
            board.set_assignment(cell, ass)

    return board


# cells transformation
def cell_add_max_ass(board: Board, cell: Cell) -> Board:
    if cell is None:
        return board

    poss_ass = get_poss_ass(board, cell)
    if poss_ass:
        ass_idx = np.argmax([cnt for _, cnt in poss_ass])
        ass, _ = poss_ass[ass_idx]
        board.set_assignment(cell, ass)
        board.last_ass.append(((cell.row_i, cell.row_j), ass))

    return board


def cell_add_min_ass(board: Board, cell: Cell) -> Board:
    if cell is None:
        return board

    poss_ass = get_poss_ass(board, cell)
    if poss_ass:
        ass_idx = np.argmin([cnt for _, cnt in poss_ass])
        ass, _ = poss_ass[ass_idx]
        board.set_assignment(cell, ass)
        board.last_ass.append(((cell.row_i, cell.row_j), ass))

    return board


# cells analyze

def get_largest_opt(board: Board) -> Cell:
    max_opt_cell = None
    max_opt = -1
    for _, cell in board.cells.items():
        if cell.ass == EMPTY_CELL and len(get_poss_ass(board, cell)) > max_opt:
            max_opt_cell = cell
            max_opt = len(cell.opt_ass)
    return max_opt_cell


def get_smallest_opt(board: Board) -> Cell:
    min_opt_cell = None
    min_opt = 10
    for _, cell in board.cells.items():
        if cell.ass == EMPTY_CELL and len(get_poss_ass(board, cell)) < min_opt:
            min_opt_cell = cell
            min_opt = len(cell.opt_ass)
    return min_opt_cell


def get_most_empty_row(board: Board) -> Cell:
    row_idx = None
    max_empty_cell = -1
    for row_i, row in enumerate(board.assignment):
        row_empty_cells = len([cell for cell in row if cell == EMPTY_CELL])
        if max_empty_cell < row_empty_cells:
            row_idx = row_i
            max_empty_cell = row_empty_cells

    empty_cells_idxs = [i for i, val in enumerate(board.assignment[row_idx]) if val == EMPTY_CELL]
    if empty_cells_idxs:
        row_j = random.choice(empty_cells_idxs)
        return board.cells[(row_idx, row_j)]

    return None


def get_least_empty_row(board: Board) -> Cell:
    row_idx = None
    min_empty_cell = 10
    for row_i, row in enumerate(board.assignment):
        row_empty_cells = len([cell for cell in row if cell == EMPTY_CELL])
        if min_empty_cell > row_empty_cells:
            row_idx = row_i
            min_empty_cell = row_empty_cells

    empty_cells_idxs = [i for i, val in enumerate(board.assignment[row_idx]) if val == EMPTY_CELL]
    if empty_cells_idxs:
        row_j = random.choice(empty_cells_idxs)
        return board.cells[(row_idx, row_j)]

    return None


# fallback cells
def backtrack_cells(board: Board, steps: int) -> Board:
    board_steps = min(steps, len(board.last_ass))
    for _ in range(board_steps):
        cell_i_j, ass = board.last_ass.pop()
        cell = board.cells[cell_i_j]
        board.set_assignment(cell, EMPTY_CELL)

    return board


def fallback_no_opt(board: Board) -> Board:
    if board.last_ass:
        for (cell_row_i, cell_row_j), cell in board.cells.items():
            if cell.ass == EMPTY_CELL and len(get_poss_ass(board, cell)) == 0:
                row_cells = [(cell_row_i, j) for j in range(board.rows_size[cell_row_i])]
                col_cells = board.cols_map[cell.col_i]
                for i, (cell_ass, _) in enumerate(board.last_ass):
                    if cell_ass in row_cells or cell_ass in col_cells:
                        break

                need_to_delete = board.last_ass[i:]
                board.last_ass = board.last_ass[:i]
                for del_cell, _ in need_to_delete:
                    board.set_assignment(board.cells[del_cell], EMPTY_CELL)

    return board


# rows transformation

def row_add_ass(board: Board, row_idx: Row) -> Board:
    if row_idx == INVALID_IDX:
        return board
    row_ass = set(board.assignment[row_idx])
    if EMPTY_CELL not in row_ass:
        return board

    row_ass.remove(EMPTY_CELL)
    rows_opt = [opt for opt in board.rows_opt[row_idx] if row_ass.issubset(set(opt))]
    if not rows_opt:  # No valid optional assignment
        return board
    opt_ass = set(random.choice(rows_opt))
    diff_opt_ass = list(opt_ass - row_ass)
    random.shuffle(diff_opt_ass)
    for idx, ass in enumerate(board.assignment[row_idx]):
        if ass == EMPTY_CELL:
            board.assignment[row_idx][idx] = diff_opt_ass.pop()
    return board


def row_delete_ass(board: Board, row_idx: Row) -> Board:
    if row_idx != INVALID_IDX:
        board.assignment[row_idx] = [EMPTY_CELL] * board.rows_size[row_idx]
    return board


def row_delete_dup(board: Board, row_idx: Row) -> Board:
    if row_idx != INVALID_IDX:
        new_ass = remove_dups_from_ass(board.assignment[row_idx])
        board.assignment[row_idx] = new_ass

    return board


def row_delete_noopt(board: Board, row_idx: Row) -> Board:
    if row_idx != INVALID_IDX:
        opts = set(sum(board.rows_opt[row_idx], ()))
        ass = board.assignment[row_idx]
        new_ass = [val if val in opts else EMPTY_CELL for val in ass]
        board.assignment[row_idx] = new_ass

    return board


# cols transformations
def col_add_ass(board: Board, col_idx: Col) -> Board:
    if col_idx == INVALID_IDX:
        return board

    col_ass = board.get_col_ass(col_idx)
    col_ass_set = set(col_ass)

    if EMPTY_CELL not in col_ass:
        return board

    col_ass_set.remove(EMPTY_CELL)
    cols_opt = [opt for opt in board.cols_opt[col_idx] if col_ass_set.issubset(set(opt))]
    if not cols_opt:  # No valid optional assignment
        return board

    opt_ass = set(random.choice(cols_opt))
    diff_opt_ass = list(opt_ass - col_ass_set)
    random.shuffle(diff_opt_ass)
    for col_cell, ass in enumerate(col_ass):
        if ass == EMPTY_CELL:
            row_idx, row_cell = get_idx_ass(board, col_idx, col_cell)
            board.assignment[row_idx][row_cell] = diff_opt_ass.pop()
    return board


def col_delete_ass(board: Board, col_idx: Col) -> Board:
    if col_idx != INVALID_IDX:
        new_ass = [EMPTY_CELL] * len(board.cols_map[col_idx])
        board.set_col_ass(col_idx, new_ass)

    return board


def col_delete_dup(board: Board, col_idx: Col) -> Board:
    if col_idx != INVALID_IDX:
        new_ass = remove_dups_from_ass(board.get_col_ass(col_idx))
        board.set_col_ass(col_idx, new_ass)

    return board


def col_delete_noopt(board: Board, col_idx: Col) -> Board:
    if col_idx != INVALID_IDX:
        opts = set(sum(board.cols_opt[col_idx], ()))
        ass = board.get_col_ass(col_idx)
        new_ass = [val if val in opts else EMPTY_CELL for val in ass]
        board.set_col_ass(col_idx, new_ass)

    return board


# analyze nodes


def get_empty_cell_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if EMPTY_CELL in row_ass:
            return row_idx

    return Row(INVALID_IDX)


def get_has_dup_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        if row_has_dups(board, row_idx):
            return row_idx

    return Row(INVALID_IDX)


def get_invalid_sum_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if sum(row_ass) != board.rows_sum[row_idx]:
            return row_idx

    return Row(INVALID_IDX)


def get_empty_cell_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        col_ass = board.get_col_ass(col_idx)
        if EMPTY_CELL in col_ass:
            return col_idx

    return Col(INVALID_IDX)


def get_has_dup_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        if col_has_dups(board, col_idx):
            return col_idx

    return Col(INVALID_IDX)


def get_invalid_sum_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        col_ass = board.get_col_ass(col_idx)
        if sum(col_ass) != board.cols_sum[col_idx]:
            return col_idx

    return Col(INVALID_IDX)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~first experiment~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_invalid_row(board: Board) -> Row:
    rows_idxs = list(range(len(board.assignment)))
    random.shuffle(rows_idxs)
    for row_idx in rows_idxs:
        row_ass = board.assignment[row_idx]
        if EMPTY_CELL in row_ass or sum(row_ass) != board.rows_sum[row_idx] or row_has_dups(board, row_idx):
            return row_idx

    return Row(INVALID_IDX)


def get_invalid_col(board: Board) -> Col:
    cols_idxs = list(range(len(board.cols_sum)))
    random.shuffle(cols_idxs)
    for col_idx in cols_idxs:
        col_ass = board.get_col_ass(col_idx)
        if EMPTY_CELL in col_ass or sum(col_ass) != board.cols_sum[col_idx] or col_has_dups(board, col_idx):
            return col_idx

    return Col(INVALID_IDX)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~first experiment~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def board_is_ok(board: Board) -> bool:
    return (np.array([get_empty_cell_col(board), get_has_dup_col(board), get_invalid_sum_col(board),
                      get_empty_cell_row(board), get_has_dup_row(board),
                      get_invalid_sum_row(board)]) == INVALID_IDX).all()


# def calc_sol1(Board):
#     return row_add_ass(row_add_ass(put_mandatory_ass(Board), get_invalid_row(row_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(col_add_ass(Board, get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))))), get_invalid_row(row_add_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)))), -1), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(col_add_ass(Board, -1), get_invalid_row(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)))), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_row(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))))))))), get_invalid_row(row_add_ass(row_add_ass(row_add_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_row(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))))), get_invalid_row(row_add_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(row_add_ass(row_add_ass(put_mandatory_ass(col_add_ass(Board, -1)), -1), get_invalid_row(Board)))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(Board, get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(put_mandatory_ass(Board)))))), get_invalid_row(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))))))), get_invalid_row(col_add_ass(col_add_ass(row_add_ass(col_add_ass(put_mandatory_ass(Board), get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(Board))))), get_invalid_col(col_add_ass(Board, -1)))))), get_invalid_row(Board)), get_invalid_col(put_mandatory_ass(col_add_ass(Board, get_invalid_col(col_add_ass(Board, get_invalid_col(Board))))))), get_invalid_col(col_add_ass(Board, get_invalid_col(put_mandatory_ass(col_add_ass(col_add_ass(row_add_ass(row_add_ass(Board, -1), get_invalid_row(Board)), get_invalid_col(Board)), get_invalid_col(Board)))))))))))

def calc_sol2(Board):
    return cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(
        cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(put_mandatory_ass_cell(Board))),
        get_smallest_opt(put_mandatory_ass_cell(Board))),
                                             get_smallest_opt(cell_add_max_ass(Board, get_smallest_opt(Board)))),
                            get_most_empty_row(cell_add_max_ass(put_mandatory_ass_cell(put_mandatory_ass_cell(
                                cell_add_max_ass(put_mandatory_ass_cell(cell_add_max_ass(
                                    cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(Board)),
                                    get_smallest_opt(Board))), get_smallest_opt(
                                    put_mandatory_ass_cell(put_mandatory_ass_cell(cell_add_max_ass(Board, None))))))),
                                                                get_largest_opt(cell_add_max_ass(cell_add_max_ass(
                                                                    cell_add_max_ass(Board, get_smallest_opt(Board)),
                                                                    get_smallest_opt(Board)), get_smallest_opt(
                                                                    cell_add_max_ass(
                                                                        cell_add_max_ass(put_mandatory_ass_cell(Board),
                                                                                         get_smallest_opt(Board)),
                                                                        get_most_empty_row(cell_add_max_ass(Board,
                                                                                                            get_smallest_opt(
                                                                                                                Board))))))))))


def calc_sol_while(Board):
    i = 0
    while not board_is_ok(Board) and i <= 100:
        # Board = cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(put_mandatory_ass_cell(Board))), get_smallest_opt(put_mandatory_ass_cell(Board))), get_smallest_opt(cell_add_max_ass(Board, get_smallest_opt(Board)))), get_most_empty_row(cell_add_max_ass(put_mandatory_ass_cell(put_mandatory_ass_cell(cell_add_max_ass(put_mandatory_ass_cell(cell_add_max_ass(cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(Board)), get_smallest_opt(Board))), get_smallest_opt(put_mandatory_ass_cell(put_mandatory_ass_cell(cell_add_max_ass(Board, None))))))), get_largest_opt(cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(Board, get_smallest_opt(Board)), get_smallest_opt(Board)), get_smallest_opt(cell_add_max_ass(cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(Board)), get_most_empty_row(cell_add_max_ass(Board, get_smallest_opt(Board))))))))))
        # Board = cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(put_mandatory_ass_cell(Board))), get_smallest_opt(put_mandatory_ass_cell(Board))), get_smallest_opt(cell_add_max_ass(Board, get_smallest_opt(Board)))), get_most_empty_row(cell_add_max_ass(put_mandatory_ass_cell(cell_add_max_ass(backtrack_cells(Board, 2), get_least_empty_row(Board))), get_largest_opt(cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(Board, get_smallest_opt(Board)), get_smallest_opt(put_mandatory_ass_cell(Board))), get_smallest_opt(cell_add_max_ass(cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(Board)), get_most_empty_row(cell_add_max_ass(Board, get_smallest_opt(Board))))))))))
        Board = cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(put_mandatory_ass_cell(Board),
                                                                                    get_smallest_opt(
                                                                                        put_mandatory_ass_cell(
                                                                                            cell_add_max_ass(Board,
                                                                                                             get_smallest_opt(
                                                                                                                 Board))))),
                                                                   get_smallest_opt(
                                                                       cell_add_max_ass(backtrack_cells(Board, 7),
                                                                                        get_most_empty_row(Board)))),
                                                  get_smallest_opt(cell_add_max_ass(Board, get_smallest_opt(Board)))),
                                 get_most_empty_row(cell_add_max_ass(put_mandatory_ass_cell(put_mandatory_ass_cell(
                                     cell_add_max_ass(put_mandatory_ass_cell(cell_add_max_ass(
                                         cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(Board)),
                                         get_smallest_opt(Board))), get_smallest_opt(put_mandatory_ass_cell(
                                         put_mandatory_ass_cell(cell_add_max_ass(Board, None))))))), get_largest_opt(
                                     cell_add_max_ass(cell_add_max_ass(cell_add_max_ass(Board, get_smallest_opt(Board)),
                                                                       get_smallest_opt(Board)), get_smallest_opt(
                                         cell_add_max_ass(
                                             cell_add_max_ass(put_mandatory_ass_cell(Board), get_smallest_opt(Board)),
                                             get_most_empty_row(Board))))))))
        i += 1

    return Board, i


if __name__ == '__main__':
    # board = get_boards()[0]
    board = random.choice(get_boards())
    # print(board.assignment, board.eval_fitness_on_board(0.2, 0.2, 0.2, 0.2, 0.2))
    # assigned_board = board
    # for _ in range(10):
    #     board = get_boards()[0]
    #     board, i = calc_sol_while(board)
    #     # for row_i, row in enumerate(board.assignment):
    #     #     for row_j in range(len(row)):
    #     #         board.assignment[row_i][row_j] = random.choice(range(1, 10))
    #
    #     print(board.assignment, board.eval_fitness_on_board(0.2, 0.2, 0.2, 0.2, 0.2), i)
    #
    #     # for row_i, row in enumerate(board.assignment):
    #     #     for row_j in range(len(row)):
    #     #         board.assignment[row_i][row_j] = EMPTY_CELL

    assigned_board, i = calc_sol_while(board)
    # assigned_board.assignment = [[5,8,1],[8,6,9,4],[9,8],[3,1],[7,9,2,3],[9,8,6]]
    # assigned_board.assignment = [[5,8,1],[8,6,9,3],[9,8],[3,1],[7,9,2,3],[9,8,6]]
    # print(np.mean([calc_sol_while(b)[0].eval_fitness_on_board(0.2, 0.2, 0.2, 0.2, 0.2) for b in get_boards()]))
    print(assigned_board.assignment, assigned_board.eval_fitness_on_board(0.2, 0.2, 0.2, 0.2, 0.2), i)
    # cells = []
    # for cell in board.cells:
    #     cells.append(cell.opt_ass)
    # print(cells)
