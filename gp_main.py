import operator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from deap import gp
from deap import creator, base, tools

from sklearn.model_selection import train_test_split

from board import Board
from board_transformations import init_ind, rows_assignment, column_assignment
from board_translator import get_boards

SEED = 42
TRAIN_SIZE = 0.7

MIN_MUTATE_HEIGHT = 0
MAX_MUTATE_HEIGHT = 2
TREE_HEIGHT_LIMIT = 17

pset = gp.PrimitiveSetTyped("main", [Board], Board)
pset.addPrimitive(init_ind, [Board], Board)
pset.addPrimitive(rows_assignment, [Board, Board], Board)
pset.addPrimitive(column_assignment, [Board, Board], Board)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveSetTyped, fitness=creator.FitnessMin,
               pset=pset)

stats = None
toolbox = base.Toolbox()
executor = ThreadPoolExecutor()
toolbox.register("map", executor.map)


def init_population(min_height, max_height):
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_height, max_=max_height)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


def init_evaluator(rows_weight, cols_weight, cols_dup_weight):
    # arithmetic sequence sum
    def dup_penalty(n, d=5):
        return (d * n * (n - 1)) / 2

    def eval_fitness_on_board(board: Board):
        rows_penalty = 0
        cols_penalty = 0
        cols_dup_penalty = 0
        # rows penalty
        for row_i, row_size in enumerate(board.rows_size):
            board_row_sum = sum(board.assignment[row_i])
            row_penalty = abs(board_row_sum - board.rows_sum[row_i])
            rows_penalty += row_penalty

        # cols penalty
        cols = [[board.assignment[row_i][cell_i] for row_i, cell_i in col_idx] for col_idx in board.cols_map]
        for col_i, col_vals in enumerate(cols):
            board_col_sum = sum(col_vals)
            col_penalty = abs(board_col_sum - board.cols_sum[col_i])
            cols_penalty += col_penalty

            num_occurrences = [0] * 9
            for col_val in col_vals:
                num_occurrences[col_val - 1] += 1
            col_dup_penalty = sum([dup_penalty(n) for n in num_occurrences])
            cols_dup_penalty += col_dup_penalty

        total_penalty = rows_weight * rows_penalty + cols_weight * cols_penalty + cols_dup_weight * cols_dup_penalty
        return total_penalty

    def eval_fitness_tree(tree):
        tree_func = toolbox.compile(expr=tree)
        boards_assigned = toolbox.map(tree_func, train_boards)
        boards_fitness = toolbox.map(eval_fitness_on_board, boards_assigned)
        return np.mean(boards_fitness),  # TODO: Normalize

    toolbox.register("evaluate", eval_fitness_tree)


def init_selections(tour_size):
    toolbox.register("select", tools.selTournament, tournsize=tour_size)


def init_crossovers():
    toolbox.register("mate", gp.cxOnePoint)


def init_mutation(min_height, max_height):
    toolbox.register("expr_mut", gp.genFull, min_=min_height, max_=max_height)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def init_bloat_control(height_limit):
    for alter_function in ["mate", "mutate"]:
        toolbox.decorate(alter_function, gp.staticLimit(key=operator.attrgetter("height"), max_value=height_limit))


def init_statistics():
    global stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("median", np.median)


def init_GP(min_init_height=1, max_init_height=3,
            rows_weight=0.33, cols_weight=0.33, cols_dup_weight=0.34,
            tour_size=3,
            min_mutate_height=0, max_mutate_height=2,
            height_limit=17
            ):
    init_population(min_init_height, max_init_height)
    init_evaluator(rows_weight, cols_weight, cols_dup_weight)
    init_selections(tour_size)
    init_crossovers()
    init_mutation(min_mutate_height, max_mutate_height)
    init_bloat_control(height_limit)
    init_statistics()


if __name__ == '__main__':
    boards = get_boards()
    train_boards, test_boards = train_test_split(boards, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)
