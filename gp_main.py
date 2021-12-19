import copy
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import random
import seaborn as sns
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from deap import creator, base, tools
from deap import gp
from sklearn.model_selection import train_test_split
from typing import Callable

from board_transformations import *
from board_translator import get_boards

SEED = 42
TRAIN_SIZE = 0.3

pset = gp.PrimitiveSetTyped("main", [Board], Board)


# # ~~~~~~~~~~~~~~~~~First Experiment~~~~~~~~~~~~~~~~~
# pset.addPrimitive(put_mandatory_ass, [Board], Board)
# pset.addPrimitive(row_add_ass, [Board, Row], Board)
# pset.addPrimitive(col_add_ass, [Board, Col], Board)
# pset.addPrimitive(row_delete_ass, [Board, Row], Board)
# pset.addPrimitive(row_delete_dup, [Board, Row], Board)
# pset.addPrimitive(row_delete_noopt, [Board, Row], Board)
# pset.addPrimitive(col_delete_ass, [Board, Col], Board)
# pset.addPrimitive(col_delete_dup, [Board, Col], Board)
# pset.addPrimitive(col_delete_noopt, [Board, Col], Board)
# pset.addPrimitive(get_empty_cell_row, [Board], Row)
# pset.addPrimitive(get_has_dup_row, [Board], Row)
# pset.addPrimitive(get_invalid_sum_row, [Board], Row)
# pset.addPrimitive(get_empty_cell_col, [Board], Col)
# pset.addPrimitive(get_has_dup_col, [Board], Col)
# pset.addPrimitive(get_invalid_sum_col, [Board], Col)
# pset.addTerminal(INVALID_IDX, Row)
# pset.addTerminal(INVALID_IDX, Col)
# # ~~~~~~~~~~~~~~~~~First Experiment~~~~~~~~~~~~~~~~~


# # ~~~~~~~~~~~~~~~~~Third Experiment~~~~~~~~~~~~~~~~~
# assignment cell nodes
pset.addPrimitive(put_mandatory_ass_cell, [Board], Board)
pset.addPrimitive(cell_add_max_ass, [Board, Cell], Board)
pset.addPrimitive(cell_add_min_ass, [Board, Cell], Board)
# select cell nodes
pset.addPrimitive(get_largest_opt, [Board], Cell)
pset.addPrimitive(get_smallest_opt, [Board], Cell)
pset.addPrimitive(get_most_empty_row, [Board], Cell)
pset.addPrimitive(get_least_empty_row, [Board], Cell)
# backtrack cell node
pset.addPrimitive(backtrack_cells, [Board, int], Board)
pset.addPrimitive(fallback_no_opt, [Board], Board)
# terminals
pset.addTerminal(None, Cell)
for i in range(1, 11):
    pset.addTerminal(i, int)
pset.addPrimitive(id, [int], int)
# # ~~~~~~~~~~~~~~~~~Third Experiment~~~~~~~~~~~~~~~~~


pset.renameArguments(ARG0='Board')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)


def init_population(min_height, max_height):
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_height, max_=max_height)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def run_func_while_not_ok(func, board, while_cap):
    i = 0
    while not board_is_ok(board) and i < while_cap:
        board = func(board)
        i += 1

    return board, i


def eval_fitness_while_tree(tree, train_boards, while_cap,
                            cols_sum_weight, rows_sum_weight,
                            rows_dup_weight, cols_dup_weight,
                            unassignment_weight):
    tree_func = toolbox.compile(expr=tree)

    def eval_b(b: Board):
        b, i = run_func_while_not_ok(tree_func, b, while_cap)
        b_fit = b.eval_fitness_on_board(rows_sum_weight, rows_dup_weight,
                                        cols_sum_weight, cols_dup_weight,
                                        unassignment_weight)
        i_fit = i/while_cap
        return 0.9*b_fit + 0.1*i_fit

    boards = [copy.deepcopy(b) for b in train_boards]
    boards_fitness = toolbox.map(eval_b, boards)
    return np.mean(list(boards_fitness))


def init_evaluator(rows_sum_weight, rows_dup_weight, cols_sum_weight, cols_dup_weight, unassignment_weight,
                   train_boards, while_cap):
    toolbox.register("evaluate", eval_fitness_while_tree,
                     train_boards=train_boards, while_cap=while_cap,
                     rows_sum_weight=rows_sum_weight, rows_dup_weight=rows_dup_weight,
                     cols_sum_weight=cols_sum_weight, cols_dup_weight=cols_dup_weight,
                     unassignment_weight=unassignment_weight)


def init_selections(tour_size, pars_size=1.4):
    # toolbox.register("select", tools.selTournament, tournsize=tour_size)
    # tools.selDoubleTournament https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selDoubleTournament
    toolbox.register("select", tools.selDoubleTournament, fitness_size=tour_size, parsimony_size=pars_size,
                     fitness_first=True)


def init_crossovers():
    toolbox.register("mate", gp.cxOnePoint)


def init_mutation(min_height, max_height):
    toolbox.register("expr_mut", gp.genFull, min_=min_height, max_=max_height)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def init_bloat_control(height_limit):
    for alter_function in ["mate", "mutate"]:
        toolbox.decorate(alter_function, gp.staticLimit(key=operator.attrgetter("height"), max_value=height_limit))


def init_statistics():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("median", np.median)
    return stats


def init_GP(train_boards,
            while_cap=100,
            min_init_height=1, max_init_height=3,
            rows_sum_weight=0.2, rows_dup_weight=0.2, cols_sum_weight=0.2, cols_dup_weight=0.2, unassignment_weight=0.2,
            tour_size=3, pars_size=1.4,
            min_mutate_height=0, max_mutate_height=2,
            height_limit=17,
            ):
    init_population(min_init_height, max_init_height)
    init_evaluator(rows_sum_weight, rows_dup_weight, cols_sum_weight, cols_dup_weight, unassignment_weight,
                   train_boards, while_cap)
    init_selections(tour_size, pars_size)
    init_crossovers()
    init_mutation(min_mutate_height, max_mutate_height)
    init_bloat_control(height_limit)


def create_offsprings(parents, xvr_over_mut_pb, cross_pb, mutation_pb, dir_expr_path, run_num, to_dump=False):
    offsprings = [toolbox.clone(indi) for indi in parents]
    random.shuffle(offsprings)

    for i in range(0, len(offsprings), 2):
        if random.random() < xvr_over_mut_pb:
            # Cross-Over
            if random.random() < cross_pb:
                offsprings[i], offsprings[i + 1] = toolbox.mate(offsprings[i], offsprings[i + 1])
                del offsprings[i].fitness.values, offsprings[i + 1].fitness.values
        else:
            # Mutation
            for j in [i, i+1]:
                if random.random() < mutation_pb:
                    offsprings[j], = toolbox.mutate(offsprings[j])
                    del offsprings[j].fitness.values

    return offsprings


def dump_population_before_after(best_fitness_before, avg_fitness_before, best_fitness_after, avg_fitness_after,
                                 dir_expr_path, run_num, type):
    with open(f"{dir_expr_path}/run-{run_num}_{type}.txt", 'a') as file:
        mean_best_to_write = [
            f"best_before(min): {float(best_fitness_before[0]):.2f} ~~ mean_before: {avg_fitness_before:.2f}\t"
            f" best_after(min): {float(best_fitness_after[0]):.2f} ~~ mean_after: {avg_fitness_after:.2f}\n"]
        file.writelines(mean_best_to_write)


def evaluate_fitness(population):
    invalid_inds = [ind for ind in population if not ind.fitness.valid]
    finesses = toolbox.map(toolbox.evaluate, invalid_inds)
    for ind, fit in zip(invalid_inds, finesses):
        ind.fitness.values = (fit,)
    return invalid_inds


def run_GP(pop_size, gen_num=100, xvr_over_mut_pb=0.5, cross_pb=0.7, mutation_pb=0.3, verbose=False, dir_expr_path='.', run_num=0):
    def evaluate_population(population, gen_idx):
        # Evaluate the individuals with an invalid fitness
        invalid_inds = evaluate_fitness(population)

        # Record generation
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen_idx, nevals=len(invalid_inds), **record)
        if verbose:
            print(logbook.stream)

    stats = init_statistics()
    logbook = tools.Logbook()
    times = []
    last_time = datetime.now()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Initialize population
    pop = toolbox.population(n=pop_size)
    evaluate_population(pop, 0)
    best_fitness = -1
    # Start Generational Loop
    for gen in range(1, gen_num + 1):
        # Record Generation Time
        cur_time = datetime.now()
        print(gen, (cur_time - last_time).total_seconds())
        times.append((gen, (cur_time - last_time).total_seconds()))
        last_time = cur_time

        # Parental Selection
        parents = toolbox.select(pop, pop_size)
        to_dump = True
        offsprings = create_offsprings(parents, xvr_over_mut_pb, cross_pb, mutation_pb, dir_expr_path, run_num, to_dump)

        # Evaluate the individuals with an invalid fitness
        evaluate_population(offsprings, gen)

        # Replace population
        pop[:] = offsprings

        fitness_values = [ind.fitness.values for ind in pop]
        best_fitness = min(fitness_values)
        if gen % (gen_num // 10) == 0 or gen == gen_num:
            dump_population(gen, pop, best_fitness, fitness_values, dir_expr_path, run_num)

    return pop, logbook, times, best_fitness


def dump_population(gen, pop, best_fitness, fitness_values, dir_expr_path, run_num):
    with open(f"{dir_expr_path}/run-{run_num}_gen-{gen}.txt", 'a') as file:
        gen_to_write = [f"gen: {gen}\n", f"best_fitness(min): {float(best_fitness[0]):.2f}\n"]
        ind_to_write = [f"\tfitness-{float(fit[0]):.2f} ~~ individual: {str(ind)}\n" for ind, fit in
                        zip(pop, fitness_values)]
        file.writelines(gen_to_write + ind_to_write)


def generate_plot(logbook, dir_expr_path, run_num):
    maxFitnessValues, meanFitnessValues, minFitnessValues, medianFitnessValues, stdFitnessValues = logbook.select("max",
                                                                                                                  "avg",
                                                                                                                  "min",
                                                                                                                  "median",
                                                                                                                  "std")

    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red', label="Worst Fitness")
    plt.plot(meanFitnessValues, color='green', label="Mean Fitness")
    plt.plot(minFitnessValues, color='blue', label="Best Fitness")
    plt.plot(medianFitnessValues, color='orange', label="Median Fitness")
    plt.plot(stdFitnessValues, color='purple', label="Std Fitness")
    plt.xlabel('Generations')
    plt.ylabel('Fitness (Minimum problem)')
    plt.title('Fitness as a function of generations')
    plt.legend(loc='upper right')
    plt.savefig(f"{dir_expr_path}/Run-{run_num}.png")
    plt.close()


# Main
def evaluate_test(test_boards):
    num_of_ok = 0
    rows_complete = []
    cols_complete = []
    for board in test_boards:
        board, _ = calc_sol_while(board)
        if board_is_ok(board):
            num_of_ok += 1
        else:
            row_sat = 0
            for row_i, row in enumerate(board.assignment):
                if EMPTY_CELL in row:
                    break
                if sum(row) != board.rows_sum[row_i]:
                    break
                if row_has_dups(board, row_i):
                    break
                row_sat += 1

            row_sat /= len(board.rows_sum)
            rows_complete.append(row_sat)

            col_sat = 0
            for col_i in range(len(board.cols_size)):
                ass = board.get_col_ass(col_i)
                if EMPTY_CELL in ass:
                    break
                if sum(ass) != board.cols_sum[col_i]:
                    break
                if col_has_dups(board, col_i):
                    break
                col_sat += 1

            col_sat /= len(board.cols_sum)
            cols_complete.append(col_sat)

        return {
            'OK Boards': num_of_ok/len(test_boards),
            'Rows Completion (Among not OK) - Average': np.mean(rows_complete),
            'Rows Completion (Among not OK) - Std': np.std(rows_complete),
            'Columns Completion (Among not OK) - Average': np.mean(cols_complete),
            'Columns Completion (Among not OK) - Std': np.mean(cols_complete),
        }


if __name__ == '__main__':
    inp = int(sys.argv[1])
    expr_num = inp % 100
    run_num = (inp // 100)+1

    boards = get_boards()
    train_boards, test_boards = train_test_split(boards, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)
    # test_results = evaluate_test(test_boards)
    # for k, res in test_results.items():
    #     print(f'{k}: {res:.2f}')
    #
    # exit()
    exprs = [(pop_size, gen_num, xvr_over_mut_pb, mutation_pb, cross_pb, tour_size)
             for pop_size in [100]  # np.arange(500, 5001, 200)
             for gen_num in [50]
             for xvr_over_mut_pb in np.arange(0.4, 0.8, 0.2)
             for mutation_pb in np.arange(0.4, 0.8, 0.2)
             for cross_pb in np.arange(0.4, 0.8, 0.2)
             for tour_size in [5, 10]
             ]

    pop_size, gen_num, xvr_over_mut_pb, mutation_pb, cross_pb, tour_size = exprs[expr_num - 1]

    dir_expr_path = os.path.join('third', f'expr-{expr_num}')
    os.makedirs(dir_expr_path, exist_ok=True)
    init_GP(train_boards, tour_size=tour_size, height_limit=15, while_cap=50)
    import multiprocessing
    print(f'Running with pool of {multiprocessing.cpu_count()}')
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    best_fitnesses = []
    population, logbook, times, best_fitness = run_GP(pop_size=pop_size, gen_num=gen_num, verbose=True,
                                                      xvr_over_mut_pb=xvr_over_mut_pb,
                                                      mutation_pb=mutation_pb,
                                                      cross_pb=cross_pb, dir_expr_path=dir_expr_path,
                                                      run_num=run_num)
    best_fitnesses.append(best_fitness)
    generate_plot(logbook, dir_expr_path, run_num)
