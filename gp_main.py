import operator
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import random
from datetime import datetime

import numpy as np
from deap import gp
from deap import creator, base, tools

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from board_transformations import *
from board_translator import get_boards

SEED = 42
TRAIN_SIZE = 0.7

pset = gp.PrimitiveSetTyped("main", [Board], Board)
pset.addPrimitive(row_add_ass, [Board, Row], Board)
pset.addPrimitive(put_mandatory_ass, [Board], Board)
pset.addPrimitive(col_trans, [Board, Col], Board)
pset.addPrimitive(get_invalid_row, [Board], Row)
pset.addPrimitive(get_invalid_col, [Board], Col)
# pset.addPrimitive(get_invalid_col, [Board], bool)

pset.addTerminal(INVALID_IDX, Row)
pset.addTerminal(INVALID_IDX, Col)
pset.renameArguments(ARG0='Board')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)

stats = None
toolbox = base.Toolbox()
# executor = ThreadPoolExecutor()
# toolbox.register("map", executor.map)


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
        boards_fitness = list(toolbox.map(eval_fitness_on_board, boards_assigned))
        return np.mean(boards_fitness)  # TODO: Normalize

    toolbox.register("evaluate", eval_fitness_tree)


def init_selections(tour_size):
    toolbox.register("select", tools.selTournament, tournsize=tour_size)
    # tools.selDoubleTournament https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selDoubleTournament


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


def create_offsprings(parents, cross_pb, mutation_pb, dir_expr_path, run_num, to_dump=False):
    offsprings = [toolbox.clone(indi) for indi in parents]
    random.shuffle(offsprings)

    # Cross-Over
    for i in range(0, len(offsprings), 2):
        if random.random() < cross_pb:
            offsprings[i], offsprings[i + 1] = toolbox.mate(offsprings[i], offsprings[i + 1])
            del offsprings[i].fitness.values, offsprings[i + 1].fitness.values

    if to_dump:
        evaluate_fitness(offsprings)
        fitness_offsprings_after_cross = [ind.fitness.values for ind in offsprings]
        fitness_values_parents = [ind.fitness.values for ind in parents]
        best_fitness_offsprings_after_cross = min(fitness_offsprings_after_cross)
        best_fitness_parents = min(fitness_values_parents)
        avg_fitness_offsprings_after_cross = np.mean(fitness_offsprings_after_cross)
        avg_fitness_parents = np.mean(fitness_values_parents)
        dump_population_before_after(best_fitness_parents, avg_fitness_parents, best_fitness_offsprings_after_cross,
                                     avg_fitness_offsprings_after_cross, dir_expr_path, run_num, 'cross-over')

    # Mutation
    for i in range(len(offsprings)):
        if random.random() < mutation_pb:
            offsprings[i], = toolbox.mutate(offsprings[i])
            del offsprings[i].fitness.values

    if to_dump:
        evaluate_fitness(offsprings)
        fitness_offsprings_after_mutation = [ind.fitness.values for ind in offsprings]
        best_fitness_offsprings_after_mutation = min(fitness_offsprings_after_mutation)
        avg_fitness_offsprings_after_mutation = np.mean(fitness_offsprings_after_mutation)
        dump_population_before_after(best_fitness_offsprings_after_cross, avg_fitness_offsprings_after_cross,
                                     best_fitness_offsprings_after_mutation, avg_fitness_offsprings_after_mutation,
                                     dir_expr_path, run_num, 'mutation')

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


def run_GA(pop_size, gen_num=100, cross_pb=0.7, mutation_pb=0.3, verbose=False, dir_expr_path='.', run_num=0):
    def evaluate_population(population, gen_idx):
        # Evaluate the individuals with an invalid fitness
        invalid_inds = evaluate_fitness(population)

        # Record generation
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen_idx, nevals=len(invalid_inds), **record)
        if verbose:
            print(logbook.stream)

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
        times.append((gen, (cur_time - last_time).total_seconds()))
        last_time = cur_time

        # Parental Selection
        parents = toolbox.select(pop, pop_size)
        to_dump = True
        offsprings = create_offsprings(parents, cross_pb, mutation_pb, dir_expr_path, run_num, to_dump)

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
if __name__ == '__main__':
    inp = int(sys.argv[1])
    expr_num = inp % 1000
    run_num = inp // 1000

    boards = get_boards()
    train_boards, test_boards = train_test_split(boards, train_size=TRAIN_SIZE, shuffle=True, random_state=SEED)

    exprs = [(pop_size, gen_num, mutation_pb, cross_pb, tour_size)
             for pop_size in np.arange(500, 5001, 200)
             for gen_num in [100]
             for mutation_pb in np.arange(0.3, 0.8, 0.2)
             for cross_pb in np.arange(0.3, 0.8, 0.2)
             for tour_size in [5, 12, 15]
             ]

    pop_size, gen_num, mutation_pb, cross_pb, tour_size = exprs[expr_num]

    dir_expr_path = os.path.join('exprs', f'expr-{expr_num}')
    os.makedirs(dir_expr_path, exist_ok=True)
    init_GP(tour_size=tour_size)

    best_fitnesses = []
    population, logbook, times, best_fitness = run_GA(pop_size=pop_size, gen_num=gen_num, verbose=True,
                                                      mutation_pb=mutation_pb,
                                                      cross_pb=cross_pb, dir_expr_path=dir_expr_path,
                                                      run_num=run_num)
    best_fitnesses.append(best_fitness)
    generate_plot(logbook, dir_expr_path, run_num)
