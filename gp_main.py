from deap import gp
from deap import creator, base, tools

from board import Board
from board_transformations import init_ind, rows_assignment, column_assignment
from board_translator import get_boards

pset = gp.PrimitiveSetTyped("main", [Board], Board)
pset.addPrimitive(init_ind, [Board], Board)
pset.addPrimitive(rows_assignment, [Board, Board], Board)
pset.addPrimitive(column_assignment, [Board, Board], Board)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveSetTyped, fitness=creator.FitnessMin,
               pset=pset)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)

boards = get_boards()


def eval_kakuro_tree(tree):
    tree_func = toolbox.compile(expr=tree)
