import deap
from deap import gp
from deap.gp import PrimitiveSetTyped
import numpy
import pandas as pd
import seaborn as sns
from deap import creator, base, tools
from matplotlib import pyplot as plt
import json

class BoardConfiguration:
    pass


class Board:
    pass


def init_ind(board_conf):
    pass


def rows_assignment(borad, board_conf):
    pass


def column_assignment(board, board_conf):
    pass


pset = PrimitiveSetTyped("main", [BoardConfiguration], Board)
pset.addPrimitive(init_ind, [BoardConfiguration], Board)
pset.addPrimitive(rows_assignment, [Board,BoardConfiguration], Board)
pset.addPrimitive(column_assignment, [Board,BoardConfiguration], Board)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf(), pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)




