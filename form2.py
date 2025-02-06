import pandas as pd
import gurobipy as gp
import gurobipy_pandas as gppd
import numpy as np
from itertools import product, combinations, permutations
from gurobipy import GRB
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import sys

def manhattan_dist(loc1, loc2):
    dx = abs(loc1[0] - loc2[0])
    dy = abs(loc1[1] - loc2[1])
    return dx + dy

def plot_solution(selected, coordinates, title=None, starting_node=None):
    plt.figure(figsize=(8, 8))

    for arc in selected:
        x_from, y_from = coordinates[arc[0]]
        x_to, y_to = coordinates[arc[1]]
        plt.plot(x_from, y_from, 'bo')
        arrow = FancyArrowPatch((x_from, y_from), (x_to, y_to), arrowstyle='->', color='blue', linewidth=1, mutation_scale=10, alpha=0.5)
        plt.gca().add_patch(arrow)

    if starting_node != None:
        x, y = coordinates[starting_node]
        plt.plot(x, y, 'ro', label="starting node")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.suptitle('Coordinate Map')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.axis('equal')
    plt.show()

class tsp_solver_1:

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.arc_lengths = {(i, j): manhattan_dist(coordinates[i], coordinates[j]) for i,j in permutations(coordinates.keys(), 2)}

    def nearest_neighbor(self, starting_node):
        
        tour = [starting_node]
        tour_tuples = []
        tour_length = 0

        while len(tour) < len(self.coordinates.keys()): 
            i = tour[-1]
            
            min_distance = min([self.arc_lengths[i, j] for j in self.coordinates.keys() if j not in tour and j != i])
            nearest = [j for j in self.coordinates.keys() if j not in tour and self.arc_lengths[i,j] == min_distance]
            j = nearest[0]

            tour.append(j)
            tour_tuples.append((i,j))
            tour_length += min_distance

        tour_tuples.append((tour[-1], tour[0]))
        tour_length += self.arc_lengths[tour_tuples[-1]]

        return tour_length, tour, tour_tuples
    
    def solve_tsp(self, tight=False, starting_node=None, plot=False):
         
        model = gp.Model()

        locations = []
        for i in self.coordinates.keys():
            locations.append(int(i))

        n = len(locations)
        first = locations[0]
        
        x = model.addVars(self.arc_lengths.keys(), obj=self.arc_lengths, vtype=GRB.BINARY, name='x')
        u = model.addVars(locations[1:], vtype=GRB.INTEGER, lb=1, ub=n-1, name='u')
        
        cons_out = model.addConstrs(x.sum(i, '*') == 1 for i in locations)
        cons_in = model.addConstrs(x.sum('*', i) == 1 for i in locations)

        title = "TSP solution using formulation (2) \n with "

        if not tight:
            cons_order = model.addConstrs(u[i] - u[j] + (n - 1) * x[i,j] <= n - 2 for i,j in x.keys() if i > 0 and j > 0)
            model.update()
            title += "standard constraints "
        else:
            cons_order = model.addConstrs(u[i] - u[j] + (n - 1) * x[i,j] + (n - 3) * x[j,i] <= n - 2 for i,j in x.keys() if i > 0 and j > 0)
            cons_extra_1 = model.addConstrs(u[i] >= 1 + (n - 3) * x[i,first] + x.sum('*', i) - x[first,i] for i in u.keys())
            cons_extra_2 = model.addConstrs(u[i] <= n - 1 - (n - 3) * x[first,i] + x.sum(i, '*') - x[i,first] for i in u.keys())
            model.update()
            title += "tighter constraints for u(i) "

        if starting_node != None:
            assert starting_node in self.coordinates.keys()
            length, tour, tour_tuples = self.nearest_neighbor(starting_node=starting_node)

            for i,j in tour_tuples:
                x[i,j].Start = 1

            if starting_node != 0:
                index = tour.index(0)
                for _ in range(n - index):
                    tour.insert(0, tour.pop())
            
            for i in range(1, n):
                u[i].Start = tour[i]

            model.update()
            title += f"and heuristic warm start from {starting_node}, with initial obj val: {length}, "
        else:
            title +="and no warm start, "

        model.update()
        model.optimize()

        title += f"\n runtime: {model.Runtime}, obj val: {model.ObjVal}"
        vals = model.getAttr('x', x)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        if plot:
            plot_solution(selected=selected, coordinates=self.coordinates, title=title, starting_node=starting_node)

        model.dispose()
        
         


if __name__ == "__main__":
    print("hello world main")
    
    data = pd.read_csv("assignment_files/assignment_2/Residential_areas_2.csv", header=None, names=["x", "y"])
    
    coords = {}
    for index, row in data.iterrows():
        coords[index] = (row["x"], row["y"])

    
    solver = tsp_solver_1(coordinates=coords)

    
    solver.solve_tsp(tight=True, starting_node=29, plot=True)
    

    # for i in range(3):
    #     l, tour, tour_tuples = solver.nearest_neighbor(i)
    #     print(l)
    #     print(tour)
    #     index = tour.index(0)
    #     print(index)
    #     for _ in range(len(coords.keys()) - index):
    #         tour.insert(0, tour.pop())
    #     print(tour)
    #     title = f"Nearest Neighbor Heuristic, starting point: {i}, objective: {l}"
    #     plot_solution(coordinates=coords, selected=tour_tuples, starting_node=i, title=title)

    


     



         

    