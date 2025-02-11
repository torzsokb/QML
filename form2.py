from collections import defaultdict
import logging
import math
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

    def __init__(self, coordinates, max_range=None, fuel_locations=None):
        
        self.coordinates = coordinates
        self.locations = list(coordinates.keys())
        self.n = len(self.locations)
        self.arc_lengths = {(i, j): manhattan_dist(coordinates[i], coordinates[j]) for i,j in permutations(coordinates.keys(), 2)}

        self.max_range=max_range
        self.fuel_locations = fuel_locations

        if max_range is not None:
            assert fuel_locations is not None, "can't use max range without fuel locations"
        if fuel_locations is not None:
            assert max_range is not None, "can't use fuel locations without max range"
            assert all(fuel_location in coordinates.keys() for fuel_location in fuel_locations), "fuel locations not consistent with nodes"

    def nearest_neighbor_heuristic(self, starting_node):

        assert starting_node in self.coordinates.keys(), "invalid starting node"
        
        tour = [starting_node]
        tour_tuples = []
        tour_length = 0

        while len(tour) < self.n: 
            i = tour[-1]
            
            min_distance = min([self.arc_lengths[i, j] for j in self.locations if j not in tour and j != i])
            nearest = [j for j in self.locations if j not in tour and self.arc_lengths[i,j] == min_distance]
            j = nearest[0]

            tour.append(j)
            tour_tuples.append((i,j))
            tour_length += min_distance

        tour_tuples.append((tour[-1], tour[0]))
        tour_length += self.arc_lengths[tour_tuples[-1]]

        return tour_length, tour, tour_tuples
    
    
    
    def solve1(self, tight=False, starting_node=None, refuel=False):
         
        model = gp.Model()
        
        first = self.locations[0]
        
        x = model.addVars(self.arc_lengths.keys(), obj=self.arc_lengths, vtype=GRB.BINARY, name='x')
        u = model.addVars(self.locations[1:], vtype=GRB.INTEGER, lb=1, ub=self.n-1, name='u')
        
        model.addConstrs(x.sum(i, '*') == 1 for i in self.locations)
        model.addConstrs(x.sum('*', i) == 1 for i in self.locations)


        if not tight:
            model.addConstrs(u[i] - u[j] + (self.n - 1) * x[i,j]
                <= self.n - 2 for i,j in x.keys() if i > 0 and j > 0)
        else:
            model.addConstrs(u[i] - u[j] + (self.n - 1) * x[i,j] + (self.n - 3) * x[j,i] <= self.n - 2 for i,j in x.keys() if i > 0 and j > 0)
            model.addConstrs(u[i] >= 1 + (self.n - 3) * x[i,first] + x.sum('*', i) - x[first,i] for i in u.keys())
            model.addConstrs(u[i] <= self.n - 1 - (self.n - 3) * x[first,i] + x.sum(i, '*') - x[i,first] for i in u.keys())


        if starting_node != None:
            length, tour, tour_tuples = self.nearest_neighbor_heuristic(starting_node=starting_node)

            for i,j in tour_tuples:
                x[i,j].Start = 1

            if starting_node != 0:
                index = tour.index(0)
                for _ in range(self.n - index):
                    tour.insert(0, tour.pop())
            
            for i in range(1, self.arc_lengthsn):
                u[i].Start = tour[i]

            model.update()

        if refuel:
            assert self.fuel_locations is not None and self.max_range is not None, "can't solve problem without refueling wihtout required parameters"
            r = model.addVars(self.locations, vtype=GRB.CONTINUOUS, lb=0, ub=self.max_range, name='r')
            model.addConstrs(r[j] <= r[i] - self.arc_lengths[i,j] * x[i,j] + self.max_range * (1 - x[i, j]) for i,j in x.keys() if i not in self.fuel_locations)
            model.addConstrs(r[j] <= self.max_range - self.arc_lengths[i,j] * x[i,j] + self.max_range * (1 - x[i, j]) for i,j in x.keys() if i in self.fuel_locations)
            model.update()

        
        model.optimize()

        vals = model.getAttr('x', x)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        runtime = model.Runtime
        objective = model.ObjVal

        model.dispose()

        return runtime, objective, selected
        
def shortest_subtour(edges):
    
    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)
    assert all(len(neighbors) == 2 for neighbors in node_neighbors.values())

    unvisited = set(node_neighbors)
    shortest = None
    while unvisited:
        cycle = []
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors.pop()
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for j in node_neighbors[current] if j in unvisited]
        if shortest is None or len(cycle) < len(shortest):
            shortest = cycle

    assert shortest is not None
    return shortest
    
    


def all_subtours(edges):

    node_neighbors = defaultdict(list)
    for i, j in edges:
        node_neighbors[i].append(j)
    assert all(len(neighbors) == 2 for neighbors in node_neighbors.values())

    unvisited = set(node_neighbors)
    subtours = []
    while unvisited:
        cycle = []
        neighbors = list(unvisited)
        while neighbors:
            current = neighbors.pop()
            cycle.append(current)
            unvisited.remove(current)
            neighbors = [j for j in node_neighbors[current] if j in unvisited]
        subtours.append(cycle)

    return subtours
       

class TSPCallback_1:
    
    def __init__(self, nodes, x):
        self.nodes = nodes
        self.x = x

    def __call__(self, model, where):
        
        if where == GRB.Callback.MIPSOL:
            try:
                self.eliminate_subtours(model)
            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                model.terminate()

    def eliminate_subtours(self, model):
        
        values = model.cbGetSolution(self.x)
        edges = [(i, j) for (i, j), v in values.items() if v > 0.5]
        subtours = all_subtours(edges)
        
        for subtour in subtours:
            if 2 <= len(subtour) and len(subtour) <= len(self.nodes) / 2:
                outside = [i for i in self.nodes if i not in subtour]
                model.cbLazy(
                    gp.quicksum(self.x[i, j] for i, j in product(subtour, outside))
                    >= 2)


        
def solve_tsp_edge(locations, edge_distance, original=True, callback_allowed=True):


    with gp.Env() as env, gp.Model(env=env) as m:
        
        # Create variables, and add symmetric keys to the resulting dictionary
        # 'x', such that (i, j) and (j, i) refer to the same variable.
        x = m.addVars(edge_distance.keys(), obj=edge_distance, vtype=GRB.BINARY, name="x")
        x.update({(j, i): v for (i, j), v in x.items()})

        # Create degree 2 constraints
        m.addConstrs(x.sum(i, '*') == 2 for i in locations)

        if callback_allowed:
            m.Params.LazyConstraints = 1
            cb = TSPCallback_1(locations, x)
            m.optimize(cb)    
        else:
            try:
                for subtour_size in range(2,  math.floor(len(locations) / 2 + 1)):
                    for subtour in combinations(locations, subtour_size):
                        outside = [i for i in locations if i not in subtour]
                        m.addConstr(
                            gp.quicksum(x[i, j] for i, j in product(subtour, outside))
                            >= 2)
                m.optimize()
            except Exception:
                logging.exception("ERROR during from model implementing all constraint")


        
        


if __name__ == "__main__":
    print("hello world main")
    
    data = pd.read_csv("assignment_files/assignment_2/Residential_areas_2.csv", header=None, names=["x", "y"])
    
    coords = {}
    locations =[]
    for index, row in data.iterrows():
        coords[index] = (row["x"], row["y"])
        locations.append(index)

    edge_distance = {(i, j): manhattan_dist(coords[i], coords[j]) for i,j in combinations(locations, 2)}

    solve_tsp_edge(locations, edge_distance, original=True)

    tsp = tsp_solver_1(coords)
    tsp.solve1(tight=True)
    for i in range(0):
        print("huhh")

    

    


     



         

    