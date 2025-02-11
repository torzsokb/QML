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

def plot_solution(selected, coordinates, title=None, starting_node=None, prev_selected=None, fuel_locations=None):
    plt.figure(figsize=(8, 8))

    for arc in selected:
        x_from, y_from = coordinates[arc[0]]
        x_to, y_to = coordinates[arc[1]]
        plt.plot(x_from, y_from, 'bo')
        arrow = FancyArrowPatch((x_from, y_from), (x_to, y_to), arrowstyle='->', color='blue', linewidth=1, mutation_scale=10, alpha=0.5)
        plt.gca().add_patch(arrow)

    if prev_selected is not None:
        reversed = not any(i in prev_selected for i in selected)
        for arc in prev_selected:
            node_from = arc[0]
            node_to = arc[1]
            if (node_from, node_to) not in selected and (node_to, node_from) not in selected:
                x_from, y_from = coordinates[node_from]
                x_to, y_to = coordinates[node_to]
                if reversed:
                    arrow = FancyArrowPatch((x_to, y_to), (x_from, y_from), arrowstyle='->', color='red', linewidth=1, mutation_scale=10, alpha=0.5)
                else:
                    arrow = FancyArrowPatch((x_from, y_from), (x_to, y_to), arrowstyle='->', color='red', linewidth=1, mutation_scale=10, alpha=0.5)
                plt.gca().add_patch(arrow)
            



    if starting_node is not None:
        x, y = coordinates[starting_node]
        plt.plot(x, y, 'ro', label="starting node")

    if fuel_locations is not None:
        for fuel_location in fuel_locations:
            x, y = coordinates[fuel_location]
            plt.plot(x, y, 'ro')

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

def load_data(path_locations, path_fuel):

    locations_df = pd.read_csv(path_locations, header=None, names=['x', 'y'])
    fuel_coordinates = list(pd.read_csv(path_fuel, header=None).itertuples(index=None, name=None))

    coordinates = {}
    locations = []
    fuel_locations = []

    for location, row in locations_df.iterrows():
        coordinate = (row['x'], row['y'])
        coordinates[location] = coordinate
        locations.append(location)
        if coordinate in fuel_coordinates:
            fuel_locations.append(location)

    arc_lengths = {(i, j): manhattan_dist(coordinates[i], coordinates[j]) for i,j in permutations(locations, 2)}
    edge_lengths = {(i, j): manhattan_dist(coordinates[i], coordinates[j]) for i,j in combinations(coordinates.keys(), 2)}

    return len(locations), locations, fuel_locations, coordinates, arc_lengths, edge_lengths

def find_subtours(edges):

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

def nearest_neighbor_heuristic(arc_lengths, locations, starting_node):

    #assert starting_node in locations, "invalid starting node"

    tour = [starting_node]
    tour_tuples = []
    tour_length = 0

    while len(tour) < len(locations): 
            i = tour[-1]
            
            min_distance = min([arc_lengths[i, j] for j in locations if j not in tour and j != i])
            nearest = [j for j in locations if j not in tour and arc_lengths[i,j] == min_distance]
            j = nearest[0]

            tour.append(j)
            tour_tuples.append((i,j))
            tour_length += min_distance

    tour_tuples.append((tour[-1], tour[0]))
    tour_length += arc_lengths[tour_tuples[-1]]

    return tour_length, tour, tour_tuples

def get_best_heuristic(arc_lengths, locations):

    best_obj_value, best_tour, best_tour_tuples = nearest_neighbor_heuristic(arc_lengths, locations, 0)

    for locatation in locations[1:]:
        obj_value, tour, tour_tuples = nearest_neighbor_heuristic(arc_lengths, locations, starting_node=locatation)
        if obj_value < best_obj_value:
            best_obj_value = obj_value
            best_tour = tour
            best_tour_tuples = tour_tuples

    return best_obj_value, best_tour, best_tour_tuples

class tsp_callback:
    
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
        subtours = find_subtours(edges)
        
        for subtour in subtours:
            if 2 <= len(subtour) and len(subtour) <= len(self.nodes) / 2:
                outside = [i for i in self.nodes if i not in subtour]
                model.cbLazy(
                    gp.quicksum(self.x[i, j] for i, j in product(subtour, outside))
                    >= 2)
                
def solve_tsp_edge(locations, edge_lengths, callback_allowed=True, time_limit=15*60, forbidden_solution=None):

    with gp.Env() as env, gp.Model(env=env) as m:

        # Set time limit as specified in the assignment
        m.Params.TimeLimit = time_limit
        
        # Create variables, and add symmetric keys to the resulting dictionary
        # 'x', such that (i, j) and (j, i) refer to the same variable.
        x = m.addVars(edge_lengths.keys(), obj=edge_lengths, vtype=GRB.BINARY, name="x")
        x.update({(j, i): v for (i, j), v in x.items()})

        # Create degree 2 constraints
        m.addConstrs(x.sum(i, '*') == 2 for i in locations)

        if forbidden_solution is not None:
            m.addConstr(gp.quicksum(x[i,j] for i,j in forbidden_solution) <= len(locations)-1)


        if callback_allowed:
            m.Params.LazyConstraints = 1
            cb = tsp_callback(locations, x)
            m.optimize(cb)    
        else:
            try:
                for subtour_size in range(2,  math.ceil(len(locations) / 2)):
                    for subtour in combinations(locations, subtour_size):
                        outside = [i for i in locations if i not in subtour]
                        m.addConstr(
                            gp.quicksum(x[i, j] for i, j in product(subtour, outside))
                            >= 2)
                m.optimize()
            except Exception:
                logging.exception("ERROR during model implementing all constraint")

        

        vals = m.getAttr('x', x)
        # transforming the outcome so that we can plot it with directions
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tour = find_subtours(selected)[0]
        selected = []
        for index in range(1, len(tour)):
            selected.append((tour[index-1], tour[index]))
        selected.append((tour[-1], tour[0]))
        runtime = m.Runtime
        objective = m.ObjVal

        return runtime, objective, selected
        
def solve_tsp_arc(locations, arc_lengths, n, tight=False, starting_node=None, fuel_locations=None, max_range=None, time_limit=15*60, forbidden_solution=None):

    with gp.Env() as env, gp.Model(env=env) as m:

        # Set time limit as specified in the assignment
        m.Params.TimeLimit = time_limit
        m.Params.LogToConsole = 0

        # Create decision variables
        first = locations[0]
        z = m.addVars(arc_lengths.keys(), obj=arc_lengths, vtype=GRB.BINARY, name="z")
        u = m.addVars(locations[1:], vtype=GRB.INTEGER, lb=1, ub=n-1, name="u")

        # Add constraint to make sure each node is visited exactly once
        m.addConstrs(z.sum(i, '*') == 1 for i in locations)
        m.addConstrs(z.sum('*', i) == 1 for i in locations)

        # Add tour order constraints (tight refers to the constraints in exercise 7)
        if not tight:
            m.addConstrs(u[i] - u[j] + (n - 1) * z[i,j] <= n - 2 for i,j in z.keys() if i > 0 and j > 0)
        else:
            m.addConstrs(u[i] - u[j] + (n - 1) * z[i,j] + (n - 3) * z[j,i] <= n - 2 for i,j in z.keys() if i > 0 and j > 0)
            m.addConstrs(u[i] >= 1 + (n - 3) * z[i,first] + z.sum('*', i) - z[first,i] for i in u.keys())
            m.addConstrs(u[i] <= n - 1 - (n - 3) * z[first,i] + z.sum(i, '*') - z[i,first] for i in u.keys())

        # Implementing warm start
        if starting_node is not None:
            
            obj, tour, tour_tuples = nearest_neighbor_heuristic(arc_lengths=arc_lengths, locations=locations, starting_node=starting_node)
            while tour.index(0) != 0:
                tour.insert(0, tour.pop())

            for i,j in tour_tuples:
                z[i,j].Start = 1
            for i in range(1, len(tour)):
                u[tour[i]].Start = i

            m.update()

        # Implementing range restrictions
        if fuel_locations is not None and max_range is not None:
            r = m.addVars(locations, vtype=GRB.CONTINUOUS, lb=0, ub=max_range, name='r')
            m.addConstrs(r[j] <= r[i] - arc_lengths[i,j] * z[i,j] + max_range * (1 - z[i, j]) for i,j in z.keys() if i not in fuel_locations)
            m.addConstrs(r[j] <= max_range - arc_lengths[i,j] * z[i,j] + max_range * (1 - z[i, j]) for i,j in z.keys() if i in fuel_locations)
            m.update()

        if forbidden_solution is not None:
            m.addConstr(gp.quicksum(z[i,j] for i,j in forbidden_solution) <= n-1)
            m.addConstr(gp.quicksum(z[j,i] for i,j in forbidden_solution) <= n-1)
            m.update()


        m.optimize()

        vals = m.getAttr('x', z)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        runtime = m.Runtime
        objective = m.ObjVal

        return runtime, objective, selected







def main():

    path_locations = "assignment_files/assignment_2/Residential_areas_2.csv"
    path_fuel = "assignment_files/assignment_2/Fuel_locations.csv"

    n, locations, fuel_locations, coordinates, arc_lengths, edge_lengths = load_data(path_locations, path_fuel)

    # Exercise 1 is done in the LateX document

    # Exercise 2

    #_, _, _ = solve_tsp_edge(locations, edge_lengths, callback_allowed=False)

    runtime_2a, objective_2a, selected_2a = solve_tsp_arc(locations, arc_lengths, n)
    print(f"tsp arc formulation with standard constraints and no warm start: \nobjective value: {objective_2a} \nrunning time: {runtime_2a}")
    plot_solution(selected=selected_2a, coordinates=coordinates, title="TSP Solution Arc Formulation")

    # Exercise 3 is done in the LateX document

    # Exercise 4 is done in the LateX document

    # Exercise 5
    runtime_5e, objective_5e, selected_5e = solve_tsp_edge(locations, edge_lengths, callback_allowed=True)
    print(f"tsp edge formulation with callbacks: \nobjective value: {objective_5e}, running time: {runtime_5e}")
    plot_solution(selected=selected_5e, coordinates=coordinates, title="TSP Solution Edge Formulation")

    # Exercise 6
    objective, _, tour_tuples = nearest_neighbor_heuristic(arc_lengths, locations, starting_node=0)
    print(f"tsp nearest neighbor heuristic \nstarting node: 0 \n objective value: {objective}")
    plot_solution(selected=tour_tuples, coordinates=coordinates, starting_node=0, title=f"TSP Heuristic Solution from node: 0 \n objective value: {objective}")

    objective, best_tour, tour_tuples = get_best_heuristic(arc_lengths, locations)
    best_starting_node = best_tour[0]
    print(f"tsp nearest neighbor heuristic \nstarting node: {best_starting_node} \n objective value: {objective}")
    plot_solution(selected=tour_tuples, coordinates=coordinates, starting_node=best_starting_node, title=f"TSP Heuristic Solution from node: {best_starting_node} \n objective value: {objective}")

    # Exercise 7
    warm_starts = [None, 0, best_starting_node]
    print("running time for tsp arc formulation with standard constraints")
    for start in warm_starts:
        runtime, _, _ = solve_tsp_arc(locations, arc_lengths, n, starting_node=start)
        print(f"heuristic warm start from {start}, runtime: {runtime}")

    print("running time for tsp arc formulation with strict constraints")
    for start in warm_starts:
        runtime, _, _ = solve_tsp_arc(locations, arc_lengths, n, starting_node=start, tight=True)
        print(f"heuristic warm start from {start}, runtime: {runtime}")

    # Exercise 8
    runtime_8a, objective_8a, selected_8a = solve_tsp_arc(locations, arc_lengths, n, forbidden_solution=selected_2a)
    print(f"tsp arc formulation with strict constraints alternative solution \nobjective value: {objective_8a} \nrunning time: {runtime_8a}")
    plot_solution(selected=selected_8a, coordinates=coordinates, title="TSP Arc Formulation Alternative Solution", prev_selected=selected_2a)

    runtime_8e, objective_8e, selected_8e = solve_tsp_edge(locations, edge_lengths, forbidden_solution=selected_5e)
    print(f"tsp edge formulation alternative solution: \nobjective value: {objective_8e} \nrunning time: {runtime_8e}")
    plot_solution(selected=selected_8e, coordinates=coordinates, title="TSP Edge Formulation Alternative Solution", prev_selected=selected_5e)


    # Exercise 9 done in the LateX document

    # Bonus Exercise
    runtime_b, objective_b, selected_b = solve_tsp_arc(locations, arc_lengths, n, fuel_locations=fuel_locations, tight=True)
    print(f"tsp arc formulation with strict constraints and fuel locations \nobjective value: {objective_b} \nrunning time: {runtime_b}")
    plot_solution(selected=selected_b, coordinates=coordinates, title=f"TSP Solution Arc Formulation with Fuel Locations \nobjective value: {objective_b}", fuel_locations=fuel_locations)



if __name__ == "__main__":
    main()

