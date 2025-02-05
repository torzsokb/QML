import form2
import pandas as pd
import sys

data = pd.read_csv("assignment_files/assignment_2/Residential_areas_2.csv", header=None, names=["x", "y"])
    
coords = {}
for index, row in data.iterrows():
    coords[index] = (row["x"], row["y"])

solver = form2.tsp_solver_1(coordinates=coords)

solver.solve_tsp(tight=True)