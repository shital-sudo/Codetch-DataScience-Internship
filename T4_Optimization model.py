# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 17:27:45 2026

@author: hp
"""

# Import PuLP
import pulp

# 1. Define the Problem
model = pulp.LpProblem("Product_Mix_Optimization", pulp.LpMaximize)

# 2. Define Decision Variables
# x = units of Product A, y = units of Product B
x = pulp.LpVariable('Product_A', lowBound=0, cat='Continuous')
y = pulp.LpVariable('Product_B', lowBound=0, cat='Continuous')

# 3. Define Objective Function (maximize profit)
model += 40*x + 50*y, "Total_Profit"

# 4. Define Constraints
model += 2*x + 1*y <= 100, "Machine_Hours_Constraint"
model += 1*x + 2*y <= 80, "Labor_Hours_Constraint"

# 5. Solve the Problem
model.solve()

# 6. Output Results
print("Status:", pulp.LpStatus[model.status])
print("Optimal units of Product A:", x.varValue)
print("Optimal units of Product B:", y.varValue)
print("Maximum Profit:", pulp.value(model.objective))