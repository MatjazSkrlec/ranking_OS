'''
Author: Matjaz Skrlec
date: 20. 08. 2025
version: 1.0
description: This code is created for testing the code generated
'''

# sys.path.append("C:\\Program Files\\DIgSILENT\\PowerFactory 2024 Preview\\Python\\3.12")
# import powerfactory as pf

#--------------------------------------------------------------------------------------------------
## imports

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import FindPoints

#--------------------------------------------------------------------------------------------------
## functions

#--------------------------------------------------------------------------------------------------
## files used in code
#--------------------------------------------------------------------------------------------------
## variable parameters of code
#-------------------------------------------------------------------------------------------------
## code

points = np.random.uniform(-1, 1, size=(10, 2))

print(type(points))
print(points)

DataDistances, DataIndices, MaxDistPoint = FindPoints.GetMostIsolatedPoint(points)
print(MaxDistPoint)

Df = FindPoints.CreatePLDfFromNpArray(points)
print(Df)

x = points[:, 0]
y = points[:, 1]

df_no_rowlabel = Df.select(pl.exclude("RowLabel"))

DataDistances, DataIndices, MaxDistPoint = FindPoints.GetMostIsolatedPoint(df_no_rowlabel)
print(MaxDistPoint)

# Scatter plot
plt.scatter(x, y, color="blue", s=80, marker="o")

# Add labels for each point
for i, (px, py) in enumerate(points):
    plt.text(px + 0.02, py + 0.02, str(i), fontsize=9, color="red")

# Add grid, axis limits, and labels
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Random 2D Points in [-1, 1] × [-1, 1]")
plt.grid(True)

plt.show()