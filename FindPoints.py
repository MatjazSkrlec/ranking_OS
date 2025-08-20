'''
Author: Matjaz Skrlec
date: 20. 08. 2025
version: 1.0
description: This code is created for doing the smart network operations in the point cloud presented
'''

# sys.path.append("C:\\Program Files\\DIgSILENT\\PowerFactory 2024 Preview\\Python\\3.12")
# import powerfactory as pf




#--------------------------------------------------------------------------------------------------
## imports

import polars as pl
import numpy as np
import statistics
from sklearn.neighbors import NearestNeighbors

#--------------------------------------------------------------------------------------------------
## functions(
def CheckNpArrayForm(arr):
    # check if all entries have the same len
    if arr.dtype == object:
        lengths = [len(x) for x in arr]
        if len(set(lengths)) == 1:
            return True
        else:
            return False
    else:
        # It's already a proper NumPy 2D array
        return True

def GetMostIsolatedPoint(Data):
    nbrs = NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree').fit(Data)
    distances, indices = nbrs.kneighbors(Data)
    max_dist_point = (0, 0)
    for dist, ind in zip(distances, indices):
        avg_dist = statistics.mean(dist[1:])
        if avg_dist > max_dist_point[1]:
            max_dist_point = (ind, avg_dist)
    return (indices, distances, max_dist_point)

def CreatePLDfFromNpArray(arr):

    #If they are not all equal
    if not CheckNpArrayForm(arr):
        print("The given Np array does not contain observation points of the same length. Can not create polaris dataframe.")
        return False

    # create row and column indecies
    rows, cols = np.shape(arr)

    rows_names = []
    cols_names = []
    for i in range(rows):
        rows_names.append(f"ob{i}")
    
    for i in range(cols):
        cols_names.append(f"x{i}")

    # create dist for polaris dataframe
    DfDict = {
        "RowLabel": rows_names
    }

    for ind, col_name in enumerate(cols_names):
        DfDict[col_name] = arr[:,ind]

    # create final DF
    return(pl.DataFrame(DfDict))

# def RemoveMostIsolatedPoint(Data):


#--------------------------------------------------------------------------------------------------
## files used in code
#--------------------------------------------------------------------------------------------------
## variable parameters of code
#-------------------------------------------------------------------------------------------------
## code