#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, knn.py

This module provides the KNN class, the base class of KNNClassifier and KNNRegressor classes.

"""
# Third party libraries
import pandas as pd

# Local imports
from p2.algorithms.utils import minkowski_distance


class KNN:
    """
    Base class for k nearest neighbors classification and regression models.
    We use a "training mask" as a way to subset the data for edited and condensed methods.
    """

    def __init__(self, data: pd.DataFrame, k: int, label: str, index: str, method: str = None, p: int = 2):
        self.data = data
        self.k = k
        self.label = label
        self.index = index
        methods = [None, "edited", "condensed"]
        if method not in methods:
            raise ValueError(f"Method {method} is not one of {methods}.")
        self.method = method
        self.p = p

        self.observations = self.data.index.tolist()

        # Initialize the training mask
        if method == "condensed":
            bools = [False for x in range(len(self.data))]
        else:
            bools = [True for x in range(len(self.data))]
        self.training_mask = pd.Series(bools, index=self.data.index)

        # Other variables to initialize
        self.lookup_table: pd.DataFrame = None

    def compute_distances(self, x_q: pd.Series, X_t: pd.DataFrame) -> pd.Series:
        """
        Compute Minkowski distance between query point and X_t.
        :param x_q: Query point
        :param X_t: Lookup table
        :param p: Minkowski p-norm
        :return: Distances between query point and lookup table
        Example output:
            index
            0      7.117855
            2      6.058951
                     ...
            995    5.711550
            997    6.415379
            Name: distance, Length: 186, dtype: float64
        """
        distances = minkowski_distance(x_q, X_t, self.p)
        return distances

    def find_k_nearest_distances(self, distances: pd.Series, name: str = "distance"):
        """
        Sort distances from query point.
        :param distances: Indexed series of distances from query point
        :return: k-nearest neighbors, their distances from query point, indices, & labels
        Example output:
                   distance    y
            index
            839    3.878051  1.0
            17     3.912822  1.0
            653    4.056486  1.0
        """
        k_sorted_distances = pd.Series(distances, name=name).sort_values().iloc[:self.k]
        k_sorted_distances = k_sorted_distances.to_frame().join(self.data[[self.label]])
        return k_sorted_distances
