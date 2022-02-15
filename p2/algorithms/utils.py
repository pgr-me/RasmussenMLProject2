#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, utils.py

This module provides miscellaneous utility functions that support the core algorithms of this program.

"""
# Third party libraries
import numpy as np
import pandas as pd


def minkowski_distance(x: pd.Series, y, p=2):
    """
    Compute the Minkowski distance.
    :param x: x vector
    :param y: y vector or matrix
    :param p: p-norm
    p-norm = 1 for Manhattan distance, 2 for Euclidean distance, and ca
    """
    diff = np.abs(x - y).T
    power = np.power(diff.T, p * np.ones(len(diff))).T
    power_sum = np.sum(power, axis=0)
    return np.power(power_sum.T, 1/p)
