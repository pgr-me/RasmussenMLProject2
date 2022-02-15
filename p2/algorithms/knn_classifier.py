#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, knn_classifer.py

This module provides the KNNClassifer class, which inherits from the KNN base class. KNNClassifier supports three
methods: default, edited, and condensed.

"""
# Standard library imports
import typing as t

# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p2.algorithms import KNN, minkowski_distance


class KNNClassifier(KNN):
    """
    k nearest neighbors classifier.
    """

    def __init__(self, data: pd.DataFrame, k: int, label: str, index: str, method: str = None):
        super().__init__(data, k, label, index, method)

    def determine_majority_label(self, k_nearest_distances: pd.DataFrame,
                                 empty_set_label: t.Union[int, bool, float] = -1) -> t.Union[int, bool, float]:
        """
        Determine the majority label among k nearest neighbors to query point.
        :param k_nearest_distances: KNN, distances from query point, indices, & labels
        :param empty_set_label: Empty set label used when there are zero votes for the majority label
        :return: Majority label
        If there are ties, randomly select one of the winners.
        """
        votes = k_nearest_distances[self.label].value_counts().sample(frac=1)
        if len(votes) == 0:
            majority_label = empty_set_label
        else:
            majority_label = votes.drop_duplicates().sort_values(ascending=False).index[0]
        return majority_label

    def predict(self, x_q: pd.Series) -> t.Union[int, bool, float]:
        """
        Predict the majority label of a query point x_q.
        :param x_q: Query point
        :return: Majority label
        """
        if self.lookup_table is None:
            raise ValueError("You must train the classifier before running the predict method.")
        distances = self.compute_distances(x_q, self.lookup_table)
        k_nearest_distances = self.find_k_nearest_distances(distances)
        majority_label = self.determine_majority_label(k_nearest_distances)
        return majority_label

    def train(self):
        """
        Train the k nearest neighbor classifier.
        Note: We don't need to "do" anything to the training mask if we're not editing or condensing.
        """

        if self.method is not None:
            for observation in self.observations:
                x_q = self.data.copy().loc[observation]

                # Compute distances for edited method
                if self.method == "edited":
                    X_t = self.data.copy()[self.training_mask].drop(labels=observation)
                    distances = minkowski_distance(x_q.iloc[1:], X_t.loc[:, 1:]).rename("distance")

                # Compute distances for condensed method
                else:
                    Z_t = self.data.copy()[self.training_mask]
                    # Compute distances among x_q and each observation in X_t
                    distances = minkowski_distance(x_q.iloc[1:], Z_t.loc[:, 1:]).rename("distance")

                # Sort distances
                sorted_distances = pd.Series(distances, name="distance").sort_values()
                k_nearest_distances = sorted_distances.iloc[:self.k].to_frame().join(self.data[[self.label]])

                # Determine majority label and extract true label
                majority_label = self.determine_majority_label(k_nearest_distances)
                true_label = x_q.iloc[0]

                # Conditionally update training_mask
                if majority_label != true_label:

                    # Update training mask for edited method
                    if self.method == "edited":
                        self.training_mask.loc[observation] = False

                    # Update training mask for condensed method
                    else:
                        self.training_mask.loc[observation] = True

        # Define lookup table as dataset masked by training mask
        self.lookup_table = self.data.copy()[self.training_mask]

        return self.training_mask

    def score(self, metric: str):
        pass
