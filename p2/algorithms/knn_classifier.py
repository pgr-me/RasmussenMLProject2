#!/usr/bin/env python3
"""Peter Rasmussen, Programming Assignment 2, knn_classifer.py

This module provides the KNNClassifer class, which inherits from the KNN base class. KNNClassifier supports three
methods: default, edited, and condensed.

"""
# Third party libraries
import numpy as np
import pandas as pd

# Local imports
from p2.algorithms import KNN


class KNNClassifier(KNN):
    """
    k nearest neighbors classifier.
    """

    def __init__(self, data: pd.DataFrame, k: int, label: str, index: str, method: str = None):
        super().__init__(data, k, label, index, method)

    def determine_majority_label(self, k_sorted_distances: pd.DataFrame) -> t.Union[int, bool, float]:
        """
        Determine the majority label among k nearest neighbors to query point.
        :param k_sorted_distances: KNN, distances from query point, indices, & labels
        :return: Majority label
        """
        votes = distances_labels[self.label].value_counts().sample(frac=1)
        majority_label = votes.drop_duplicates().sort_values(ascending=False).index[0]
        return majority_label

    def predict(self):
        pass

    def train(self):
        """
        Train the k nearest neighbor classifier.
        We don't need to "do" anything to the training mask if we're not editing or condensing.
        """

        if self.method is not None:
            for observation in self.observations:
                x_q = self.data.copy().loc[observation]

                # Compute distances for edited method
                if self.method == "edited":
                    X_t = data.copy()[edit_mask].drop(labels=observation)
                    distances = minkowski_distance(x_q.iloc[1:], X_t.loc[:, 1:]).rename("distance")

                # Compute distances for condensed method
                else:
                    Z_t = data.copy()[Z_mask]
                    # Compute distances among x_q and each observation in X_t
                    distances = minkowski_distance(x_q.iloc[1:], Z_t.loc[:, 1:]).rename("distance")

                # Sort distances
                sorted_distances = pd.Series(distances, name="distance").sort_values()
                distances_labels = sorted_distances.iloc[:k].to_frame().join(data[[label]])

                # Determine majority label and extract true label
                majority_label = get_majority_label(distances_labels, label, k)
                true_label = x_q.iloc[0]

                # Conditionally update training_mask
                if majority_label != true_label:

                    # Update training mask for edited method
                    if self.method == "edited":
                        self.training_mask.loc[observation] = False

                    # Update training mask for condensed method
                    else:
                        self.training_mask.loc[observation] = True

    return self.training_mask


def score(self, metric: str):
    pass
