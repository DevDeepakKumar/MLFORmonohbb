import numpy as np
import pandas as pd

class DecisionTree:

    def __init__(self, criterion="gini", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict(x))
        return predictions

    def _build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return np.unique(y)[0]

        if self.max_depth is not None and self.max_depth <= 0:
            return np.unique(y)[0]

        best_feature, best_threshold = self._find_best_split(X, y)

        left_X = X[X[best_feature] <= best_threshold]
        right_X = X[X[best_feature] > best_threshold]
        left_y = y[X[best_feature] <= best_threshold]
        right_y = y[X[best_feature] > best_threshold]

        left_child = self._build_tree(left_X, left_y)
        right_child = self._build_tree(right_X, right_y)

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left_child": left_child,
            "right_child": right_child
        }

    def _find_best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in X.columns:
            if X[feature].dtype == "object":
                continue

            unique_values = np.unique(X[feature])
            for threshold in unique_values:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        entropy_before = self._entropy(y)

        left_X = X[X[feature] <= threshold]
        left_y = y[X[feature] <= threshold]
        entropy_left = self._entropy(left_y)

        right_X = X[X[feature] > threshold]
        right_y = y[X[feature] > threshold]
        entropy_right = self._entropy(right_y)

        gain = entropy_before - (len(left_y) / len(y) * entropy_left + len(right_y) / len(y) * entropy_right)

        return gain

    def _entropy(self, y):
        if len(np.unique(y)) == 1:
            return 0

        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def _predict(self, x):
        node = self.tree
        while node["left_child"] is not None or node["right_child"] is not None:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left_child"]
            else:
                node = node["right_child"]

        return node
