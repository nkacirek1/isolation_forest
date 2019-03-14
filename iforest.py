# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = math.ceil(math.log2(sample_size))
        self.trees = []

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        for i in range(self.n_trees):
            x_prime = X[np.random.choice(X.shape[0], size=self.sample_size, replace=False)]
            it = IsolationTree(self.height_limit)
            it.fit(x_prime, improved=improved)
            self.trees.append(it)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, computes the average path length
        for each observation in X. Computes the path length for x_i using every
        tree in self.trees then computes the average for each x_i. Returns an
        ndarray of shape (len(X),1).
        """
        result_array = np.zeros((X.shape[0], len(self.trees)))

        for i in range(X.shape[0]):
            x = X[i, :]
            one_obs = [self.path_length_recursive(x, T.root, 0.) for T in self.trees]
            result_array[i] = one_obs

        return np.mean(result_array, axis=1).reshape(X.shape[0], 1)

    def path_length_recursive(self, x, T, e):
        if isinstance(T, LeafNode):
            return e + self.c(T.x_shape_0)*1.0

        a = T.split_attr_index

        if x[a] < T.split_value:
            return self.path_length_recursive(x, T.left, e+1)
        if x[a] >= T.split_value:
            return self.path_length_recursive(x, T.right, e+1)

    def c(self, n):
        if n > 2:
            return (2.0 * (np.log(n-1.0) + 0.5772156649)) - (2.0 * (n - 1.0) / (n*1.0))
        elif n == 2:
            return 1.0
        else:
            return 0.0

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, computes the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        mean_path_lengths = self.path_length(X)
        return 2**(-mean_path_lengths / self.c(self.sample_size))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, returns an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        scores[scores >= threshold] = 1
        scores[scores < threshold] = 0
        return scores

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        """
        A shorthand for calling anomaly_score() and predict_from_anomaly_scores().
        """
        scores = self.anomaly_score(X)
        return self.predict_from_anomaly_scores(scores, threshold)


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None
        self.n_nodes = 0

    def fit(self, X: np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Returns
        the root of the tree.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """
        self.root = self.fit_recursive(X, depth=1, improved=improved)

        return self.root

    def fit_recursive(self, X: np.ndarray, depth: int, improved=False):

        if depth == self.height_limit or (X.shape[0] <= 1):
            return LeafNode(x_shape_0=X.shape[0])

        else:

            if improved:

                q_index, p = self.find_best_split(X, min(X.shape[1], 5), 5)
                q = X[:, q_index]
                min_val = np.min(q)
                max_val = np.max(q)

                if min_val == max_val:
                    return LeafNode(x_shape_0=X.shape[0])

                p = np.random.uniform(min_val, max_val)

            else:
                q_index = np.random.randint(0, X.shape[1])

                q = X[:, q_index]
                min_val = np.min(q)
                max_val = np.max(q)

                if min_val == max_val:
                    return LeafNode(x_shape_0=X.shape[0])

                p = np.random.uniform(min_val, max_val)

            X_left = X[q < p]
            X_right = X[q >= p]

            root = DecisionNode(split_attr_index=q_index, split_value=p)
            root.left = self.fit_recursive(X_left, depth+1)
            root.right = self.fit_recursive(X_right, depth+1)
            self.n_nodes += 2

        return root

    def find_best_split(self, X: np.ndarray, num_features: int, num_split_vals: int):
        """
        Imrpoved algorithm: tries multiple columns and multiple split values for 
        each of those columns. Returns the column index and column value that 
        yields the "best" split - aka the split that gives the smallest left or 
        right subregion
        """
        q_col_index = np.random.choice(X.shape[1], num_features, replace=False)
        best_p = []
        best_size = []

        for q_i in q_col_index:
            q = X[:, q_i]
            min_val = np.min(q)
            max_val = np.max(q)
            p_s = np.random.uniform(min_val, max_val, num_split_vals)
            min_p = p_s[0]
            min_child_size = 99999999999

            for p in p_s:

                if len(X[q < p]) < min_child_size:
                    min_p = p
                    min_child_size = len(X[q < p])
                    continue

                elif len(X[q >= p]) < min_child_size:
                    min_p = p
                    min_child_size = len(X[q >= p])

            best_p.append(min_p)
            best_size.append(min_child_size)

        best_index = best_size.index(min(best_size))

        return q_col_index[best_index], best_p[best_index]


class DecisionNode:
    def __init__(self, split_attr_index, split_value):
        self.split_attr_index = split_attr_index
        self.split_value = split_value


class LeafNode:
    def __init__(self, x_shape_0):
        self.x_shape_0 = x_shape_0


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Helper function to find the threshold to meet our desired True 
    Positive Rate (TPR).

    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.

    Used by scoring test rig
    """

    it = IsolationTreeEnsemble(100)

    threshold_range = np.arange(1.0, 0.0, -0.005)

    for threshold in threshold_range:
        temp = scores.copy()
        y_pred = it.predict_from_anomaly_scores(temp, threshold=threshold)
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        if TPR >= desired_TPR:
            return threshold, FPR

    print("never met condition")

    return -1
