import numpy as np
from collections import Counter
import nltk
import math
import time
import movies_data_utils as mdu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

NUM_TRAINING_EXAMPLES = 5

class RandomForestClassifier:
    # Calculate entropy of a probability distribution
    def entropy(p):
        pass

    # Calculate information gain from splitting a node
    def information_gain(left_child, right_child):
        pass

    # Generate bootstrap samples from the training data
    def draw_bootstrap(X_train, y_train):
        pass

    # Calculate the out-of-bag (OOB) score for a tree
    def oob_score(tree, X_test, y_test):
        pass

    # Find the optimal split point in the data
    def find_split_point(X_bootstrap, y_bootstrap, max_features):
        pass

    # Create a terminal node in the decision tree
    def terminal_node(node):
        pass

    # Recursively split nodes in the decision tree
    def split_node(node, max_features, min_samples_split, max_depth, depth):
        pass

    # Build a decision tree using bootstrap samples
    def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
        pass

    # Build a random forest from multiple decision trees
    def random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
        pass

    # Predict output using a single decision tree
    def predict_tree(tree, X_test):
        pass

    # Predict output using the ensemble of trees in a random forest
    def predict_rf(tree_ls, X_test):
        pass

    # Description of what a tree looks like (the first 3 nodes)
    
