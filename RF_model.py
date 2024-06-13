import numpy as np
from collections import Counter
import nltk
import math
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  # PROVIDED
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents

fake_path = "FakeNewsNet/dataset/politifact_fake.csv"
df = pd.read_csv(fake_path)
fake_titles = df['title'].tolist()
print(fake_titles[0])  # Display the first few titles to verify

# PROVIDED
tfidf_vectorizer = TfidfVectorizer(max_features=500)
# this will return a sparse matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(fake_titles)
# change the sparse matrix to an array
tfidf_matrix = tfidf_matrix.toarray()

print("tfidf matrix: ", tfidf_matrix[0])
non_zero_percent = np.count_nonzero(tfidf_matrix[0]) / len(tfidf_matrix[0])


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
    
