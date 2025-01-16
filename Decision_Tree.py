import numpy as np
from collections import Counter
import nltk
import math
import time
#import movies_data_utils as mdu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

NUM_TRAINING_EXAMPLES = 5

class Node:
    def __init__(self, feature=None, threshold=None, right=None, left=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value
        
    def is_leaf(self):
        '''
        Returns true false if node is a leaf
        '''
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth = 75, min_s_s = 2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_s_s
        self.n_features = n_features
        self.root = None


    def _most_common_label(self, y):
        counts = Counter(y)
        val = counts.most_common(1)[0][0]
        return val
    
    def _entropy(self, y):
        hist = np.bincount(y)
        #P(x)
        px = hist/len(y)
        
        
        return -1*np.sum([p*np.log2(p) for p in px if p>0])


    def _split(self, x_column, threshold):
        left_idxs = np.argwhere(x_column < threshold).flatten()
        right_idxs = np.argwhere(x_column>= threshold).flatten()
        return left_idxs, right_idxs

    def _information_gain(self, y, x_column, threshold):
        # parent entropy

        p_entropy = self._entropy(y)
        #create children

        left_idx, right_idx = self._split(x_column, threshold)

        
        if len(left_idx) == 0 or len(right_idx)==0:
            return 0
        
        
        #calculate their weighted entropy of children

        n = len(y)

        n_left, n_right = len(left_idx), len(right_idx)


        e_left, e_right = self._entropy(np.take(y, [int(x) for x in left_idx])), self._entropy(np.take(y, [int(x) for x in right_idx]))

        child_entropy = (n_left/n)*e_left + (n_right/n)*e_right

        #return info gain

        return  p_entropy - child_entropy




    def _best_split(self, x, y, feat_indices):


        best_gain = -1

        split_idx, split_threshold = None, None

        for feat_idx in feat_indices:
            x_column = x[:, feat_idx]
            thresholds = np.unique(x_column)
            for thr in thresholds:
                #calculating info gain
                #print('\n\n\n THR', thr)
                gain = self._information_gain(y, x_column, thr)
                
                if gain >best_gain:
                    best_gain=gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

            
    def _grow_tree(self, x, y, depth=0):
        #check stop
        print(depth)
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels==1 or n_samples< self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        #find split
        feat_idx = np.random.choice(n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(x, y, feat_idx)
        print('best_feature', best_feature)
        print('best_thresh', best_threshold)
        #create children

        left_idxs, right_idxs = self._split(x[:,best_feature], best_threshold)
        left = self._grow_tree(x[left_idxs, :], np.take(y, [int(x) for x in left_idxs]), depth+1)
        right = self._grow_tree(x[right_idxs, :], np.take(y, [int(x) for x in right_idxs]), depth+1)

        return Node(best_feature, best_threshold, left, right)

    def fit(self, x, y):
        self.n_features = x.shape[1] if not self.n_features else min(x.shape[1], self.n_features)
        self.root = self._grow_tree(x, y)

        

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, x):
        return np.array([self._traverse_tree(i, self.root) for i in x])

