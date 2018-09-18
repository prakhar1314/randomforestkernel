import numpy as np
from sklearn.svm import SVR
from sklearn import cross_validation
from numpy import unravel_index

def get_lineage(tree,leaf_index,node_depth):
  left      = tree.tree_.children_left
  right     = tree.tree_.children_right
  threshold = tree.tree_.threshold
  lineage = []

  for child in leaf_index:
    lineageTemp = [child]
    parent = 1  # Something else than zero
    while parent !=0:
      # if lineageTemp is None:
        # lineageTemp = [child]
        # pass
      if child in left:
        parent = np.where(left == child)[0].item()
        split = 'l'
        depth = node_depth[parent]
      else:
        parent = np.where(right == child)[0].item()
        split = 'r'
        depth = node_depth[parent]

      child = parent
      lineageTemp.append(parent)

      if parent == 0:
        lineageTemp.reverse()
        lineage.append(lineageTemp)
        break

  return lineage
     
def get_leaves(children_left,children_right,n_nodes):

  is_leaves = np.zeros(shape=n_nodes, dtype=bool)
  node_depth = np.zeros(shape=n_nodes)
  stack = [(0, -1)]  # seed is the root node id and its parent depth
  while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
      stack.append((children_left[node_id], parent_depth + 1))
      stack.append((children_right[node_id], parent_depth + 1))
    else:
      is_leaves[node_id] = True
            
  return (is_leaves,node_depth)
      
def select_parameters(C,gamma,features,measures):

    scores = np.zeros([len(C),len(gamma)])
    for i, c in enumerate(C):
        for j, g in enumerate(gamma):
            clf = SVR(kernel='rbf', C=c, gamma=g)
            scoresTemp = cross_validation.cross_val_score(clf, features, measures, cv=5)
            # print(scoresTemp)
            scores[i][j] = -np.mean(scoresTemp)
    idx = unravel_index(scores.argmin(), scores.shape)
    print(idx)
    return C[idx[0]], gamma[idx[1]]

def normalize_features(features):
    out = features - np.mean(features, axis=0)
    out = np.divide(out,np.std(features,axis=0))

    return out