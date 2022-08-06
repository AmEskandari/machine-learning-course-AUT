
import os
import copy

import numpy as np
import pandas as pd

from numpy import log2
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter


#Data reading ->
data_train = pd.read_csv(os.getcwd()+'/adult.train.10k.discrete', sep=",", header=None)
data_train.columns = ['label','workclass','education','marital-status','occupation',
                      'relationship','race','sex','native-country']
data_test = pd.read_csv(os.getcwd()+'/adult.test.10k.discrete', sep=",", header=None)
data_test.columns = ['label','workclass','education','marital-status','occupation',
                     'relationship','race','sex','native-country']

#Data preprocessing ->
enc = OrdinalEncoder()
data_encoded_train = enc.fit_transform(data_train)
feature_list = ['workclass','education','marital-status','occupation',
                'relationship','race','sex','native-country']
features_dict = {1:'workclass',2:'education',3:'marital-status',4:'occupation',5:'relationship'
                 ,6:'race',7:'sex',8:'native-country'}

value_dict = {label_item[1]:{item[0]:item[1] for item in enumerate(enc.categories_[label_item[0]])} 
              for label_item in enumerate(data_train.columns)}
feature_list_idx = list(range(1,data_encoded_train.shape[1]))

def makeTree(X, features_dict,feature_list,value_dict):
    """
    Core function of the code. In this function the tree grows gradually from the root. 
    Tree growth is guided by ID3 alghorithm.
    
    Inputs -->
        X : ordinary encoded train data
        feature_dict : a dict contains features name associated by their indexes
        features_list : contains features name
        value_dict : dictionary contain each feature and label indexes
    return -->
        return full grown tree by training data


    """
    X = copy.deepcopy(X)
    
    default  = Counter(X[:,0]).most_common(1)[0][0]
    
    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    #  need to subtract 1 to account for the target attribute.
    if  X.shape[0] == 1 or (len(feature_list) - 1) <= 0:
        return default
        
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif np.unique(X[:,0]).size == 1:
        return default 

    else:
        # Choose the next best attribute to best classify our data
        parent_entropy = entropy(X[:,0])
        gain_dic = {}

        for feature_idx in feature_list:
            gain_dic[feature_idx] = gain_cal(X[:,feature_idx],parent_entropy)

        best_feature_idx = max(gain_dic,key=gain_dic.get)

        tree = {features_dict[best_feature_idx]:{}}

        feature_unique_value = value_dict[features_dict[best_feature_idx]].keys()

        for val in feature_unique_value:
            examples = X[X[:,best_feature_idx] == val,:]
            
            if examples.shape[0] != 0:
                
                newAttr = feature_list[:]
                newAttr.remove(best_feature_idx)
                subtree = makeTree(examples,features_dict, newAttr,value_dict)
                tree[features_dict[best_feature_idx]][value_dict[features_dict[best_feature_idx]][val]] = subtree

        tree[features_dict[best_feature_idx]]['voted'] = default
    
    return tree

def prunning(tree,recursion,treshold):
    """
    Clear by name this function prune the tree.

    input -->
        tree : dic format full grown tree
        recursion : indicated the depth of tree tree 
        treshold : indicated in whiche depth prunning will happen
    return -->
        pruned tree
    """
    
    recursion += 1

    if recursion == treshold:
        if 'voted' in tree.keys():
            keys = list(tree.keys())
            for key in filter(lambda key: key!='voted',keys):
                tree.pop(key)
            return tree
            
    for node,subtree in tree.items():
        if isinstance(subtree,dict):
            tree[node] = prunning(subtree,recursion,treshold)

    return tree

def count_leaf(tree, c):
    """
    this function counts number of leaf and non-leaf nodes

    inpute ->
        tree : constructed tree
        c : empty tree
    return ->
        tree contains number of leaf and non-leaf
    """
    c['non-leaf']+=1    #count non-leaf nodes
    nodes = tree.keys()
    for node in nodes:
      subtrees = tree[node].values()
      for subtree in subtrees:
          if isinstance(subtree, dict):           
              count_leaf(subtree,c)
          else:
              c['leaf']+=1  #count leaf nodes and remove 'voted'

    return c

def entropy(y):
    """
    inpute -->
        y : list of labales
    return -->
        entropy of inpute list
    """
    count_labels = Counter(y)
    n = y.shape[0]
    entropy = 0
    if count_labels[0] != 0 & count_labels[1] !=0: 
        entropy = -(count_labels[0]/n)*log2(count_labels[0]/n)-(count_labels[1]/n)*log2(count_labels[1]/n)
    return entropy

def gain_cal(x,parent_entropy):
    """
    calculate gain
    
    inpute ->
        x: inpute sequence of data
        parent_entropy: associated entropy
    return ->
        gain
    """
    cnt_values = Counter(x)
    gain = parent_entropy
    for item in cnt_values.items():
        gain -= (-(item[1]/x.shape[0]) * log2(item[1]/x.shape[0]))
    return gain


def predictation(tree,record,label,value_dict):
    """
    Prediction each instance of data by tree

    Inpute: tree -> dict type data
    record: one instance of data in pandas dataFram format
    label: associated label 

    return : predicted label > '<=50K' or '>50K'
    """
    temptree = tree.copy()
    while isinstance(temptree,dict):
        #if (record[list(temptree.keys())[0]] != 'voted') & (record[list(temptree.keys())[0]] in list(temptree[list(temptree.keys())[0]].keys())):
        if record[list(temptree.keys())[0]] in list(temptree[list(temptree.keys())[0]].keys()):
            temptree = temptree[list(temptree.keys())[0]][record[list(temptree.keys())[0]]]
        else:
            return value_dict['label'][temptree[list(temptree.keys())[0]]['voted']]
    predict = value_dict['label'][temptree]
    return predict

def evaluation(tree,dataset,value_dict):
    """
    This function evaluates the inpute tree by inpute datasets.
    
    Inpute -> 
        tree : dict type data
        datasets: unencoded dataFrame
        value_dict: dictionary contain each feature and label indexes
    return ->
        accuracy of tree in Associated tree
    """

    correct = 0
    for idx in range(dataset.shape[0]):
        
        record = dataset.iloc[idx,1:]
        label = dataset.iloc[idx,0]
        predict = predictation(tree,record,label,value_dict)
        if predict == label:
            correct += 1
    
    return (correct / dataset.shape[0])*100

def questions_one(data_train,data_test,data_encoded_train,feature_list,feature_list_idx,value_dict,sampling_ratio,rng):
    """
    All part of question one implemented here. 
    
    inputs ->
        data_train : unencoded training dataframe 
        data_test : unencoded test dataframe
        data_encoded_train : ordinary encoded train data np array
        feature_list : contains feature name 
        feature_list_idx : associated feature idx
        value_dict : dictionary contain each feature and label indexes
        sampling_ration : demonstrate share of the training data
        rng : seed object for random generation

    return ->
        associated result in dictionary format 
    """

    result = {'iteration': [],'leaf':[],'train_accuracy':[],'test_accuracy':[]}
    
    for iteration in range(3):
        
        random_idx = rng.choice(data_train.shape[0],size = int(data_train.shape[0] * sampling_ratio),replace = False)
        data_train_sample = data_encoded_train[random_idx,:] 
        
        tree = makeTree(data_train_sample,features_dict,feature_list_idx,value_dict) 
        
        cnt_leaf = count_leaf(tree, {'non-leaf':0, 'leaf':0})
        train_accuracy = evaluation(tree,data_train,value_dict)
        test_accuracy = evaluation(tree,data_test,value_dict)
        
        result['iteration'].append(iteration)
        result['leaf'].append(cnt_leaf['leaf'])
        result['train_accuracy'].append(train_accuracy)
        result['test_accuracy'].append(test_accuracy)
        
        print(f'Constructed tree by {sampling_ratio*100} percent of train data, Iteration {iteration+1}:')
        print(f'Train Accuracy : {train_accuracy:.2f} | Test Accuracy : {test_accuracy:.2f}')
        print(f'Leaf Nodes: {cnt_leaf["leaf"]}')
        print(' ')
    
    print('-------------------------------------------')
    

    return result

def questions_two(data_train,data_test,data_encoded_train,feature_list,feature_list_idx,value_dict,section,rng):
    """
    part a and b of question two implemented here. 
    inputs ->
        data_train : unencoded training dataframe 
        data_test : unencoded test dataframe
        data_encoded_train : ordinary encoded train data np array
        feature_list : contains feature name 
        feature_list_idx : associated feature idx
        value_dict : dictionary contain each feature and label indexes
        section : assert the section of question
        rng : seed object for random generation

    return ->
        associated result in dictionary format 

    """
  
    train_idx = rng.choice(np.array(range(data_encoded_train.shape[0])), size = int(data_encoded_train.shape[0] * 0.75),replace = False)
    valid_idx = np.delete(np.array(range(data_train.shape[0])), train_idx)
    
    if section == 'a':
        print('Part a')
        tree = makeTree(data_encoded_train[train_idx,:],features_dict,feature_list_idx,value_dict)
    elif section == 'b':
        print('Part b')
        tree = makeTree(data_encoded_train[:,:],features_dict,feature_list_idx,value_dict)
    
    print('Full tree number of leaf node:',count_leaf(tree,{'non-leaf':0, 'leaf':0})['leaf'])
        
    result = {'leaf':[],'train_accuracy':[],'valid_accuracy':[],'test_accuracy':[]}
    
    for iteration,treshold in enumerate(list(range(20)[::2])):
        temptree = copy.deepcopy(tree)
        pruned_tree = prunning(temptree,0,treshold) 
        
        if section == 'a':
            train_accuracy = evaluation(pruned_tree,data_train.iloc[train_idx,:],value_dict)
        elif section == 'b':
            train_accuracy = evaluation(pruned_tree,data_train,value_dict)

        valid_accuracy = evaluation(pruned_tree,data_train.iloc[valid_idx,:],value_dict)
        test_accuracy = evaluation(pruned_tree,data_test,value_dict)
        cnt_leaf = count_leaf(pruned_tree,{'non-leaf':0, 'leaf':0})
        
        result['leaf'].append(cnt_leaf['leaf'])
        result['train_accuracy'].append(train_accuracy)
        result['valid_accuracy'].append(valid_accuracy)
        result['test_accuracy'].append(test_accuracy)
        
        
        print(f'Iteration : {iteration} | Treshold : {treshold}')
        print(f'Number of leaf: {cnt_leaf["leaf"]}')
        print(f'Train accuracy: {train_accuracy:.2f} | Valid accuracy: {valid_accuracy:.2f} | Test accuracy: {test_accuracy:.2f}')
        print(' ')
    
    return result

def questions_two_bonus(data_encoded_train,data_train,data_test,feature_list,feature_list_idx,value_dict):
    
    """
    last part of question two implemented here. 
    inputs ->
        data_train : unencoded training dataframe 
        data_test : unencoded test dataframe
        data_encoded_train : ordinary encoded train data np array
        feature_list : contains feature name 
        feature_list_idx : associated feature idx
        value_dict : dictionary contain each feature and label indexes

    return ->
        associated result in dictionary format 
    """
    interval = int(data_encoded_train.shape[0]/4)
    
    result = {}
    
    for k in range(4):
        
        result[k+1] = {}
        valid_idx = np.array(range(k*(interval),(k+1)*interval))
        train_idx = np.delete(np.array(range(data_train.shape[0])), valid_idx)

        tree = makeTree(data_encoded_train[train_idx,:],features_dict,feature_list_idx,value_dict)    
        
        print(f'Full tree number of leaf node:',count_leaf(tree,{'non-leaf':0, 'leaf':0})['leaf'],f'in {k+1}th fold')
        result[k+1]['leaf'] = []
        result[k+1]['train_accuracy'] = []
        result[k+1]['valid_accuracy'] = []
        result[k+1]['test_accuracy'] = []
        
        for iteration,treshold in enumerate(list(range(20))[2::2]):
            
            temptree = copy.deepcopy(tree)
            pruned_tree = prunning(temptree,0,treshold)
            
            train_accuracy = evaluation(pruned_tree,data_train.iloc[train_idx,:],value_dict)
            valid_accuracy = evaluation(pruned_tree,data_train.iloc[valid_idx,:],value_dict)
            test_accuracy = evaluation(pruned_tree,data_test,value_dict)
            cnt_leaf = count_leaf(pruned_tree,{'non-leaf':0, 'leaf':0})
            
            result[k+1]['leaf'].append(cnt_leaf['leaf'])
            result[k+1]['train_accuracy'].append(train_accuracy)
            result[k+1]['valid_accuracy'].append(valid_accuracy)
            result[k+1]['test_accuracy'].append(test_accuracy)
            
            print(f'Fold: {k+1}')
            print(f'Iteration : {iteration} ')
            print(f'Number of leaf: {cnt_leaf["leaf"]}')
            print(f'Train accuracy: {train_accuracy:.2f} | Valid accuracy: {valid_accuracy:.2f} | Test accuracy: {test_accuracy:.2f}')
            print(' ')
    
    return result

if __name__ == '__main__':
    rng = np.random.default_rng(seed=3138)

    #----------Question One------------
    print('Question one part a: ')
    result = questions_one(data_train,data_test,data_encoded_train,feature_list,feature_list_idx,value_dict,0.3,rng)
    np.save(os.getcwd()+'/result_q_1.npy', result)
    
    print('Question one part b: ')
    for sampling_ratio in [0.4,0.5,0.6,0.7]:
        result = questions_one(data_train,data_test,data_encoded_train,feature_list,feature_list_idx,value_dict,sampling_ratio,rng)
        np.save(os.getcwd()+f'/result_q_1_p_b_sampling_{sampling_ratio}.npy', result)
    
    #----------Question Two-----------
    result = questions_two(data_train,data_test,data_encoded_train,feature_list,feature_list_idx,value_dict,'a',rng)
    np.save(os.getcwd()+'/result_q_2_p_a.npy', result) 

    result = questions_two(data_train,data_test,data_encoded_train,feature_list,feature_list_idx,value_dict,'b',rng)
    np.save(os.getcwd()+'/result_q_2_p_a.npy', result) 

    result = questions_two_bonus(data_train,data_test,data_encoded_train,feature_list,feature_list_idx,value_dict,'b',rng)
    np.save(os.getcwd()+'/result_q_2_p_bounos.npy', result) 




    