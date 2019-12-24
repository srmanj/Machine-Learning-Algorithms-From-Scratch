import pandas as pd
import numpy as np
import itertools as itr
import random
import collections as col
import math
import time 
import pickle

class plant():
    def __init__(self,data = None):
        self.left = None
        self.right = None
        self.data = data
    
    def left_Insert(self, subtree):
        self.left = subtree

    def right_Insert(self,subtree):        
        self.right = subtree 
    
    def retreive_Left_Tree(self):
        return self.left
        
    def retreive_Right_Tree(self):
        return self.right

class decision_Tree():
    def __init__(self,**sysargs):
        self.TrainedTree = None
        self.featuresConsideredInTree = None
        self.isTrained = False
        
    def train(self,TrainX,TrainY):
        features = list(range(192))
        tree_Features = features[:192]
        decisionTree = self.construct_Tree(tree_Features,TrainX,TrainY)
        self.featuresConsideredInTree = tree_Features
        self.TrainedTree = decisionTree
        return self
    
    def predict(self,TestX):
        decisionTree = self.TrainedTree
        finalPredictionList = []
        for row in TestX:
            predictionForRow = self._traverse_Decided_Tree(decisionTree,row)
            finalPredictionList.append(predictionForRow)
        return np.array(finalPredictionList)
    
    def _create_seed(self,temp_Dict,pos_Labels,neg_Labels):
        leaf = plant(temp_Dict)
        if len(pos_Labels) > 0:
            l_Pred = max(col.Counter(pos_Labels), key=lambda k: col.Counter(pos_Labels)[k])
        else:
            l_Pred = None

        if len(neg_Labels) > 0:
            r_Pred = max(col.Counter(neg_Labels), key=lambda k: col.Counter(neg_Labels)[k])
        else:
            r_Pred = None

        l_Tree = plant({'colname':None,'alpha':None,'pred':l_Pred})
        r_Tree = plant({'colname':None,'alpha':None,'pred':r_Pred})

        leaf.left_Insert(l_Tree)
        leaf.right_Insert(r_Tree)
        return leaf
    
    def construct_Tree(self,feature_Names,feature_Rows,feature_Labels,depth = 10):
        #If there is only 1 class then directly create a node object
        if len(set(feature_Labels.tolist())) == 1 :
            dominant_Class, = set(feature_Labels.tolist())
            temp = {'colname':None,'alpha':None,'pred':dominant_Class} 
            return plant(temp)

        #When there are no rows
        if len(feature_Names) == 0 or len(feature_Labels) == 0:
            return None

        ##Computing the bestfeature to split and its threshhold
        minEntropy,bestSplitFeatureName ,bestSplitThresh = self._find_Split_Candidate(feature_Names,feature_Rows,feature_Labels)
        ##segregating observations based on the test conditions
        positiveObservations,negativeObservations,pos_Labels,neg_Labels = self._execute_Split(feature_Rows,bestSplitFeatureName,bestSplitThresh,feature_Labels)

        # as a dictionary of featurename, test threshold and prediction class
        temp = {'colname':bestSplitFeatureName,'alpha':bestSplitThresh,'pred':None} 
        if depth == 0:
            leaf_Node = self._create_seed(temp,pos_Labels,neg_Labels)
            return leaf_Node

        #Recursive self call
        left_Node =  self.construct_Tree(feature_Names,positiveObservations,pos_Labels,depth-1)
        right_Node = self.construct_Tree(feature_Names,negativeObservations,neg_Labels,depth-1)

        #Finally attach to the root node
        root_Node = plant(temp)
        root_Node.left_Insert(left_Node)
        root_Node.right_Insert(right_Node)
        return root_Node


    def _traverse_Decided_Tree(self,root_Node,row):
        trees = [root_Node]
        while len(trees) > 0 :
            child_Tree = trees.pop(0)
            #If at a leaf
            if child_Tree.retreive_Right_Tree() is None and child_Tree.retreive_Left_Tree() is None:
                prediction = child_Tree.data['pred']
                return prediction 

            if child_Tree.retreive_Left_Tree() is not None and (row[child_Tree.data['colname']] >= child_Tree.data['alpha']) == True:
                trees.append(child_Tree.retreive_Left_Tree())

            if child_Tree.retreive_Right_Tree() is not None and (row[child_Tree.data['colname']] >= child_Tree.data['alpha']) == False:
                trees.append(child_Tree.retreive_Right_Tree())      
        return "Failure" 
        
    def _computeEntropy(self,alpha,rows,labels):
        temp_Dict = col.defaultdict(lambda: col.defaultdict(int))
        numTotalObs = len(rows)      

        #Step 1- Split the Rows
        pos_Rows = rows[rows >= alpha]
        neg_Rows = rows[rows < alpha]

        #Step 2- Split the labels
        pos_Labels = labels[rows >= alpha]
        neg_Labels = labels[rows < alpha]

        temp_Dict['Positive'] = dict(col.Counter(pos_Labels))
        temp_Dict['Negative'] = dict(col.Counter(neg_Labels))

        #Calculate the probabilities and compute their entropy
        pos_Entropy = 0
        for Labelcount in temp_Dict['Positive'].values():
            pos = Labelcount/len(pos_Rows)
            pos_Entropy+=((-pos)*math.log2(pos))

        neg_Entropy = 0
        for Labelcount in temp_Dict['Negative'].values():
            neg = Labelcount/len(neg_Rows)
            neg_Entropy+=((-pos)*math.log2(pos))

        final_Entropy = (len(pos_Rows)/numTotalObs)*pos_Entropy + (len(neg_Rows)/numTotalObs)*neg_Entropy        
        return final_Entropy
    
    def _find_Split_Candidate(self,feature_Names,feature_Rows,feature_Labels):
        split_Candidates = []

        rows = feature_Rows[:,feature_Names]
        for featureNum, featureValues in zip(feature_Names,rows.T):
            # splitThresholds = np.int16(np.percentile(featureValues,[25,50,75]))
            splitThresholds = np.int16(np.percentile(featureValues,[50]))
            for splitThreshold in splitThresholds:
                split_Candidates.append((self._computeEntropy(splitThreshold,featureValues,feature_Labels),featureNum,splitThreshold))
        minEntropy,bestSplitFeatureName ,bestSplitThresh = min(split_Candidates)
        return minEntropy,bestSplitFeatureName ,bestSplitThresh 

    def _execute_Split(self,feature_Names,split_Candidate,alpha,feature_Labels):
        #Split the Positive rows
        pos_Rows = feature_Names[feature_Names[:,split_Candidate] >= alpha,:]
        pos_Labels = feature_Labels[feature_Names[:,split_Candidate] >= alpha]
        #Split the Negative rows
        neg_Rows = feature_Names[feature_Names[:,split_Candidate] < alpha,:]
        neg_Labels = feature_Labels[feature_Names[:,split_Candidate] < alpha]
        return pos_Rows,neg_Rows,pos_Labels,neg_Labels 
    
    
    
    