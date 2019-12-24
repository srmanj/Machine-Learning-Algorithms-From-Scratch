import numpy as np
import pandas as pd
class knn():
    def __init__(self,k):
        self.k=k
        self.trainX=None
        self.trainY=None
    
    def eucledian_dist(self,xtrain,xtest):
        xtr_sq=np.sum(xtrain**2,axis=1)
        xte_sq=np.sum(xtest**2,axis=1)[:, np.newaxis]
        prob_fac=(2)*np.dot(xtrain,xtest.T)
        res=xtr_sq+xte_sq-prob_fac.T
        return res
    
    def train(self,trainX,trainY):
        self.trainX=trainX
        self.trainY=trainY
        return self
    
    def find_output(self,matrix_knn):
        sorted_knn=matrix_knn.sort_values(by="distances")
        
        topK=np.array(sorted_knn["output"][:self.k])
        
        pred=np.bincount(topK).argmax()
        return pred
    
    def predict(self,testX):
        prediction=[]
        distance_mat=self.eucledian_dist(self.trainX,testX)
        for distances in distance_mat:
            matrix_knn = pd.DataFrame({'distances':distances,"output":self.trainY})
            prediction.append(self.find_output(matrix_knn))
        return prediction