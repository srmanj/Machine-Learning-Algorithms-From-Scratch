from dtrees import *
from nnet import *
from nearest import *
import sys
import pandas as pd

def read_data(filename_):
    test_data = pd.read_csv(filename_,header = None, sep = ' ' )
    col_list = ['path','label']
    for i  in test_data.columns[2:].tolist():
        col_list.append(i-2)
    test_data.columns = col_list

    X = np.array(test_data[col_list[2:len(col_list)]])
    Y = test_data['label']
    Path = test_data['path']
    return X,Y,Path


# This normalization code is inspired from the website 
# https://kenzotakahashi.github.io/scikit-learns-useful-tools-from-scratch.html
def normalization(X):
    new = X - np.mean(X, axis=0)
    return new / np.std(new, axis=0)
    

if __name__ == "__main__":

    if(len(sys.argv) < 4):
        raise Exception('Usage: \n./orient.py train train_file.txt model_file.txt [model]')
        sys.exit()  
    trainOrTest = sys.argv[1]
    modelName = sys.argv[4]
    if trainOrTest == 'train':
        print("Learning model...")
        train_file, model_file = sys.argv[2],sys.argv[3]
        #Read Train and Test data
        trainX,trainY,trainPath=read_data(train_file)
        if modelName == 'tree':
            #create class object and train model
            df = decision_Tree()
            model = df.train(trainX,trainY)
            # save the model to disk
            pickle.dump(model, open(model_file, 'wb'))

        elif modelName == 'nnet':
            epochs=100
            numlayers=3
            nodesPerLayer=[10,10,4]
            Learning_rate=0.01
            rho=0
            #creating object
            mynet=NeuralNetwork(epochs,Learning_rate,numlayers,nodesPerLayer,rho)
            trainX=normalization(trainX)
            final_model=mynet.fit(trainX,np.array(pd.get_dummies(trainY)))
            #save the model to disk
            pickle.dump(final_model, open(model_file, 'wb'))
        
        elif modelName=="nearest":
            k=48
            myclass=knn(k)
            final_model=myclass.train(trainX,trainY)
            #save the model to disk
            pickle.dump(final_model, open(model_file, 'wb'))
        elif modelName == "best":
            k=48
            myclass=knn(k)
            final_model=myclass.train(trainX,trainY)
            #save the model to disk
            pickle.dump(final_model, open(model_file, 'wb'))
        else:
            print("Enter the modelNames: tree,nnet,nearest,best")
            
    else:
        #Read Test data
        print("Loading test data...")
        test_file, model_file = sys.argv[2],sys.argv[3]
        testX,testY,testPath=read_data(test_file)
        if modelName == 'tree':
            # load the model from disk
            loaded_model = pickle.load(open(model_file, 'rb'))
            predictions = loaded_model.predict(testX)

            #Output the predictions
            output = pd.DataFrame({'path':testPath,'Predictions':predictions})
            output.to_csv(path_or_buf = 'output.txt' ,sep = ' ',header = False ,index = False)
            
        elif modelName=="nnet":
            # load the model from disk
            loaded_model = pickle.load(open(model_file, 'rb'))
            testX=normalization(testX)
            predictions = loaded_model.predict(testX)

            #writing to ouput file
            output = pd.DataFrame({'path':testPath,'Predictions':predictions})
            output.to_csv(path_or_buf = 'output.txt' ,sep = ' ',header = False ,index = False)
        
        elif modelName=="nearest":
            # load the model from disk
            loaded_model = pickle.load(open(model_file, 'rb'))
            predictions=loaded_model.predict(testX)

            #writing to ouput file
            output = pd.DataFrame({'path':testPath,'Predictions':predictions})
            output.to_csv(path_or_buf = 'output.txt' ,sep = ' ',header = False ,index = False)

        elif modelName=="best":
            # load the model from disk
            loaded_model = pickle.load(open(model_file, 'rb'))
            predictions=loaded_model.predict(testX)

            #writing to ouput file
            output = pd.DataFrame({'path':testPath,'Predictions':predictions})
            output.to_csv(path_or_buf = 'output.txt' ,sep = ' ',header = False ,index = False)

        else:
            print("Enter the modelNames: tree,nnet,nearest,best")
        
        print("Accuracy is: " ,sum(predictions==testY)/len(testY))
            
         
