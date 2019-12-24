import numpy as np
import math
#RMS prop var

# contains all the data related to layer
class NeuralLayer_weights():
    def __init__(self,Nodes_present_layer,Nodes_next_layer,acti_val):
        self.Present_Nodes=Nodes_present_layer
        self.Next_Nodes=Nodes_next_layer
        self.Weights_vector=np.random.rand(Nodes_present_layer,Nodes_next_layer)
        self.gradient=None
        self.input=None
        self.output=None
        self.backpropoutput=None
        self.activation_val=acti_val
    

    # weights initialization and return
    def getWeights(self):
        return self.Weights_vector

    def setWeights(self,updated_weights):
        self.Weights_vector=updated_weights
    
    #input initialization and return
    def inputnodes(self,x):
        self.input=x

    def getinput(self):
        return self.input


    #cal output node
    def outputnodes(self):
        if self.activation_val==None:
            self.output=self.input
            return self.output
        elif self.activation_val=="sigmoid":
            #print("im here")
            #print(self.input)
            act_val = np.array(list(map(self.sigmoid,self.input)))
            self.output=act_val
            #print(self.output)
            return self.output

    def getoutput(self):
        return self.output

    ###calculate gradient,initialization and return--backprop
    def gradcalc(self):
        if self.activation_val==None:
            der_actVal=np.ones(len(self.input))
            self.backpropoutput=der_actVal
            return der_actVal
        elif self.activation_val=="sigmoid":
            act_val = np.array(list(map(self.derivation_sigmoid,self.input)))
            self.backpropoutput=act_val
            return act_val

    def set_gradient(self,gradient):
        self.gradient = gradient
        
    def get_gradient(self):
        return self.gradient
    
    #sigmoid activation ftn and sigmoid prime
    def sigmoid(self,input_x):
        activationValue = 1/(1+math.exp(-input_x))
        return activationValue

    def derivation_sigmoid(self,input_y):
        output = self.sigmoid(input_y)*(1-self.sigmoid(input_y))
        return output


class NeuralNetwork():
    def __init__(self,epochs, learning_rate,Num_Layer, NodesPerLayer,rho):
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.Num_Layer=Num_Layer
        self.NodesPerLayer=NodesPerLayer
        self.all_layers_weights=0
        self.rho=rho
    
    #initialization of weights for all the layers-->x_train(vec) input
    def Layers_weight_initialization(self,x_train):
        input_nodes=x_train.shape[1]
        Nodes_Input=[input_nodes]+self.NodesPerLayer
    
        Nodes_output=self.NodesPerLayer+[self.NodesPerLayer[-1]]
        layers_weights=[]
        activation_funtion=[None]
        activation_funtion += ["sigmoid"]*(len(self.NodesPerLayer))
        
        for (i,j,k) in zip(Nodes_Input,Nodes_output,activation_funtion):
            layers_weights.append(NeuralLayer_weights(i,j,k))
        self.all_layers_weights=layers_weights


    #fits the training data
    def fit(self,x_train,y_train):
        #x_train_bias=np.hstack((np.ones(len(x_train),1)),x_train)
        self.Layers_weight_initialization(x_train)
        accuracy=[]
        for epoch in range(0,self.epochs):
            for (x,y) in zip(x_train,y_train):
                self.feed_forward(x)
                self.backprop(y)
                self.updateweights()
        return self
    
    # forward propagation
    def feed_forward(self,x):
        for (layer_index,layer) in enumerate(self.all_layers_weights):
            if layer_index==0:
                layer.inputnodes(x)
                output=layer.outputnodes()
                nextlayer_input=np.dot(output,layer.getWeights())
            else:
                layer.inputnodes(nextlayer_input)
                output = layer.outputnodes()
                #output= np.array(list(map(float,output)))
                nextlayer_input=np.dot(output,layer.getWeights())
        return output
    
    # back propagation 
    def backprop(self,y_train):
        for (layer_index,layer) in enumerate(self.all_layers_weights[::-1]):
            if layer_index==0:
                output=layer.getoutput()
                backprop_out=layer.gradcalc()
                grad=backprop_out*(y_train-output)
                layer.set_gradient(grad)
            else:
                backprop_out=layer.gradcalc()
                temp=np.dot(layer.getWeights(),grad)
                grad=backprop_out*temp
                layer.set_gradient(grad)


    #updating the weights after every epoch 
    def updateweights(self):
        for currentLayer,nextLayer in zip(self.all_layers_weights,self.all_layers_weights[1:]):
            curr_weights=currentLayer.getWeights()
            delta=np.tile(nextLayer.get_gradient(),(curr_weights.shape[0],1))
            temp=(currentLayer.getoutput())[:,np.newaxis]
            prod_output_delta = float(self.learning_rate)*temp*delta
            UpdatedWeights = curr_weights+prod_output_delta
            currentLayer.setWeights(UpdatedWeights)


    # predict for the data--does only forward prop and for this prob maps the output the corresponding degree       
    def predict(self,xtest):
        predictions = np.array([self.feed_forward(x_t) for x_t in xtest])
        max_values = predictions.max(axis=1).reshape(-1, 1)
        pred = np.where(predictions == max_values, 1, 0)
        final_predictions=np.array(list(map(self.mapping,pred)))
        return final_predictions
    
    #mapping the output to corresponding degree
    def mapping(self,pred):
        pred=list(pred)
        if pred==[1,0,0,0]:
            return 0
        elif pred==[0,1,0,0]:
            return 90
        elif pred==[0,0,1,0]:
            return 180
        elif pred==[0,0,0,1]:
            return 270
        
