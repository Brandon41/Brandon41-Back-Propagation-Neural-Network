#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
import os
#!pip install pyexcel
import math


i = 0
X_values = [] * 500
while i <= 100: 
      i += 0.2
      X_values.append(i)
      


count = 0
Y_values = [] * 500
while count <= 499:
    Y_values.append(1/X_values[count])
    count += 1

print('\n')
print('\n')
plt.figure(figsize=(20,10))
plt.xlabel('X_values')
plt.ylabel('Y_values')
#plt.plot(X_values, Y_values)
plt.scatter(X_values, Y_values)
plt.show()

count = 0
Test_Xvalues = [] * 100
Test_Yvalues = [] * 100
Training_Xvalues = [] * 400
Training_Yvalues = [] * 400

while count <= 499:
      if count % 5 == 0:
            Test_Xvalues.append(X_values[count])
          
            Test_Yvalues.append(Y_values[count])
      else:
        Training_Xvalues.append(X_values[count])
       
        Training_Yvalues.append(Y_values[count])    
     
      count += 1
        
x1 = len(Test_Xvalues)
x2 = len(Test_Yvalues)
x3 = len(Training_Xvalues)
x4 = len(Training_Yvalues)
x5 = len(X_values)
print('\n')
print(x1)
print(x2)
print(x3)
print(x4)
print(x5)

print('\n')
print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Training_Xvalues')
plt.ylabel('Training_Yvalues')
plt.scatter(Training_Xvalues, Training_Yvalues)
plt.show()

print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Test_Xvalues')
plt.ylabel('Test_Yvalues')
plt.scatter(Test_Xvalues, Test_Yvalues)
plt.show()



########################################################
#How to initialize a one d-array
#Y_values = []
#Y_values = [0 for i in range(500)] 

#How to initialize a 2-d array
#t = [ [0]*4 for i in range(5)]
#row = 0
#col = 0
#while row < 5:
#    col = 0
#    while col < 4:
#        t[row][col] = 1
#        print(col)
#        col += 1
#    row += 1
#print('\n')
#print(t)
#the first value 2 is the col and the second value 10 is the row
#x = [[0 for i in range(2)] for j in range(10)]
#print("The count is "  + str(count) + " that is count")
#print("Please, what is the number")
#input_a = input()
#print(t)
# string input 
#input_a = input() 
  
# print type 
#print(type(input_a))   
# integer input 
#input_b = int(input())   
# print type 
#print(type(input_b))


######################################################################




#NodesOfInputs = number of nodes in the input layer
#NodesOfOutputs =number of nodes in the output layer
#NumberOfHiddenLayer = number of Hidden layer
def gradient_descent(NodesOfInputs, NodesOfOutputs, NumberOfHiddenLayer):
    print('\n')
    
    #This create array with the size of the hidden layer
    #We want each index of the array to be a hidden layer and the number is carries 
    # the node of a hidden layer
    #For example index 0 stands for hidden layer 1 and the number it holds the nodes in hidden layer 1
    Hidden = []
    Hidden = [0 for i in range(NumberOfHiddenLayer)] 
    
    #InputNodes = 1
    
    print("print out the input node")
    print(NodesOfInputs)
    
    
    #Find the number of nodes for each hidden layer
    count = 0
    while count < NumberOfHiddenLayer:
        print("Please Enter the Nodes of Hidden Layer " + str(count))
        Nodes = int(input()) 
        Hidden[count] = Nodes
        count += 1
    
    #Find the Total number of bias
    count = 0
    bias = 0
    bias = NodesOfOutputs
    while count < NumberOfHiddenLayer:
        if count == 0:
            bias = NodesOfOutputs
            bias += Hidden[count] 
        bias += Hidden[count]
        count += 1
    
    #create an array for the bias and initialize to 0
    BiasArray = []
    BiasArray = [0 for i in range(bias)] 
    
    print("print out outputnodes")
    print(NodesOfOutputs)
    print("print out the array with all the bias")
    print(BiasArray)
    
    #Find the Total number of weights
    count = 0
    weights = NodesOfInputs
    hold1 = NumberOfHiddenLayer - 1
    while count < NumberOfHiddenLayer:
        weights += weights * Hidden[count]
        if count == hold1:
            weights += weights * NodesOfOutputs
        count += 1
    print("print out the weights")
    print(weights)
    #create an array for the weights and initialize to 0.5
    WeightsArray = [0.5 for i in range(weights)] 
    
    count = 0
    hold1 = 0
   
    
    length = len(Training_Xvalues)
    HoldArray = Training_Xvalues
    
    #create a container which will hold all the values for each node
    container = [[0 for i in range(length)] for j in range(bias)]
    
    
    #keep track of which node 
    N = 0
    #to keep track of the weights
    W = 0
    #to keep track of the bias
    B = 0

    # If the number of hidden layer is eqaul to zero
    if NumberOfHiddenLayer == 0:
            
        #iterate through each node of the output layer
        count6 = 0
        while count6 < NodesOfOutputs:
            
            hold2 = 1
            print("print count6")
            print(count6)
            #condition if the input has only one node
            if NodesOfInputs == 1:
                
                # transversing the entire array
                count7 = 0
                while count7 < length:
                    #The only time you transverse an output node, replace the value in the container
                    container[N][count7] = Training_Xvalues[count7] * WeightsArray[W]
                    hold1 = count6
                    count7 += 1    
                
                print("print out container")
                print(container)
                
                
              
                
            #condition if the input is greater than one node
            else:
                # Transversing each input Node
                count8 = 0
                while count8 < NodesOfInputs:
                    count9 = 0
                    # Transversing the length of the array or the column
                    while count9 < length:
                        # If you are in the first weight iteration of the output node, replace all the values in the container
                        if hold2 != count8:
                            container[N][count9] = Training_Xvalues[count9] * WeightsArray[W]
                            hold2 = count8
                        # if you are not in the first weight iteration just add the input weights to the current values
                        else:
                            container[N][count9] += Training_Xvalues[count9] * WeightsArray[W]
                        count9 += 1
                    #increment the current weight as the input Node changes
                    W += 1
                    count8 += 1
                    hold2 += 1
            #increment the output node, count 6, the weight, and the current node
            count6 +=1
            W +=1
            N +=1
            
    
    
    
    
    
    
    
    
    #iterate through the hidden layers
    count1 = 0
    while count1 < NumberOfHiddenLayer:
        
        
      
        #########################iterate through the node of each hidden layer
        
        count2 = 0
        M = Hidden[count1]
        while count2 < M:
            
                hold4 = 1
                hold5 = 1
                hold7 = 1
                #condition if the input has only one node
                if NodesOfInputs == 1:
                    # first iteration through the hidden layers
                     if count1 == 0:
                        #transversing the entire array        
                        count3 = 0
                        while count3 < length:
                            container[N][count3] = Training_Xvalues[count3] * WeightsArray[W]
                            count3 += 1
            
                     #When the hidden layer is at the last layer before the output layer
                     if count1 == NumberOfHiddenLayer-1:
                         count10 = 0
                         # going through each node in the output layer
                         while count10 < NodesOfOutputs:
                            count4 = 0
                            hold3 = 1
                             # going through each node in the last hidden layer, before the output layer
                            M = Hidden[count1] 
                            while count4 < M:
                                  count3 =0
                                  # tranversing the array of the inputs
                                  while count3 < length:
                                     #For every first iteration of the node in the output layer, replace the current values
                                     if hold3 !=count4:
                                         container[N][count3] = Training_Xvalues[count3] * WeightsArray[W]
                                         hold3 = count4
                                     else:
                                         container[N][count3] += Training_Xvalues[count3] * WeightsArray[W]
                                     count3 += 1
                                  count4 += 1
                                  hold3 += 1
                                  W += 1
                            count10 += 1
                            W += 1
                            N +=1
            
                     #When you still have multiple remaining hidden layers
                     else:
                         count10 = 0
                         # Iterate through the nodes of the previous hidden layer
                         M = Hidden[count-1]
                         while count10 < M:
                                 count3 = 0
                                 #transversing the entire array for each node of the hidden layer
                                 #and adding the values of weights of the previous node to that node
                                 while count3 < length:
                                     if hold4 != count10:
                                         container[N][count3] = Training_Xvalues[count3] * WeightsArray[W]
                                         hold4 = count10
                                     else:
                                         container[N][count3] += Training_Xvalues[count3] * WeightsArray[W]
                                     count +=3
                                 count10 += 1
                                 hold4 += 1
                                 W +=1
                        

            
                #condition if the input is greater than one node
                else:
                    # going through the very first iteration
                    if count1 == 0:
                        # Transversing each input Node
                        count4 = 0
                        while count4 < NodesOfInputs:
                                count5 = 0
                                # Transversing the length of the array or the column
                                while count5 < length:
                                    # If you are in the first input replace all the values
                                    if hold5 != count4:
                                        container[N][count5] = Training_Xvalues[count5] * WeightsArray[W]
                                        hold5 = count4
                                    # if you are not in the first input just add the input weights to the current values
                                    else:
                                        container[N][count5] += Training_Xvalues[count5] * WeightsArray[W]
                                    count5 += 1
                                #increment the current weight as the input Node changes
                                W += 1
                                count4 += 1
                                hold5 += 1
               
                    
                     #When the hidden layer is at the last layer before the output layer
                    if count1 == NumberOfHiddenLayer-1:
                        count10 = 0
                        while count10 < NodesOfOutputs:
                            count4 = 0
                            hold6 = 1
                            while count4 < Hidden[count1]:
                                count3 =0
                                while count3 < length:
                                    if hold6 != count4:
                                        container[N][count3] = Training_Xvalues[count3] * WeightsArray[W]
                                        hold6 = count4
                                    else:
                                        container[N][count3] += Training_Xvalues[count3] * WeightsArray[W]
                                    count3 += 1
                                count4 += 1
                                hold6 += 1
                                W += 1
                            count10 += 1
                            W += 1
                            N +=1
                    
                    # When you still have multiple remaining hidden layers
                    else:
                        count10 = 0
                        # Iterate through the previous value nodes
                        while count10 < Hidden[count-1]:
                            count3 = 0
                            #transvering the entire array for each node of the hidden layer
                            # and adding the values of weights of the previous node to that node
                            while count3 < length:
                                if hold7 != count10:
                                    container[N][count3] = Training_Xvalues[count3] * WeightsArray[W]
                                    hold7 = count10
                                else:
                                    container[N][count3] += Training_Xvalues[count3] * WeightsArray[W]
                                count +=3
                            count10 += 1
                            hold7 += 1
                            W +=1
                
                
                #increment the current Node as you switch to the next node in the hidden layer
                #increment the current weight as you switch to the next node in the hidden layer
                N += 1
                W += 1
                print(container)
                count2 += 1
        count1 += 1
    print("print out the container")
    print(container)
       
        
    
gradient_descent(1, 1, 1)
        
       
        


# In[50]:


import pandas as pd
from PIL import Image
from numpy import asarray
import numpy as np
import math
import matplotlib.pyplot as plt
import os






#initializing random weights
w1 = 0.3
w2 = 0.4
w3 = 0.5
w4 = 0.2
w5 = 0.7
w6 = 0.6
w7 = 0.3
w8 = 0.1

#initialize the bias to be a constant 1
bias = 1



#this will save the X values at each node
array1 = [1, 2, 3, 4, 5]
array2 = [1, 2, 3, 4, 5]
array3 = [1, 2, 3, 4, 5]
array4 = [1, 2, 3, 4, 5]
array5 = [1, 2, 3, 4, 5]



#this will save the Y values at each node
Node1 = [1, 2, 3, 4, 5]
Node2 = [1, 2, 3, 4, 5]
Node3 = [1, 2, 3, 4, 5]
Node4 = [1, 2, 3, 4, 5]
Node5 = [1, 2, 3, 4, 5]



#intialize the learning rate
learningrate = 0.03


#initialize the actual Y value for the output
observe = [1, 2, 5, 8, 11]


#initialize the actual X value for the input
arr = [1, 2, 3, 4, 5]


#Graph the X and Y values 
print('\n')
plt.xlabel('Xvalues')
plt.ylabel('Yvalues')
plt.scatter(arr, observe)
plt.show()



# Train your data
epoch = 0
while epoch < 10000:
    

    # Weight1 multiplied by the values of the input array
    array1 = w1 * np.array(arr) + bias
    
    # Weight2 multiplied by the vales of the input array
    array2 = w2 * np.array(arr) + bias
    
    print(array1)
    


    count = 0
    while count < 5:
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1[count] = 1 / (1 + math.exp(-(array1[count])))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2[count] = 1 / (1 + math.exp(-array2[count]))
        count += 1

    
    
    
    array3 = w3 * np.array(Node1) + w4 * np.array(Node2) + bias

    array4 = w5 * np.array(Node1) + w6 * np.array(Node2) + bias

    count = 0
    # function 
    while count < 5:
        # Node 1
        Node3[count] = 1 / (1 + math.exp(-array3[count]))
    
        # Node 2
        Node4[count] = 1 / (1 + math.exp(-array4[count]))
        count += 1





    array5 = w7 * np.array(Node3) + w8 * np.array(Node4) + bias
    count = 0
    while count < 5:  
        # Node 5
        Node5[count] = 1 / (1 + math.exp(-array5[count]))
        count += 1
    # The predicted Y values are stores at node 5
    predicted = Node5
    
    

    
    
    

    # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
    w1old = w1
    w2old = w2
    w3old = w3
    w4old = w4
    w5old = w5
    w6old = w6
    w7old = w7
    w8old = w8    
    
    #gradient descent
    count = 0
    while count < 5:
        # Get the derivative for each node using it's correspoding X values
        s = 1/(1 + math.exp(-array1[count]))
        dernode1 = s*(1-s)
    
        s = 1/(1 + math.exp(-array2[count]))
        dernode2 = s*(1-s)
    
        s = 1/(1 + math.exp(-array3[count]))
        dernode3 = s*(1-s)
    
        s = 1/(1 + math.exp(-array4[count]))
        dernode4 = s*(1-s)
    
        s = 1/(1 + math.exp(-array5[count]))
        dernode5 = s*(1-s)
    
    #Adding the derivative for all the indexes using sum of sqaure Residuals
    #For the first iteration, replace all the weights at that index
        if count == 0:
            
            w1 = -2 * (observe[count] - predicted[count]) * arr[count] * dernode1 * w3 * dernode1 * w5 * dernode3 * w7 * dernode4 * w8 * dernode5

            w2 = -2 * (observe[count] - predicted[count]) * arr[count] * w4 * dernode2 * w6 * dernode2 * w7 * dernode3 * w8 * dernode4 * dernode5
    
    
            w3 = -2 * (observe[count] - predicted[count]) * Node1[count] * dernode3 * w7 * dernode5
    
            w4 = -2 * (observe[count] - predicted[count]) * Node2[count] * dernode3 * w7 * dernode5
    
            w5 = -2 * (observe[count] - predicted[count]) * Node1[count] * dernode4 * w8 * dernode5
    
            w6 = -2 * (observe[count] - predicted[count]) * Node2[count] * dernode4 * w8 * dernode5
    
    
            w7 = -2 * (observe[count] - predicted[count]) * Node3[count] * dernode5
    
            w8 = -2 * (observe[count] - predicted[count]) * Node4[count] * dernode5
        
       #For all the other iteration, add the weight to that index to get your final weights 
        else:
            

        
            w1 += -2 * (observe[count] - predicted[count]) * arr[count] * dernode1 * w3 * dernode1 * w5 * dernode3 * w7 * dernode4 * w8 * dernode5      

            w2 += -2 * (observe[count] - predicted[count]) * arr[count] * w4 * dernode2 * w6 * dernode2 * w7 * dernode3 * w8 * dernode4 * dernode5
    
    
            w3 += -2 * (observe[count] - predicted[count]) * Node1[count] * dernode3 * w7 * dernode5
    
            w4 += -2 * (observe[count] - predicted[count]) * Node2[count] * dernode3 * w7 * dernode5
    
            w5 += -2 * (observe[count] - predicted[count]) * Node1[count] * dernode4 * w8 * dernode5
    
            w6 += -2 * (observe[count] - predicted[count]) * Node2[count] * dernode4 * w8 * dernode5
    
    
            w7 += -2 * (observe[count] - predicted[count]) * Node3[count] * dernode5
    
            w8 += -2 * (observe[count] - predicted[count]) * Node4[count] * dernode5
        
        count += 1
    
    #Using gradient descent to find the where the slope equals to zero fast for each weight.

    stepsize = w1 * learningrate
    print("print out learning rate")
    print(learningrate)
    
    print("print out the slope")
    print(w1)
    
    print("print step size")
    print(stepsize)
    
    w1 = w1old - stepsize


    stepsize = w2 * learningrate
    w2 = w2old - stepsize

    stepsize = w3 * learningrate
    w3 = w3old - stepsize

    stepsize = w4 * learningrate
    w4 = w4old - stepsize

    stepsize = w5 * learningrate
    w5 = w5old - stepsize

    stepsize = w6 * learningrate
    w6 = w6old - stepsize

    stepsize = w7 * learningrate
    w7 = w7old - stepsize

    stepsize = w8 * learningrate
    w8 = w8old - stepsize
    
    epoch += 1




####################################################################################
#Using the same inputs to see if your algorithm is correct, then graph at the end using the observe X values with the final
 #predicted Y values



# Weight1 multiplied by the values of the input array
array1 = w1 * np.array(arr) + bias
# Weight2 multiplied by the vales of the input array
array2 = w2 * np.array(arr) + bias


count = 0
while count < 5:
    # Node 1
    Node1[count] = 1 / (1 + math.exp(-array1[count]))
    
    # Node 2
    Node2[count] = 1 / (1 + math.exp(-array2[count]))
    count += 1

    
    
    
array3 = w3 * np.array(Node1) + w4 * np.array(Node2) + bias

array4 = w5 * np.array(Node1) + w6 * np.array(Node2) + bias

count = 0
while count < 5:
    # Node 1
    Node3[count] = 1 / (1 + math.exp(-array3[count]))
    
    # Node 2
    Node4[count] = 1 / (1 + math.exp(-array4[count]))
    count += 1





array5 = w7 * np.array(Node3) + w8 * np.array(Node4) + bias
count = 0
while count < 5:  
    # Node 5
    Node5[count] = 1 / (1 + math.exp(-array5[count]))
    count += 1
    
predicted = Node5
    

print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Xvalues')
plt.ylabel('Yvalues')
plt.scatter(arr,predicted)
plt.show()



# In[2]:


import pandas as pd
from PIL import Image
from numpy import asarray
import numpy as np
import math
import matplotlib.pyplot as plt
import os




######################################################
i = 0
X_values = [] * 500
while i <= 100: 
      i += 0.2
      X_values.append(i)
      


count = 0
Y_values = [] * 500
while count <= 499:
    Y_values.append(1/X_values[count])
    count += 1

print('\n')
print('\n')
plt.figure(figsize=(20,10))
plt.xlabel('X_values')
plt.ylabel('Y_values')
#plt.plot(X_values, Y_values)
plt.scatter(X_values, Y_values)
plt.show()

count = 0
Test_Xvalues = [] * 100
Test_Yvalues = [] * 100
Training_Xvalues = [] * 400
Training_Yvalues = [] * 400

while count <= 499:
      if count % 5 == 0:
            Test_Xvalues.append(X_values[count])
          
            Test_Yvalues.append(Y_values[count])
      else:
        Training_Xvalues.append(X_values[count])
       
        Training_Yvalues.append(Y_values[count])    
     
      count += 1
        
x1 = len(Test_Xvalues)
x2 = len(Test_Yvalues)
x3 = len(Training_Xvalues)
x4 = len(Training_Yvalues)
x5 = len(X_values)


count = 0
while count < 400:
    Training_Yvalues[count] = Training_Yvalues[count] / 100
    count += 1

count = 0
while count < 100:
    Test_Yvalues[count] = Test_Yvalues[count] / 100
    count += 1

print('\n')
print(x1)
print(x2)
print(x3)
print(x4)
print(x5)

print('\n')
print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Training_Xvalues')
plt.ylabel('Training_Yvalues')
plt.scatter(Training_Xvalues, Training_Yvalues)
plt.show()

print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Test_Xvalues')
plt.ylabel('Test_Yvalues')
plt.scatter(Test_Xvalues, Test_Yvalues)
plt.show()

#########################################################








#initializing random weights
w1 = 0.3
w2 = 0.4
w3 = 0.5
w4 = 0.2
w5 = 0.7
w6 = 0.6
w7 = 0.3
w8 = 0.1

#initialize the bias to be a constant 1
bias = 1



#this will save the X values at each node
array1 = [0 for i in range(400)] 
array2 = [0 for i in range(400)] 
array3 = [0 for i in range(400)] 
array4 = [0 for i in range(400)] 
array5 = [0 for i in range(400)] 



#this will save the Y values at each node
Node1 = [0 for i in range(400)] 
Node2 = [0 for i in range(400)] 
Node3 = [0 for i in range(400)] 
Node4 = [0 for i in range(400)] 
Node5 = [0 for i in range(400)] 



#intialize the learning rate
learningrate = 0.03


#initialize the actual Y value for the output
observe = Training_Yvalues


#initialize the actual X value for the input
arr = Training_Xvalues


#Graph the X and Y values 
#print('\n')
#plt.xlabel('Xvalues')
#plt.ylabel('Yvalues')
#plt.scatter(arr, observe)
#plt.show()



# Train your data
epoch = 0
while epoch < 10000:
    
    # Get the X-values at each node
    # Weight1 multiplied by the values of the input array
    array1 = w1 * np.array(arr) + bias
    
    # Weight2 multiplied by the vales of the input array
    array2 = w2 * np.array(arr) + bias
    
    #print(array1)
    
    
     # Get the Y-values at each node
    count = 0
    while count < 400:
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1[count] = 1 / (1 + math.exp(-(array1[count])))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2[count] = 1 / (1 + math.exp(-array2[count]))
        count += 1

    
    
    
    array3 = w3 * np.array(Node1) + w4 * np.array(Node2) + bias

    array4 = w5 * np.array(Node1) + w6 * np.array(Node2) + bias

    count = 0
    while count < 400:
        # Node 1
        Node3[count] = 1 / (1 + math.exp(-array3[count]))
    
        # Node 2
        Node4[count] = 1 / (1 + math.exp(-array4[count]))
        count += 1





    array5 = w7 * np.array(Node3) + w8 * np.array(Node4) + bias
    count = 0
    while count < 400:  
        # Node 5
        Node5[count] = 1 / (1 + math.exp(-array5[count]))
        count += 1
    # The predicted Y values are stores at node 5
    predicted = Node5
    
    

    
    
    

    # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
    w1old = w1
    w2old = w2
    w3old = w3
    w4old = w4
    w5old = w5
    w6old = w6
    w7old = w7
    w8old = w8    
    
    #gradient descent
    count = 0
    while count < 400:
        # Get the derivative for each node using it's correspoding X values
        s = 1/(1 + math.exp(-array1[count]))
        dernode1 = s*(1-s)
    
        s = 1/(1 + math.exp(-array2[count]))
        dernode2 = s*(1-s)
    
        s = 1/(1 + math.exp(-array3[count]))
        dernode3 = s*(1-s)
    
        s = 1/(1 + math.exp(-array4[count]))
        dernode4 = s*(1-s)
    
        s = 1/(1 + math.exp(-array5[count]))
        dernode5 = s*(1-s)
    
    #Adding the derivative for all the indexes using sum of sqaure Residuals
    #For the first iteration, replace all the weights at that index
        if count == 0:
            
            w1 = -2 * (observe[count] - predicted[count]) * arr[count] * dernode1 * w3 * dernode1 * w5 * dernode3 * w7 * dernode4 * w8 * dernode5

            w2 = -2 * (observe[count] - predicted[count]) * arr[count] * w4 * dernode2 * w6 * dernode2 * w7 * dernode3 * w8 * dernode4 * dernode5
    
    
            w3 = -2 * (observe[count] - predicted[count]) * Node1[count] * dernode3 * w7 * dernode5
    
            w4 = -2 * (observe[count] - predicted[count]) * Node2[count] * dernode3 * w7 * dernode5
    
            w5 = -2 * (observe[count] - predicted[count]) * Node1[count] * dernode4 * w8 * dernode5
    
            w6 = -2 * (observe[count] - predicted[count]) * Node2[count] * dernode4 * w8 * dernode5
    
    
            w7 = -2 * (observe[count] - predicted[count]) * Node3[count] * dernode5
    
            w8 = -2 * (observe[count] - predicted[count]) * Node4[count] * dernode5
        
       #For all the other iteration, add the weight to that index to get your final weights 
        else:
            

        
            w1 += -2 * (observe[count] - predicted[count]) * arr[count] * dernode1 * w3 * dernode1 * w5 * dernode3 * w7 * dernode4 * w8 * dernode5      

            w2 += -2 * (observe[count] - predicted[count]) * arr[count] * w4 * dernode2 * w6 * dernode2 * w7 * dernode3 * w8 * dernode4 * dernode5
    
    
            w3 += -2 * (observe[count] - predicted[count]) * Node1[count] * dernode3 * w7 * dernode5
    
            w4 += -2 * (observe[count] - predicted[count]) * Node2[count] * dernode3 * w7 * dernode5
    
            w5 += -2 * (observe[count] - predicted[count]) * Node1[count] * dernode4 * w8 * dernode5
    
            w6 += -2 * (observe[count] - predicted[count]) * Node2[count] * dernode4 * w8 * dernode5
    
    
            w7 += -2 * (observe[count] - predicted[count]) * Node3[count] * dernode5
    
            w8 += -2 * (observe[count] - predicted[count]) * Node4[count] * dernode5
        
        count += 1
    
    #Using gradient descent to find the where the slope equals to zero fast for each weight.

    stepsize = w1 * learningrate
    #print("print out learning rate")
    #print(learningrate)
    
    #print("print out the slope")
    #print(w1)
    
    #print("print step size")
    #print(stepsize)
    
    w1 = w1old - stepsize


    stepsize = w2 * learningrate
    w2 = w2old - stepsize

    stepsize = w3 * learningrate
    w3 = w3old - stepsize

    stepsize = w4 * learningrate
    w4 = w4old - stepsize

    stepsize = w5 * learningrate
    w5 = w5old - stepsize

    stepsize = w6 * learningrate
    w6 = w6old - stepsize

    stepsize = w7 * learningrate
    w7 = w7old - stepsize

    stepsize = w8 * learningrate
    w8 = w8old - stepsize
    
    epoch += 1




####################################################################################
#Using the same inputs to see if your algorithm is correct, then graph at the end using the observe X values with the final
 #predicted Y values

#this will save the X values at each node
array1 = [0 for i in range(100)] 
array2 = [0 for i in range(100)] 
array3 = [0 for i in range(100)] 
array4 = [0 for i in range(100)] 
array5 = [0 for i in range(100)] 



#this will save the Y values at each node
Node1 = [0 for i in range(100)] 
Node2 = [0 for i in range(100)] 
Node3 = [0 for i in range(100)] 
Node4 = [0 for i in range(100)] 
Node5 = [0 for i in range(100)] 


arr = Test_Xvalues
    
# Weight1 multiplied by the values of the input array
array1 = w1 * np.array(arr) + bias
# Weight2 multiplied by the vales of the input array
array2 = w2 * np.array(arr) + bias


count = 0
while count < 100:
    # Node 1
    Node1[count] = 1 / (1 + math.exp(-array1[count]))
    
    # Node 2
    Node2[count] = 1 / (1 + math.exp(-array2[count]))
    count += 1

    
    
    
array3 = w3 * np.array(Node1) + w4 * np.array(Node2) + bias

array4 = w5 * np.array(Node1) + w6 * np.array(Node2) + bias

count = 0
while count < 100:
    # Node 1
    Node3[count] = 1 / (1 + math.exp(-array3[count]))
    
    # Node 2
    Node4[count] = 1 / (1 + math.exp(-array4[count]))
    count += 1





array5 = w7 * np.array(Node3) + w8 * np.array(Node4) + bias
count = 0
while count < 100:  
    # Node 5
    Node5[count] = 1 / (1 + math.exp(-array5[count]))
    count += 1
    
predicted = Node5

print("print the length of prediction")
print(len(predicted))

print("Print the length of X values")
print(len(arr))

print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Xvalues')
plt.ylabel('Yvalues')
plt.scatter(arr,predicted)
plt.show()


# In[83]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import numpy as np
import sklearn
from sklearn.decomposition import PCA

#open iris dataset
iris = datasets.load_iris()
#print(iris)

#print(len(iris))

#X = iris.data[:, :2]  # we only take the first two features.
#print(X)
#print(len(X))




array = iris.data[:, :4]



#print(len(array))

print(array)







#Setosa
group1 = [[0 for i in range(4)] for j in range(50)]

#Vesicolour
group2 = [[0 for i in range(4)] for j in range(50)]

#Virginia
group3 = [[0 for i in range(4)] for j in range(50)]

#Find the maximum value of each column
col0max = 0
col1max = 0
col2max = 0
col3max = 0
row = 0
col = 0
while row < 150:
    col = 0
    while col < 4:
        if col == 0:
            if col0max < array[row][col]:
                col0max = array[row][col]
        if col == 1:
            if col1max < array[row][col]:
                col1max = array[row][col]
        if col == 2:
            if col2max < array[row][col]:
                col2max = array[row][col]        
        if col == 3:
            if col3max < array[row][col]:
                col3max = array[row][col]
        col += 1
    row += 1

    
print("print maximum value at column 0")
print(col0max)

print("print maximum value at column 1")
print(col1max)




row = 0
col = 0
while row < 150:
    col = 0
    while col < 4:
        if col == 0:
            
            array[row][col] = (array[row][col])/col0max
            
        if col == 1:
            
            array[row][col] = (array[row][col])/col1max
            
        if col == 2:
            
            array[row][col] = (array[row][col])/col2max 
            
        if col == 3:
            
            array[row][col] = (array[row][col])/col3max
            
        col += 1
    row += 1

print("print the normalized matrix")
print(array)



# Seperate all the different classes

row1 = 0
row2 = 0

col = 0
row = 0
while row < 150:
    col = 0
    while col < 4:
        if row < 50:
            group1[row][col] = array[row][col]
        
        elif 50 <= row and row < 100:
            group2[row1][col] = array[row][col]
           
        else:
            group3[row2][col] = array[row][col]
        col += 1
    
    if row < 50:
        row1 = 0
    else:
        row1 += 1
    
    if row < 100:
        row2 = 0
    else:
        row2 += 1
    
    row += 1
    
    
    
#print("print out group1")
#print(group1)
    
#print("print out group2")
#print(group2)

#print("print out group3")
#print(group3)





# The shuffle of group1, Setosa
print("print out again group1")
print(group1)
np.random.shuffle(group1)

#The shuffle of group2, Vesicolour
np.random.shuffle(group2)
np.random.shuffle(group3)
#print(group1)

#The shuffle of group3, Virginia
print('\n')
print("print group 1 shuffle")
print(group1)


#initialize the array of targeted value
targetvalues = [0 for i in range(150)]


#initialize the groups which will be use for training and testing
mixgroup1 = group1
mixgroup2 = group2
mixgroup3 = group3



combine = [[0 for i in range(4)] for j in range(150)]


#combine the mix matrix for each group
col = 0
row = 0
while row < 150:
    col = 0
    while col < 4:
        if row < 50:
            combine[row][col] = group1[row][col]
        
        elif 50 <= row and row < 100:
            combine[row][col] = group2[row1][col]
           
        else:
            combine[row][col] = group3[row2][col]
        col += 1
    
    if row < 50:
        row1 = 0
    else:
        row1 += 1
    
    if row < 100:
        row2 = 0
    else:
        row2 += 1
    
    row += 1

    
print('\n')
print("print combine")
print(combine)  










# The entire matrix is now mix and alternating between the classes
combinemix = [[0 for i in range(4)] for j in range(150)]


row1 = 50
row2 = 100
row3 = 0

count = 0
resetcount = 1
reset = 1
col = 0
row = 0
while row < 150:
    reset = resetcount % 3
    
    if reset == 1:
        targetvalues[count] = 0
        col = 0
        while col < 4:
            combinemix[row][col] = combine[row3][col]
            col += 1
        row3 += 1
    elif reset == 2:
        targetvalues[count] = 1
        col = 0
        while col < 4:
            combinemix[row][col] = combine[row1][col]
            col += 1
        row1 += 1
    else:
        targetvalues[count] = 2
        col = 0
        while col < 4:
            combinemix[row][col] = combine[row2][col]
            col += 1
        row2 += 1
    
    if reset == 0:
        resetcount = 1
    else:
        resetcount += 1
    count += 1
    row += 1

print('\n')
print("print the combine matrix")    
print(combinemix)
    
    

    
target1 = [0 for i in range(50)]
target2 = [0 for i in range(50)]
target3 = [0 for i in range(50)]

    
print("print the values of the target array")
print(targetvalues)


# Seperate the mixgroups respective targeted values
count1 = 0
count2 = 0
count = 0    
while count < 150:
    if count < 50:
        target1[count] = targetvalues[count]
    elif count < 100:
        target2[count1] = targetvalues[count]
        count1 += 1
    else:
        target3[count2] = targetvalues[count]
        count2 += 1
    count += 1
    
print("print target1 values")  
print(target1)

print("print target2 values")
print(target2)

print("print target3 values")
print(target3)    

#normalize the targeted values

count = 0
while count < 50:
    target1[count] = target1[count] / 2
    target2[count] = target2[count] / 2
    target3[count] = target3[count] / 2
    count += 1
# Now save the combine matrix into their respective groups
row1 = 0
row2 = 0

col = 0
row = 0
while row < 150:
    col = 0
    while col < 4:
        if row < 50:
            mixgroup1[row][col] = combinemix[row][col]
        
        elif 50 <= row and row < 100:
            mixgroup2[row1][col] = combinemix[row][col]
           
        else:
            mixgroup3[row2][col] = combinemix[row][col]
        col += 1
    
    if row < 50:
        row1 = 0
    else:
        row1 += 1
    
    if row < 100:
        row2 = 0
    else:
        row2 += 1
    
    row += 1   

print('\n')
print("print out the mix group 1")
print(mixgroup1)


sett = [0 for i in range(50)]

count = 0
while count < 50:
    sett[count] = count
    count += 1

print("print sections testing, which will be later use to compare the predicted values")

print("print group 3, section 1")

plt.xlabel('input')
plt.ylabel('grp 3 pred')
plt.scatter(sett, target3)
plt.show()
     
    
     


print("print group 1, section 2")

plt.xlabel('input')
plt.ylabel('grp 1 pred')
plt.scatter(sett, target1)
plt.show()


print("print group 2, section 3")

plt.xlabel('input')
plt.ylabel('grp 2 pred')
plt.scatter(sett, target2)
plt.show()



#################################################################################

wrong = 0

#initializing random weights
w1 = 0.3
w2 = 0.4
w3 = 0.5
w4 = 0.2
w5 = 0.7
w6 = 0.6
w7 = 0.3
w8 = 0.1
w9 = 0.61
w10 = 0.24
w11 = 0.9
w12 = 0.11
w13 = 0.41
w14 = 0.32

#initialize the bias to be a constant 1
bias = 1

#intitialize the weights of the bias
bw1 = 0.2
bw2 = 0.5
bw3 = 0.7
bw4 = 0.3
bw5 = 0.4



#this will save the X values at each node
array1 = 1
array2 = 1
array3 = 1
array4 = 1
array5 = 1



#this will save the Y values at each node
Node1 = 1
Node2 = 1
Node3 = 1
Node4 = 1
Node5 = 1



#intialize the learning rate
learningrate = 0.01


#initialize the actual Y value for the output
observe = 1

# I know after you have to regularize the data of observe


#initialize the actual X value for the input/ you don't regularize the X input
arr1 = 1
arr2 = 1
arr3 = 1
arr4 = 1

#Graph the X and Y values 
#print('\n')
#plt.xlabel('Xvalues')
#plt.ylabel('Yvalues')
#plt.scatter(inp, ob)
#plt.show()

er= [0 for i in range(50)]

bob= [0 for i in range(50)]

count = 0
while count < 50:
    bob[count] = count
    count += 1

    



index = 0
# Train your data
epoch = 0
while epoch < 200:
    mixgroup1, target1  = sklearn.utils.shuffle(mixgroup1, target1)
    row = 0
    
    while row < 50:
        
        observe = target1[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup1[row][col]
            if col == 1:
                arr2 = mixgroup1[row][col]
            if col == 2:
                arr3 = mixgroup1[row][col]
            if col == 3:
                arr4 = mixgroup1[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        #print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
        w9old = w9
        w10old = w10
        w11old = w11
        w12old = w12
        w13old = w13
        w14old = w14
       
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * error
        w13  = learningrate * L5 * Node3
        w14  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w13)
        bw3 = learningrate * L3 * bias
        w9  = learningrate * L3 * Node1
        w10  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w14)
        bw4 = learningrate * L4 * bias
        w11  = learningrate * L4 * Node1
        w12  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * (L3 * w9) * (L4 * w11)
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr1 
        w2  = learningrate * L1 * arr2 
        w3  = learningrate * L1 * arr3 
        w4  = learningrate * L1 * arr4 
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * (L3 * w10) * (L4 * w12)
        bw2 = learningrate * L2 * bias
        w5  = learningrate * L2 * arr1
        w6  = learningrate * L2 * arr2
        w7  = learningrate * L2 * arr3
        w8  = learningrate * L2 * arr4
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
        w9 = w9 + w9old
        w10 = w10 + w10old
        w11 = w11 + w11old
        w12 = w12 + w12old
        w13 = w13 + w13old
        w14 = w14 + w14old
       
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        row += 1
 
    
   
    epoch += 1

index = 0
# Train your data
epoch = 0
while epoch < 200:
    mixgroup2, target2  = sklearn.utils.shuffle(mixgroup2, target2)
    row = 0

    while row < 50:
        
        observe = target2[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup2[row][col]
            if col == 1:
                arr2 = mixgroup2[row][col]
            if col == 2:
                arr3 = mixgroup2[row][col]
            if col == 3:
                arr4 = mixgroup2[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        #print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
        w9old = w9
        w10old = w10
        w11old = w11
        w12old = w12
        w13old = w13
        w14old = w14
       
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * error
        w13  = learningrate * L5 * Node3
        w14  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w13)
        bw3 = learningrate * L3 * bias
        w9  = learningrate * L3 * Node1
        w10  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w14)
        bw4 = learningrate * L4 * bias
        w11  = learningrate * L4 * Node1
        w12  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * (L3 * w9) * (L4 * w11)
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr1 
        w2  = learningrate * L1 * arr2 
        w3  = learningrate * L1 * arr3 
        w4  = learningrate * L1 * arr4 
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * (L3 * w10) * (L4 * w12)
        bw2 = learningrate * L2 * bias
        w5  = learningrate * L2 * arr1
        w6  = learningrate * L2 * arr2
        w7  = learningrate * L2 * arr3
        w8  = learningrate * L2 * arr4
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
        w9 = w9 + w9old
        w10 = w10 + w10old
        w11 = w11 + w11old
        w12 = w12 + w12old
        w13 = w13 + w13old
        w14 = w14 + w14old
       
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        row += 1
 
    
   
    epoch += 1

print("The error for the first section of testing")
print("The first section of testing use the first and second group for training, and the third group for testing")
row = 0
while row < 50:
        
        observe = target3[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup3[row][col]
            if col == 1:
                arr2 = mixgroup3[row][col]
            if col == 2:
                arr3 = mixgroup3[row][col]
            if col == 3:
                arr4 = mixgroup3[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
        
        er[row] = error
        print(error)
        
            
        row += 1

        
plt.xlabel('error')
plt.ylabel('Tested values')
plt.scatter(bob,er)
plt.show()
#plt.scatter(Test_Xvalues, Test_Yvalues)       
        
print("The error for the second section of testing")

print("In this section, will use the second and third for training and use the first group for testing")
index = 0
# Train your data
epoch = 0
while epoch < 200:
    mixgroup2, target2  = sklearn.utils.shuffle(mixgroup2, target2)
    row = 0
    
    while row < 50:
        
        observe = target2[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup2[row][col]
            if col == 1:
                arr2 = mixgroup2[row][col]
            if col == 2:
                arr3 = mixgroup2[row][col]
            if col == 3:
                arr4 = mixgroup2[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        #print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
        w9old = w9
        w10old = w10
        w11old = w11
        w12old = w12
        w13old = w13
        w14old = w14
       
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * error
        w13  = learningrate * L5 * Node3
        w14  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w13)
        bw3 = learningrate * L3 * bias
        w9  = learningrate * L3 * Node1
        w10  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w14)
        bw4 = learningrate * L4 * bias
        w11  = learningrate * L4 * Node1
        w12  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * (L3 * w9) * (L4 * w11)
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr1 
        w2  = learningrate * L1 * arr2 
        w3  = learningrate * L1 * arr3 
        w4  = learningrate * L1 * arr4 
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * (L3 * w10) * (L4 * w12)
        bw2 = learningrate * L2 * bias
        w5  = learningrate * L2 * arr1
        w6  = learningrate * L2 * arr2
        w7  = learningrate * L2 * arr3
        w8  = learningrate * L2 * arr4
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
        w9 = w9 + w9old
        w10 = w10 + w10old
        w11 = w11 + w11old
        w12 = w12 + w12old
        w13 = w13 + w13old
        w14 = w14 + w14old
       
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        row += 1
 
    
   
    epoch += 1
    
    

    
    
    
index = 0
# Train your data
epoch = 0
while epoch < 200:
    mixgroup3, target3  = sklearn.utils.shuffle(mixgroup3, target3)
    row = 0
    
    while row < 50:
        
        observe = target3[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup3[row][col]
            if col == 1:
                arr2 = mixgroup3[row][col]
            if col == 2:
                arr3 = mixgroup3[row][col]
            if col == 3:
                arr4 = mixgroup3[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        #print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
        w9old = w9
        w10old = w10
        w11old = w11
        w12old = w12
        w13old = w13
        w14old = w14
       
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * error
        w13  = learningrate * L5 * Node3
        w14  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w13)
        bw3 = learningrate * L3 * bias
        w9  = learningrate * L3 * Node1
        w10  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w14)
        bw4 = learningrate * L4 * bias
        w11  = learningrate * L4 * Node1
        w12  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * (L3 * w9) * (L4 * w11)
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr1 
        w2  = learningrate * L1 * arr2 
        w3  = learningrate * L1 * arr3 
        w4  = learningrate * L1 * arr4 
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * (L3 * w10) * (L4 * w12)
        bw2 = learningrate * L2 * bias
        w5  = learningrate * L2 * arr1
        w6  = learningrate * L2 * arr2
        w7  = learningrate * L2 * arr3
        w8  = learningrate * L2 * arr4
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
        w9 = w9 + w9old
        w10 = w10 + w10old
        w11 = w11 + w11old
        w12 = w12 + w12old
        w13 = w13 + w13old
        w14 = w14 + w14old
       
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        row += 1
 
    
   
    epoch += 1
        

row = 0
while row < 50:
        
        observe = target1[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup1[row][col]
            if col == 1:
                arr2 = mixgroup1[row][col]
            if col == 2:
                arr3 = mixgroup1[row][col]
            if col == 3:
                arr4 = mixgroup1[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
        
        er[row] = error
        print(error)
        
            
        row += 1    

plt.xlabel('error')
plt.ylabel('Tested values')
plt.scatter(bob, er)
plt.show()
#plt.scatter(Test_Xvalues, Test_Yvalues)


print("The third section uses the first and the third group for training and the second group for testing")
print("print the error for the third section")
index = 0
# Train your data
epoch = 0
while epoch < 200:
    mixgroup3, target3  = sklearn.utils.shuffle(mixgroup3, target3)
    row = 0
    
    while row < 50:
        
        observe = target3[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup3[row][col]
            if col == 1:
                arr2 = mixgroup3[row][col]
            if col == 2:
                arr3 = mixgroup3[row][col]
            if col == 3:
                arr4 = mixgroup3[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        #print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
        w9old = w9
        w10old = w10
        w11old = w11
        w12old = w12
        w13old = w13
        w14old = w14
       
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * error
        w13  = learningrate * L5 * Node3
        w14  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w13)
        bw3 = learningrate * L3 * bias
        w9  = learningrate * L3 * Node1
        w10  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w14)
        bw4 = learningrate * L4 * bias
        w11  = learningrate * L4 * Node1
        w12  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * (L3 * w9) * (L4 * w11)
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr1 
        w2  = learningrate * L1 * arr2 
        w3  = learningrate * L1 * arr3 
        w4  = learningrate * L1 * arr4 
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * (L3 * w10) * (L4 * w12)
        bw2 = learningrate * L2 * bias
        w5  = learningrate * L2 * arr1
        w6  = learningrate * L2 * arr2
        w7  = learningrate * L2 * arr3
        w8  = learningrate * L2 * arr4
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
        w9 = w9 + w9old
        w10 = w10 + w10old
        w11 = w11 + w11old
        w12 = w12 + w12old
        w13 = w13 + w13old
        w14 = w14 + w14old
       
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        row += 1
 
    
   
    epoch += 1      
      
index = 0
# Train your data
epoch = 0
while epoch < 200:
    mixgroup1, target1  = sklearn.utils.shuffle(mixgroup1, target1)
    row = 0
    
    while row < 50:
        
        observe = target1[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup1[row][col]
            if col == 1:
                arr2 = mixgroup1[row][col]
            if col == 2:
                arr3 = mixgroup1[row][col]
            if col == 3:
                arr4 = mixgroup1[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        #print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
        w9old = w9
        w10old = w10
        w11old = w11
        w12old = w12
        w13old = w13
        w14old = w14
       
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * error
        w13  = learningrate * L5 * Node3
        w14  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w13)
        bw3 = learningrate * L3 * bias
        w9  = learningrate * L3 * Node1
        w10  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w14)
        bw4 = learningrate * L4 * bias
        w11  = learningrate * L4 * Node1
        w12  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * (L3 * w9) * (L4 * w11)
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr1 
        w2  = learningrate * L1 * arr2 
        w3  = learningrate * L1 * arr3 
        w4  = learningrate * L1 * arr4 
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * (L3 * w10) * (L4 * w12)
        bw2 = learningrate * L2 * bias
        w5  = learningrate * L2 * arr1
        w6  = learningrate * L2 * arr2
        w7  = learningrate * L2 * arr3
        w8  = learningrate * L2 * arr4
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
        w9 = w9 + w9old
        w10 = w10 + w10old
        w11 = w11 + w11old
        w12 = w12 + w12old
        w13 = w13 + w13old
        w14 = w14 + w14old
       
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        row += 1
 
    
   
    epoch += 1      

      

      
row = 0
while row < 50:
        
        observe = target1[row]
        col = 0
        
        while col < 4:
            
            if col == 0:
                arr1 = mixgroup1[row][col]
            if col == 1:
                arr2 = mixgroup1[row][col]
            if col == 2:
                arr3 = mixgroup1[row][col]
            if col == 3:
                arr4 = mixgroup1[row][col]
                
            col += 1
        

    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr1 + w2 * arr2 + w3 * arr3 + w4 * arr4 + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w5 * arr1 + w6 * arr2 + w7 * arr3 + w8 * arr4 + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w9 * Node1 + w10 * Node2 + bias * bw3

        array4 = w11 * Node1 + w12 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w13 * Node3 + w14 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
        
    
        print(error)
        er[row] = error
            
        row += 1    

plt.xlabel('error')
plt.ylabel('Tested values')
plt.scatter(bob, er)
plt.show()
#plt.scatter(Test_Xvalues, Test_Yvalues)            

#array [samples: setosa, vesicolour, and Virginia] [ Sepal length, sepal width, petal length, petal width]
#print(array)

#So should I begin by seperating each sample(row) from the stacks of continous rows
#Before that, how do I find the target values first for each row

#y = iris.target

















# In[8]:


import pandas as pd
from PIL import Image
from numpy import asarray
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sklearn

graph1 = [0, 0, 0, 0, 0]


#initialize the actual Y value for the output
ob = [1, 2, 5, 8, 11]

# get the maximum value of the array
max_value = np.max(ob)

#normalize the data
count = 0
while count < len(ob):
    ob[count] = ob[count]/max_value
    count += 1
    
    


print("print out normalize observe")
print(ob)
#initialize the actual X value for the input
inp = [1, 2, 3, 4, 5]

#normalize the data
count = 0
while count < len(ob):
    inp[count] = inp[count]/max_value
    count += 1
    
    
#initializing random weights
w1 = 0.3
w2 = 0.4
w3 = 0.5
w4 = 0.2
w5 = 0.7
w6 = 0.6
w7 = 0.3
w8 = 0.1

#initialize the bias to be a constant 1
bias = 1

#intitialize the weights of the bias
bw1 = 0.2
bw2 = 0.5
bw3 = 0.7
bw4 = 0.3
bw5 = 0.4



#this will save the X values at each node
array1 = 1
array2 = 1
array3 = 1
array4 = 1
array5 = 1



#this will save the Y values at each node
Node1 = 1
Node2 = 1
Node3 = 1
Node4 = 1
Node5 = 1



#intialize the learning rate
learningrate = 0.03


#initialize the actual Y value for the output
observe = 1

# I know after you have to regularize the data of observe


#initialize the actual X value for the input/ you don't regularize the X input
arr = 1

#Graph the X and Y values 
print('\n')
plt.xlabel('Xvalues')
plt.ylabel('Yvalues')
plt.scatter(inp, ob)
plt.show()


index = 0
# Train your data
epoch = 0
while epoch < 200:
    inp, ob = sklearn.utils.shuffle(inp, ob)
    index = 0
    print(inp)
    print(ob)
    while index < len(inp):
        arr = inp[index]
        observe = ob[index]
    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w2 * arr + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w3 * Node1 + w4 * Node2 + bias * bw3

        array4 = w5 * Node1 + w6 * Node2 + bias * bw4

        #count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w7 * Node3 + w8 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
    
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * bias
        w7  = learningrate * L5 * Node3
        w8  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w7)
        bw3 = learningrate * L3 * bias
        w3  = learningrate * L3 * Node1
        w4  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w8)
        bw4 = learningrate * L4 * bias
        w5  = learningrate * L4 * Node1
        w6  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * ((L3 * w3) + (L4 * w5))
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr
    
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * ((L3 * w4) + (L4 * w6))
        bw2 = learningrate * L2 * bias
        w2  = learningrate * L2 * arr
     
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        index += 1
 
    
   
    epoch += 1

    
print("print out the length")
print(len(inp))    

    
index = 0
while index < len(inp):
    
    arr = inp[index]
    
    # Weight1 multiplied by the values of the input array
    array1 = w1 * arr + bias * bw1
    
    # Weight2 multiplied by the vales of the input array
    array2 = w2 * arr + bias * bw2
    
    #print(array1)
    


   
    # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
    Node1 = 1 / (1 + math.exp(-(array1)))
 
    # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
    Node2 = 1 / (1 + math.exp(-array2))


    
    
    
    array3 = w3 * Node1 + w4 * Node2 + bias * bw3

    array4 = w5 * Node1 + w6 * Node2 + bias * bw4

    count = 0
    
    # function 
    # Node 3 and 4
    Node3 = 1 / (1 + math.exp(-array3))
    
    # Node 2
    Node4 = 1 / (1 + math.exp(-array4))
    





    array5 = w7 * Node3 + w8 * Node4 + bias * bw5
 
    # Node 5
    Node5 = 1 / (1 + math.exp(-array5))
    # The predicted Y values are stores at node 5
    
    graph1[index] = Node5
    index += 1

    
#count = 0
#while count < len(graph1):
#    graph1[count] = graph1[count] * max_value
#    count += 1

plt.xlabel('TrainXvalues')
plt.ylabel('TrainYvalues')
plt.scatter(inp, graph1)


# In[24]:


import pandas as pd
from PIL import Image
from numpy import asarray
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sklearn


inp = [1, 2, 3, 4, 5]
ob = [1, 2, 5, 8, 11]



inp, ob = sklearn.utils.shuffle(inp, ob)
#inp, ob = np.random.shuffle(inp, ob)
print(inp)
print(ob)


# In[67]:




import pandas as pd
from PIL import Image
from numpy import asarray
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sklearn

graph1 = [0 for i in range(100)]

######################################################
i = 0
X_values = [] * 500
while i <= 100: 
      i += 0.2
      X_values.append(i)
      


count = 0
Y_values = [] * 500
while count <= 499:
    Y_values.append(1/X_values[count])
    count += 1

print('\n')
print('\n')
plt.figure(figsize=(20,10))
plt.xlabel('X_values')
plt.ylabel('Y_values')
#plt.plot(X_values, Y_values)
plt.scatter(X_values, Y_values)
plt.show()

count = 0
Test_Xvalues = [] * 100
Test_Yvalues = [] * 100
Training_Xvalues = [] * 400
Training_Yvalues = [] * 400

while count <= 499:
      if count % 5 == 0:
            Test_Xvalues.append(X_values[count])
          
            Test_Yvalues.append(Y_values[count])
      else:
        Training_Xvalues.append(X_values[count])
       
        Training_Yvalues.append(Y_values[count])    
     
      count += 1

        
        
print("length of input testing")
x1 = len(Test_Xvalues)

print("length of output testing")
x2 = len(Test_Yvalues)

print("length of input training")
x3 = len(Training_Xvalues)

print("length of output training")
x4 = len(Training_Yvalues)

print("total length of training and testing")
x5 = len(X_values)



#normalizing the dataset
count = 0
while count < 400:
    Training_Yvalues[count] = Training_Yvalues[count] / 100
    count += 1

count = 0
while count < 100:
    Test_Yvalues[count] = Test_Yvalues[count] / 100
    count += 1

print('\n')
print(x1)
print(x2)
print(x3)
print(x4)
print(x5)

print('\n')
print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Training_Xvalues')
plt.ylabel('Training_Yvalues')
plt.scatter(Training_Xvalues, Training_Yvalues)
plt.show()

print('\n')
#plt.figure(figsize=(20,10))
plt.xlabel('Test_Xvalues')
plt.ylabel('Test_Yvalues')
plt.scatter(Test_Xvalues, Test_Yvalues)
plt.show()

#########################################################







#initializing random weights
w1 = 0.3
w2 = 0.4
w3 = 0.5
w4 = 0.2
w5 = 0.7
w6 = 0.6
w7 = 0.3
w8 = 0.1

#initialize the bias to be a constant 1
bias = 1

#intitialize the weights of the bias
bw1 = 0.2
bw2 = 0.5
bw3 = 0.7
bw4 = 0.3
bw5 = 0.4



#this will save the X values at each node
array1 = 1
array2 = 1
array3 = 1
array4 = 1
array5 = 1



#this will save the Y values at each node
Node1 = 1
Node2 = 1
Node3 = 1
Node4 = 1
Node5 = 1



#intialize the learning rate
learningrate = 0.1


#initialize the actual Y value for the output
observe = 1

# I know after you have to regularize the data of observe


#initialize the actual X value for the input/ you don't regularize the X input
arr = 1

#Graph the X and Y values 
print('\n')
plt.xlabel('Xvalues')
plt.ylabel('Yvalues')
plt.scatter(inp, ob)
plt.show()


index = 0
# Train your data
epoch = 0
while epoch < 200:
    Training_Xvalues, Training_Yvalues = sklearn.utils.shuffle(Training_Xvalues, Training_Yvalues)
    index = 0

    while index < len(Training_Xvalues):
        arr = Training_Xvalues[index]
        observe = Training_Yvalues[index]
    

        # Weight1 multiplied by the values of the input array
        array1 = w1 * arr + bias * bw1
    
        # Weight2 multiplied by the vales of the input array
        array2 = w2 * arr + bias * bw2
    
        #print(array1)
    


   
        # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
        Node1 = 1 / (1 + math.exp(-(array1)))
 
        # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
        Node2 = 1 / (1 + math.exp(-array2))


    
    
    
        array3 = w3 * Node1 + w4 * Node2 + bias * bw3

        array4 = w5 * Node1 + w6 * Node2 + bias * bw4

        count = 0
    
        # function 
        # Node 3 and 4
        Node3 = 1 / (1 + math.exp(-array3))
    
        # Node 2
        Node4 = 1 / (1 + math.exp(-array4))
    





        array5 = w7 * Node3 + w8 * Node4 + bias * bw5
 
        # Node 5
        Node5 = 1 / (1 + math.exp(-array5))
        # The predicted Y values are stores at node 5
        predicted = Node5
    
    
        error = observe - predicted 
    
        print(error)
   
    

        # intiliazing a storage for each weight, which you will use to obtain the new weight in gradient descent
        w1old = w1
        w2old = w2
        w3old = w3
        w4old = w4
        w5old = w5
        w6old = w6
        w7old = w7
        w8old = w8    
    
    
        oldbw1 = bw1
        oldbw2 = bw2
        oldbw3 = bw3
        oldbw4 = bw4
        oldbw5 = bw5
    
    
    
    
        #gradient descent
    
        # Get the derivative for each node using it's correspoding X values
        s   = 1/(1 + math.exp(-array5))
        dernode5 = s*(1-s)
        L5  = dernode5 * error
        bw5 = learningrate * L5 * error
        w7  = learningrate * L5 * Node3
        w8  = learningrate * L5 * Node4
    
    
        s   = 1/(1 + math.exp(-array3))
        dernode3 = s*(1-s)
        L3  = dernode3 * ( L5 * w7)
        bw3 = learningrate * L3 * bias
        w3  = learningrate * L3 * Node1
        w4  = learningrate * L3 * Node2
    
    
        s   = 1/(1 + math.exp(-array4))
        dernode4 = s*(1-s)
        L4  = dernode4 * (L5 * w8)
        bw4 = learningrate * L4 * bias
        w5  = learningrate * L4 * Node1
        w6  = learningrate * L4 * Node2
    
    
        s   = 1/(1 + math.exp(-array1))
        dernode1 = s*(1-s)
        L1  = dernode1 * (L3 * w3) * (L4 * w5)
        bw1 = learningrate * L1 * bias
        w1  = learningrate * L1 * arr
    
    
        s   = 1/(1 + math.exp(-array2))
        dernode2 = s*(1-s)
        L2  = dernode2 * (L3 * w4) * (L4 * w6)
        bw2 = learningrate * L2 * bias
        w2  = learningrate * L2 * arr
     
    
        #Using gradient descent to find the where the slope equals to zero fast for each weight.
        w1 = w1 + w1old
        w2 = w2 + w2old
        w3 = w3 + w3old
        w4 = w4 + w4old
        w5 = w5 + w5old
        w6 = w6 + w6old
        w7 = w7 + w7old
        w8 = w8 + w8old
    
    
        bw1 = bw1 + oldbw1
        bw2 = bw2 + oldbw2
        bw3 = bw3 + oldbw3
        bw4 = bw4 + oldbw4
        bw5 = bw5 + oldbw5
        
        index += 1
 
    
   
    epoch += 1

       
print("print Test_values length")
print(len(Test_Xvalues))

print("print graph1 length")
print(len(graph1))
    
index = 0
while index < len(Test_Xvalues):
    
    arr = Test_Xvalues[index]
    
    # Weight1 multiplied by the values of the input array
    array1 = w1 * arr + bias * bw1
    
    # Weight2 multiplied by the vales of the input array
    array2 = w2 * arr + bias * bw2
    
    #print(array1)
    


   
    # Node 1: takes in the input X and passes it through the sigmoid function then stores it at the corresponding Y
    Node1 = 1 / (1 + math.exp(-(array1)))
 
    # Node 2: takes in the input X and passes it through the sigmoid function then stores it at the correspodning X
    Node2 = 1 / (1 + math.exp(-array2))


    
    
    
    array3 = w3 * Node1 + w4 * Node2 + bias * bw3

    array4 = w5 * Node1 + w6 * Node2 + bias * bw4

    count = 0
    
    # function 
    # Node 3 and 4
    Node3 = 1 / (1 + math.exp(-array3))
    
    # Node 2
    Node4 = 1 / (1 + math.exp(-array4))
    





    array5 = w7 * Node3 + w8 * Node4 + bias * bw5
 
    # Node 5
    Node5 = 1 / (1 + math.exp(-array5))
    # The predicted Y values are stores at node 5
    
    graph1[index] = Node5
    index += 1

    
#count = 0
#while count < len(graph1):
#    graph1[count] = graph1[count] * 100
#    count += 1

plt.xlabel('TestedXvalues')
plt.ylabel('TestedYvalues')
plt.scatter(Test_Xvalues, graph1)
#plt.scatter(Test_Xvalues, Test_Yvalues)



# In[ ]:




