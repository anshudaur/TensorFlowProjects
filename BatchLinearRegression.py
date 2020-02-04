"""Batch linear regression
@author: anshu daur , matriculation ID - 223853
"""
import numpy
import argparse
#Code for passing/accepting Command Line arguments
parser = argparse.ArgumentParser(description='Linear Regression')
parser.add_argument("--fileName", type= str )
parser.add_argument("--learningRate", type=float)
parser.add_argument("--threshold", type=float)
args = parser.parse_args()
fileName = args.fileName

learning_rate = 0.0001
iterations =  0
csv = numpy.genfromtxt(fileName, dtype=float, delimiter=",")
input = numpy.array(csv,float)
column_count = input.shape[1]
row_count = input.shape[0]
grad_array = numpy.ndarray(shape=(column_count),dtype=float)
weight_array = numpy.ndarray(shape=(column_count),dtype=float)
cost=cost_old=0
threshold = 0.0001
threshold_flag = True

for k in range(column_count):
    weight_array[k]=0 
    
while (threshold_flag) :  
        cost = 0
        for i in range(row_count):
            Y_new =0                              
            for k in range(column_count):
               if k ==0:
                    Y_new = weight_array[k]
               else: 
                    Y_new = Y_new + weight_array[k]*input[i][k-1]
            error = input[i][column_count-1]-Y_new
            cost = float(cost + error*error)
            for k in range(column_count):
                if i==0:
                    grad_array[k]=0 
                else :
                    if k==0 : 
                        grad_array[k]= grad_array[k]+error
                    else:
                        grad_array[k] = grad_array[k]+input[i][k-1]*error 
                        
        print(iterations,*weight_array[0:column_count],cost,sep=",") 
        if (abs(cost-cost_old)<=threshold):
            threshold_flag = False
        cost_old = cost
        
        for k in range(column_count):
            weight_array[k]=float(weight_array[k]+learning_rate*grad_array[k])  
        iterations=iterations+1
        