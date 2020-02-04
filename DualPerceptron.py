# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:55:41 2018

@author: anshu
"""
import csv
import argparse

class anIterator(object):
    def __init__(self, l):
        self.data = l

    def __iter__(self):
        return iter(self.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perceptron')
    parser.add_argument("--InputFileName", type= str )
    parser.add_argument("--OutputFileName", type=str)
    args = parser.parse_args()
    fileName = args.InputFileName
    outputfileName = args.OutputFileName
    #fileName = "Gauss.tsv"
    iteration = 100
    TotalColumn = 0
    Data = []
    
    with open(fileName, 'r') as csvFile:  
        fileReader = csv.reader(csvFile,delimiter='\t')
        for row in fileReader:
            if (row.count('') > 0):
                row.remove('')
            Data.append(row)
            TotalColumn = len(row)
        
    #print(TotalColumn)
    
    weightArr = [0]*TotalColumn
    weight_AnnealingETA_arr = [0]*TotalColumn
    gradArray = [0]*TotalColumn
    grad_AnnealingETA_arr = [0]*TotalColumn
    Y_NEW = [0]*len(Data)
    Y_Ann_NEW = [0]*len(Data)
    IterationArr = []
    IterationETAArr = []
    ETA_A = 1
    for i in range(iteration+1): 
        ETA_A = 1/(i+1)
                
        for row in range(len(Data)):
            const_output = 0
            ann_output = 0
            for col in range(TotalColumn):
                gradArray[col] = 0
                grad_AnnealingETA_arr[col] = 0
                if col ==0: #0th column is output
                    const_output = weightArr[col]
                    ann_output = weight_AnnealingETA_arr[col]
                else:
                    #print(Data[row][col])
                    const_output = const_output+ weightArr[col] * float(Data[row][col])
                    ann_output =ann_output+weight_AnnealingETA_arr[col]* float(Data[row][col])
            if const_output>0 :
                Y_NEW[row] = 1
            else:
                Y_NEW[row] = 0
                
            if ann_output >0 :
                Y_Ann_NEW[row] = 1
            else:
                Y_Ann_NEW[row] = 0
        misclassified_count = 0  
        misclassified_ann_count = 0  
            
        for row in range(len(Data)):
            output_Val = Data[row][0]
            actualOutput = 0
            if output_Val == "A" :
                actualOutput = 1
                
            if Y_NEW[row] != actualOutput:
                misclassified_count = misclassified_count+1
            if Y_Ann_NEW[row] != actualOutput:
                misclassified_ann_count = misclassified_ann_count+1
           
            for col in range(TotalColumn): 
                if col == 0:
                    gradArray[col] = gradArray[col]+actualOutput-Y_NEW[row]         
                    grad_AnnealingETA_arr[col] = grad_AnnealingETA_arr[col]+(actualOutput-Y_Ann_NEW[row])*ETA_A
                else:
                    gradArray[col] = gradArray[col]+(actualOutput-Y_NEW[row])*float(Data[row][col])
                    grad_AnnealingETA_arr[col] = grad_AnnealingETA_arr[col]+(actualOutput-Y_Ann_NEW[row])*float(Data[row][col])*ETA_A
                    
        IterationArr.append(misclassified_count)
        IterationETAArr.append(misclassified_ann_count)
        
        newMisClassifiedList = anIterator(IterationArr)
        newAnnMissClassifiedList = anIterator(IterationETAArr)
        for col in range(TotalColumn): 
            #update weights
            weightArr[col] = weightArr[col] + gradArray[col]
            weight_AnnealingETA_arr[col]=weight_AnnealingETA_arr[col]+grad_AnnealingETA_arr[col]
    
    
    with open(outputfileName,'w') as csvFile:
        spamwriter = csv.writer(csvFile)
        spamwriter.writerow(newMisClassifiedList)
        spamwriter.writerow(newAnnMissClassifiedList)
        csvFile.close         
    #print(*IterationArr)
    #print(*IterationETAArr)
         