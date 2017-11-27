# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 07:07:02 2017
@author: baradhwaj
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

os.getcwd()
os.chdir('G:\\Python\\Learning\\Lectures\\ML-Dataset')
wineData = pd.read_csv('wine.data',header=None)
print(wineData)
wineNames = ['Alcohol',
'Malic acid',
'Ash',
'Alcalinity of ash',
'Magnesium',
'Total phenols',
'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity',
'Hue',
'OD280/OD315 of diluted wines',
'Proline']
print(wineNames)
# Del first column baed on index as mentioned in the problem statement
wineData = wineData.drop(wineData.columns[[0]], axis=1) 
wineData.columns = wineNames

columnVals = ['Mean','Median']
#meanMedianValues = pd.DataFrame(data=np.zeros((0,len(columnVals))), columns=columnVals)
def meanValues (df):
    meanValues = df[['Alcohol','Malic acid','Ash','Alcalinity of ash',
       'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
       'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
       'Proline']].mean()
    medianValues = df[['Alcohol','Malic acid','Ash','Alcalinity of ash',
       'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
       'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
       'Proline']].median()
    return meanValues,medianValues
meanValues,medianValues =  meanValues(wineData)
meanMedianValues = pd.concat([meanValues, medianValues],axis=1)
meanMedianValues.columns = columnVals

#div_by_2[i,j] = mat1[i,j]/2
# Failed Attempt wrting to plot inside a method
def drawPlots():  
    for i in wineData.columns:
        for j in wineData.columns:      
            if (i!=j):
              plt.scatter(i,j,data = wineData)              
drawPlots()

plt.scatter('Alcohol','Malic acid',data = wineData)
plt.scatter('Alcohol','Ash',data = wineData)
plt.scatter('Alcohol','Alcalinity of ash',data = wineData)
plt.scatter('Alcohol','Magnesium',data = wineData)
plt.scatter('Alcohol','Total phenols',data = wineData)
plt.scatter('Alcohol','Flavanoids',data = wineData)
plt.scatter('Alcohol','Nonflavanoid phenols',data = wineData)
plt.scatter('Alcohol','Proanthocyanins',data = wineData)
plt.scatter('Alcohol','Color intensity',data = wineData)
plt.scatter('Alcohol','Hue',data = wineData)
plt.scatter('Alcohol','OD280/OD315 of diluted wines',data = wineData)
plt.scatter('Alcohol','Proline',data = wineData)


plt.scatter('Malic acid','Alcohol',data = wineData)
plt.scatter('Malic acid','Ash',data = wineData)
plt.scatter('Malic acid','Alcalinity of ash',data = wineData)
plt.scatter('Malic acid','Magnesium',data = wineData)
plt.scatter('Malic acid','Total phenols',data = wineData)
plt.scatter('Malic acid','Flavanoids',data = wineData)
plt.scatter('Malic acid','Nonflavanoid phenols',data = wineData)
plt.scatter('Malic acid','Proanthocyanins',data = wineData)
plt.scatter('Malic acid','Color intensity',data = wineData)
plt.scatter('Malic acid','Hue',data = wineData)
plt.scatter('Malic acid','OD280/OD315 of diluted wines',data = wineData)
plt.scatter('Malic acid','Proline',data = wineData)


plt.scatter('Alcohol','Malic acid',data = wineData)
plt.scatter('Alcohol','Ash',data = wineData)
plt.scatter('Alcohol','Alcalinity of ash',data = wineData)
plt.scatter('Alcohol','Magnesium',data = wineData)
plt.scatter('Alcohol','Total phenols',data = wineData)
plt.scatter('Alcohol','Flavanoids',data = wineData)
plt.scatter('Alcohol','Nonflavanoid phenols',data = wineData)
plt.scatter('Alcohol','Proanthocyanins',data = wineData)
plt.scatter('Alcohol','Color intensity',data = wineData)
plt.scatter('Alcohol','Hue',data = wineData)
plt.scatter('Alcohol','OD280/OD315 of diluted wines',data = wineData)
plt.scatter('Alcohol','Proline',data = wineData)