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

wine = pd.read_csv('wine.data',header=None)
print(wine)
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

wineData = wine.drop(wine.columns[[0]], axis=1) 
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

wineData.describe()

wineData['Alcohol'].corr(wineData['Proline'])
wineData['Malic acid'].corr(wineData['Proline'])
wineData['Ash'].corr(wineData['Proline'])
wineData['Alcalinity of ash'].corr(wineData['Proline'])
wineData['Magnesium'].corr(wineData['Proline'])
wineData['Total phenols'].corr(wineData['Proline'])
wineData['Flavanoids'].corr(wineData['Proline'])
wineData['Nonflavanoid phenols'].corr(wineData['Proline'])
wineData['Proanthocyanins'].corr(wineData['Proline'])
wineData['Color intensity'].corr(wineData['Proline'])
wineData['Hue'].corr(wineData['Proline'])
wineData['OD280/OD315 of diluted wines'].corr(wineData['Proline'])

## Step 2: K Means Clustering
# Let's build 3 cluster model (k = 3)
wineCluster = KMeans(n_clusters=3,random_state=100).fit(wineData)
wineCluster.labels_
wineData_withClusterlbl = wineData.copy()
wineData_withClusterlbl['Cluster_Label'] = wineCluster.labels_

# Alcohol
plt.scatter('Alcohol','Malic acid',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Alcalinity of ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Magnesium',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Total phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Flavanoids',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Nonflavanoid phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Proanthocyanins',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Color intensity',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','Hue',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcohol','OD280/OD315 of diluted wines',data = wineData,c=wineCluster.labels_)

# Malic Acid
plt.scatter('Malic acid','Alcohol',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Alcalinity of ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Magnesium',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Total phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Flavanoids',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Nonflavanoid phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Proanthocyanins',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Color intensity',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','Hue',data = wineData,c=wineCluster.labels_)
plt.scatter('Malic acid','OD280/OD315 of diluted wines',data = wineData,c=wineCluster.labels_)

plt.scatter('Ash','Alcohol',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Malic acid',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Alcalinity of ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Magnesium',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Total phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Flavanoids',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Nonflavanoid phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Proanthocyanins',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Color intensity',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','Hue',data = wineData,c=wineCluster.labels_)
plt.scatter('Ash','OD280/OD315 of diluted wines',data = wineData,c=wineCluster.labels_)

plt.scatter('Alcalinity of ash','Alcohol',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Malic acid',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Magnesium',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Total phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Flavanoids',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Nonflavanoid phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Proanthocyanins',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Color intensity',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','Hue',data = wineData,c=wineCluster.labels_)
plt.scatter('Alcalinity of ash','OD280/OD315 of diluted wines',data = wineData,c=wineCluster.labels_)

plt.scatter('Magnesium','Alcohol',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Malic acid',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Alcalinity of ash',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Total phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Flavanoids',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Nonflavanoid phenols',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Proanthocyanins',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Color intensity',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','Hue',data = wineData,c=wineCluster.labels_)
plt.scatter('Magnesium','OD280/OD315 of diluted wines',data = wineData,c=wineCluster.labels_)


# In General , all columns seem to be form cluster with Proline column


#plt.scatter("Petal.Length","Petal.Width",data=wineData_withClusterlbl, c=wineCluster.labels_)

plt.scatter('Alcohol','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_)
            #data = wineData,c=wineCluster.labels_) # Best
plt.scatter('Malic acid','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_)  # Good
plt.scatter('Ash','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Good
plt.scatter('Alcalinity of ash','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Magnesium','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Flavanoids','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Best
plt.scatter('Nonflavanoid phenols','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Proanthocyanins','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Color intensity','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Green cluster is better
plt.scatter('Hue','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) ## Good 
plt.scatter('OD280/OD315 of diluted wines','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # good

## Step 3&4: Evaluate Clustering and Optimizing Clusters 
wine_cluster_centers = wineCluster.cluster_centers_
# euclidean distance between each observation and the centroid of the cluster it belongs to
wine_cdist_output = np.min(cdist(wineData, wineCluster.cluster_centers_,'euclidean'),axis=1)

# Sum of squared distance for K = 2
wine_within_cluster_distance = np.sum(wine_cdist_output**2)

# Building clusters with k=1 till k=10 and comparing the
#within cluster distance between each clustering output
wine_wss = pd.Series([0.0]*10,index = range(1,11))
for k in range(1,11):
    wineclust = KMeans(n_clusters = k,random_state=100).fit(wineData)
    wine_cdist_op = np.min(cdist(wineData,wineCluster.cluster_centers_,'euclidean'),axis=1)
    #np.min(cdist(wineData,wineCluster.cluster_centers_,'euclidean'),axis=1)
    wine_wss[k] = np.sum(wine_cdist_op**2)

print(wine_wss)

# Elbow curve
plt.plot(wine_wss)
