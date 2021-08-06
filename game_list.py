# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:09:43 2021

@author: Poojan
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# Importing the dataset
dataset = pd.read_csv('game_data.csv')

X = dataset.iloc[:, [1,2,3,4]].values
y = dataset.iloc[:, 5].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X_train, y_train)

print(X_test)

def game_list(ip_critic, ip_userrating, ip_globalsales, ip_year):
  # Predicting the Test set results
  y_pred = classifier.predict([[ip_critic, ip_userrating, ip_globalsales, ip_year]])

  print(y_pred)

  game_dict = {}
  with open('game_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    print(reader)
    count = 0 
    for row in reader:
      if int(row['Result']) == int(y_pred[0]):
        game_dict.update({count:[row['Name']]})
        count+=1  
  print(game_dict)  
  return game_dict

