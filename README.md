# NBALineupPredictor
SOFE 4620U - Machine Learning & Data Mining

**Group 2 Members:**

Javier Chung - 100785653

Laksumi Somaskanthamoorthy - 100782723 

Zainab Nomani - 100784761

# Introduction
The repository contains a machine learning algorithm that can predict the optimal fifth player of a home team lineup when given, the season, starting minute, both team names, four home team players and five away team players. It was trained using past historical data of NBA games ranging from 2007 - 2015.

# Objective
The objective of this project is to develop a machine learning predicate model that is capable of processing data from a given year and find the optimal fifth player of the partial starting lineup of the following year. The model will be given the starting four players of the home team, all five players of the away team and then the model will give the fifth player who will help the team succeed. The model is built on selected features, as not all features are permitted to be used.
 
## Requirements
- Tested using python version: **python3.11**
  
Libraries:
- xgboost
- scikit-learn
- pandas
- numpy
- gensim

```bash
pip install xgboost scikit-learn pandas numpy gensim 
```
## Execution
Clone and go to the repository 
```bash
git clone https://github.com/Javiebear/NBALineupPredictor
```
Execute Script
```bash
python model.py
```

# Overview

