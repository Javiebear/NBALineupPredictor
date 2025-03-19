# Filename: EngFeatures.py
# Date: February 14, 2025
# Authors: Javier Chung, Laksumi, Zainab

import itertools
from collections import defaultdict
import pandas as pd
import numpy as np

# pip install numpy

# Computing Synergies between each unique pairs of players in each lineup that result in an outcome of 1(!10 - 10 factorial)
# this is done by using the intertools library
# @param:
#   df - is the dataframe of the csv file
# @return
#   retruns synergy matrix dictionary
def getSynergyMatrix(df):

    # dictionary containing the amount of times the pair of player have played with each other and won
    synergy = defaultdict(lambda: {'count': 0, 'win': 0})
    for _, row in df.iterrows():

        # normalizing outcome to be 1 or 0
        outcome = 1 if row['outcome'] == 1 else 0
        # getting team lineups
        homeLineup = [row[f'home_{i}'] for i in range(5)]
        awayLineup = [row[f'away_{i}'] for i in range(5)]

        # Computing the synergy of pairs of players for wins
        for p1, p2 in itertools.combinations(homeLineup, 2):
            key = tuple(sorted((p1, p2)))
            synergy[key]['count'] += 1
            synergy[key]['win'] += outcome

        # Computing the synergy of pairs of player for losses (want to maximize wins on home team)
        for p1, p2 in itertools.combinations(awayLineup, 2):
            key = tuple(sorted((p1, p2)))
            synergy[key]['count'] += 1
            synergy[key]['win'] += (1 - outcome)  # because outcome refers to home team

    return synergy
    
# this method calculates the win rate of each player
# this is done by using the intertools library
# @param:
#   df - is the dataframe of the csv file
# @return
#   returns winrate matrix
def getWinRateMatrix(df):
    
    playerWinRate = defaultdict(lambda: {"wins": 0, "games": 0})

    for idx, row in df.iterrows():
        # normalizing outcome to be 1 or 0
        outcome = 1 if row['outcome'] == 1 else 0     
        homeLineup = [row[f'home_{i}'] for i in range(5)]
        awayLineup = [row[f'away_{i}'] for i in range(5)]
   
        # Incresing win rate of players in home team
        for player in homeLineup:
            playerWinRate[player]["wins"] += outcome
            playerWinRate[player]["games"] += 1

        # Incresing win rate of players in away team
        for player in awayLineup:
            playerWinRate[player]["wins"] += (1 - outcome)
            playerWinRate[player]["games"] += 1
    return (playerWinRate)

# This method is used to compute the average winrate each players in that lineup
# @params:
#    lineup - the 1d array of the players in the lineup
#    winRateMatrix - win rate matrix dictionary
# @return:
#    returns the average winrate
def getLineupWinRate(lineup, winRateMatrix):
    rates = []
    for player in lineup:
        stats = winRateMatrix.get(player, {"wins": 0, "games": 0})
        if stats["games"] > 0:
            rates.append(stats["wins"] / stats["games"])
        else:
            rates.append(0.5)  # fallback
    return np.mean(rates) if rates else 0.5

# This method is used to compute the average synergy of each pairs of players in that lineup
# @params:
#    lineup - the 1d array of the players in the lineup
#    synergyMatrix - synergy matrix dictionary
# @return:
#    returns the average score of how well each pair performs
def getLineupSynergy(lineup, synergyMatrix):

    score = 0
    pairCount = 0

    # Checking how often each pair wins to how many times they occur
    for p1, p2 in itertools.combinations(lineup, 2):
        key = tuple(sorted((p1, p2)))

        if key in synergyMatrix and synergyMatrix[key]['count'] > 0:
            win_rate = synergyMatrix[key]['win'] / synergyMatrix[key]['count']
            score += win_rate
            pairCount += 1

    # if no pairs are found, the synergy is 0.5 for a neutral synergy (unknown pair)
    # other wise it is the score of the synergy
    return score / pairCount if pairCount > 0 else 0.5

# This method is used to compute the average synergy betwen each team
# @params:
#    homeLineup - the 1d array of the players in the home lineup
#    awayLineup - the 1d array of the players in the home lineup
#    synergyMatrix - synergy matrix dictionary
# @return:
#    returns the average score of how well each pair performs between the teams
def getCrossSynergy(homeLineup, awaylineup, synergyMatrix):
    crossScore = 0
    pairAmt = 0
    for homeP in homeLineup:
        for awayP in awaylineup:
            key = tuple(sorted((homeP, awayP)))
            if key in synergyMatrix and synergyMatrix[key]['count'] > 0:
                win_rate = synergyMatrix[key]['win'] / synergyMatrix[key]['count']
                crossScore += win_rate
                pairAmt += 1
    return crossScore / pairAmt if pairAmt > 0 else 0.5