# Filename: model.py
# Date: February 14, 2025
# Authors: Javier Chung, Laksumi, Zainab

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import EngFeatures
import heapq
from collections import defaultdict

# pip install xgboost scikit-learn pandas

LINEUP_ATTRIBUTES = [
    'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
    'away_0', 'away_1', 'away_2', 'away_3', 'away_4', 
    'home_team', 'away_team', 'starting_min', 'outcome',
    'season'
]

# ----------------------------- Preprocessing the data for training  ------------------------------------- #
def aquireData(csv):
    # Opening the data
    df = pd.read_csv(csv)

    filteredDF = df[LINEUP_ATTRIBUTES]

    # Getting all unique players and teams as well as there length
    players = set(filteredDF[['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                            'away_0', 'away_1', 'away_2', 'away_3', 'away_4']].stack().unique())

    # making it so that the teams will always be named the same regardless of the year
    teams = set()
    for t in filteredDF['home_team'].unique():
        if t in ("NOK", "NOH"): 
            teams.add("NOP")
        elif t == ("SEA"): 
            teams.add("OKC")
        elif t == ("NJN"): 
            teams.add("BRK")
        elif (t == "CHO"):
            teams.add("CHA")
        else: 
            teams.add(t)
    teams = sorted(teams)
    numTeams = len(teams)

    # Convert to sorted list and add the unknown token
    players = sorted(players)
    if "<UNK>" not in players:
        players.append("<UNK>")
    numPlayers = len(players)

    # getting the roster of each team
    rosters = { 
        teams[0]: [], teams[1]: [], teams[2]: [], teams[3]: [], teams[4]: [], teams[5]: [],
        teams[6]: [], teams[7]: [], teams[8]: [], teams[9]: [], teams[10]: [], teams[11]: [],
        teams[12]: [], teams[13]: [], teams[14]: [], teams[15]: [], teams[16]: [], teams[17]: [],
        teams[18]: [], teams[19]: [], teams[20]: [], teams[21]: [], teams[22]: [], teams[23]: [],
        teams[24]: [], teams[25]: [], teams[26]: [], teams[27]: [], teams[28]: [], teams[29]: []
    }

    # Creating a dictionary to map each player and team to an indexed value
    playerToIndex = {player: idx for idx, player in enumerate(players)}
    teamToIndex = dict(zip(teams, range(numTeams)))
    playerToIndex = dict(playerToIndex)

    homeLineups = []
    awayLineups = []
    homeTeams = []
    awayTeams = []
    startingMin = []
    year = []
    y = []
    x = []

    # PREPROCESSING DATA
    # Iterating through the lineup dataframe to convert each player to indexes
    for _, row in filteredDF.iterrows():

        homeTeamID = row['home_team']
        awayTeamID = row['away_team']

        # making it so that the teams will always be named the same regardless of the year
        if (homeTeamID == "NOK" or homeTeamID == "NOH"):
            homeTeamID = "NOP"
        if (homeTeamID == "SEA"):
            homeTeamID = "OKC"
        if (homeTeamID == "NJN"):
            homeTeamID = "BRK"
        if (homeTeamID == "CHO"):
            homeTeamID = "CHA"

        if (awayTeamID == "NOK" or awayTeamID == "NOH"):
            awayTeamID = "NOP"
        if (awayTeamID == "SEA"):
            awayTeamID = "OKC"
        if (awayTeamID == "NJN"):
            awayTeamID = "BRK"
        if (awayTeamID == "CHO"):
            awayTeamID = "CHA"

        # Populating players to the roster
        for team in teams:
            if (team == homeTeamID):
                rosters[team].append(row['home_0'])
                rosters[team].append(row['home_1'])
                rosters[team].append(row['home_2'])
                rosters[team].append(row['home_3'])
                rosters[team].append(row['home_4'])
            if (team == awayTeamID):
                rosters[team].append(row['away_0'])
                rosters[team].append(row['away_1'])
                rosters[team].append(row['away_2'])
                rosters[team].append(row['away_3'])
                rosters[team].append(row['away_4'])

        home = [row[f'home_{j}'] for j in range(5)]
        away = [row[f'away_{j}'] for j in range(5)]

        # Indexing players and teams in the lineups
        indexedLineupHome = [playerToIndex.get(p, playerToIndex["<UNK>"]) for p in home]
        homeLineups.append(indexedLineupHome)
        indexedLineupAway = [playerToIndex.get(p, playerToIndex["<UNK>"]) for p in away]
        awayLineups.append(indexedLineupAway)
        indexedHomeTeam = [teamToIndex[homeTeamID]]
        indexedAwayTeam = [teamToIndex[awayTeamID]]
        homeTeams.append(indexedHomeTeam)
        awayTeams.append(indexedAwayTeam)

        # Normalizing the starting minute and outcome data and year
        startingMin.append(row['starting_min'] / 47) 
        year.append((row['season'] - 2007)/ (2016 - 2007))
        classifier = row['outcome']
        if(classifier == -1):
            classifier = 0
        y.append(classifier)

    # Populating training data
    for i in range(len(homeLineups)):
        # features = homeTeams[i] + awayTeams[i] + homeLineups[i] + awayLineups[i] + [startingMin[i]] + [year[i]]
        features = homeLineups[i] + awayLineups[i] + [startingMin[i]] + [year[i]]

        x.append(features)

    return x, y, rosters, playerToIndex, teamToIndex

# ----------------------------- Training the model using the XGBoost model  ------------------------------------- #
# This method is used to 
def trainModel(x,y):

    # # Evaluating model
    # X_train, X_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=42
    # )

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=50,
        learning_rate=0.001,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(x, y)


    # Evaluating model
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return (model)

# -------------------------------- This Methods is used for prediction ---------------------------- #
# This method is used to predict which player will provide the outcome of 1 when given both team names
# 4 home players, 5 away players, starting minute and season for each test example in the test data given
# It will then track if the player that resulted in the outcome of 1 is the same player that was in the result data
# @params: 
#   rosters - rosters of the data aquired
#   tests - array containing all test examples 
#   testResults - arrays of the actual player that was playing in the home lineup for the example
#   playerToIndex - the dictionary of mapping players to an index
#   teamToIndex - the dictionary of mapping team names to indexes
# @return:
#   This method returns the amount of correct results there were
def predictTests(rosters, tests, testResults, playerToIndex, teamToIndex, freqOfEachPlayer, synergyMatrix, winRateMatrix, model, predData):
    
    correctResults = 0

    # Now for each year test case we must predict the player
    for i in range(len(tests)):
        
        bestPlayer = None
        bestProb = -float('inf');

        # getting the test input
        homeTeamInput = tests[i]['home_team']
        awayTeamInput = tests[i]['away_team']

        # making it so that the teams will always be named the same regardless of the year
        if (homeTeamInput == "NOK" or homeTeamInput == "NOH"):
            homeTeamInput = "NOP"
        if (homeTeamInput == "SEA"):
            homeTeamInput = "OKC"
        if (homeTeamInput == "NJN"):
            homeTeamInput = "BRK"
        if (homeTeamInput == "CHO"):
            homeTeamInput = "CHA"

        if (awayTeamInput == "NOK" or awayTeamInput == "NOH"):
            awayTeamInput = "NOP"
        if (awayTeamInput == "SEA"):
            awayTeamInput = "OKC"
        if (awayTeamInput == "NJN"):
            awayTeamInput = "BRK"
        if (awayTeamInput == "CHO"):
            awayTeamInput = "CHA"

        # Getting given data in the tests
        homeTeamLineupInput = [tests[i][col] for col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4'] if tests[i][col] != "?"]
        awayTeamLineupInput = [tests[i]['away_0'], tests[i]['away_1'],tests[i]['away_2'], tests[i]['away_3'], tests[i]['away_4']]
        startingMinInput = tests[i]['starting_min']/47        
        year = (tests[i]['season'] - 2007)/ (2016 - 2007)

        # Sorting and turning the names into index
        homeTeamIndex = [teamToIndex[homeTeamInput]]
        awayTeamIndex = [teamToIndex[awayTeamInput]]
        sortedawayLineupInput = sorted(awayTeamLineupInput)
        awayLineUpIndices = [playerToIndex.get(p, playerToIndex["<UNK>"]) for p in sortedawayLineupInput]

        # Making sure that the players that are being tested are in the correct home roster and isnt any of the current players
        viablePlayers = [p for p in set(rosters[homeTeamInput]) if p not in homeTeamLineupInput]

        topPlayers = []
        w1, w2, w3, w4 = 0.8, 0.1, 0.05, 0.05

        # predicing the player that would fit in this lineup
        for player in viablePlayers:
            homeLineupsPred = homeTeamLineupInput + [player] # Adding the last player to make the lineup have 5 players
            sortedHomeLineupInput = sorted(homeLineupsPred)
            homeLineUpIndices = [playerToIndex.get(p, playerToIndex["<UNK>"]) for p in sortedHomeLineupInput]

            features = homeLineUpIndices + awayLineUpIndices + [startingMinInput] + [year]
            # features = homeTeamIndex + awayTeamIndex + homeLineUpIndices + awayLineUpIndices + [startingMinInput] + [year]

            # Predict the probability of home team winning with this player
            # Probability for class = 1
            winProb = model.predict_proba([features])[0][1]  

            # Computing additional metrics
            lineupWinRate = EngFeatures.getLineupWinRate(homeLineupsPred, winRateMatrix)
            lineupSynergy = EngFeatures.getLineupSynergy(homeLineupsPred, synergyMatrix)
            crossSynergy = EngFeatures.getCrossSynergy(homeLineupsPred, sortedawayLineupInput, synergyMatrix)
            freq = freqOfEachPlayer.get(player, 0)

            overallScore = w1 * winProb + w2 * lineupWinRate + w3 * lineupSynergy + w4 * crossSynergy 

            # selecting the best probability top 10 players
            if len(topPlayers) < 10:
                heapq.heappush(topPlayers, (overallScore, player))
            else:
                # Adjusting the players so that only the top probailities will be stored
                heapq.heappushpop(topPlayers, (overallScore, player))

        # Sotring the players by descending order
        topPlayers = sorted(topPlayers, key=lambda x: (-x[0], x[1]))

        # Postproccessing to select the player with highest historical frequency, synergies and winrates
        maxFreq = -1
        bestProb = -1
        for prob, player in topPlayers:
            freq = freqOfEachPlayer.get(player, 0)

            # selecting the player with the higher frequency (if theres a tie, higher prob wins)
            if freq > maxFreq or (freq == maxFreq and prob > bestProb):
                bestPlayer = player
                maxFreq = freq
                bestProb = prob

        # Check if the selected player is correct
        if bestPlayer == testResults[i]:
            correctResults += 1
        
        # print(f"actualPlayer {testResults[i]} vs predicted player {bestPlayer}\n")

        # Writing to the dictionary containing the predicted results
        predData['Game_ID'].append(str(tests[i]['season']) + awayTeamInput + "@" + homeTeamInput)
        predData['Home_Team'].append(homeTeamInput)
        predData['Fifth_Player'].append(bestPlayer)

        

        if (bestPlayer == testResults[i]):
            correctResults = correctResults + 1


    return (correctResults)

# getting each synergy, winrate and frequency of each player of each year
def getPostProcData(csv):
    df = pd.read_csv(csv)
    filteredDF = df[LINEUP_ATTRIBUTES]
    freqOfEachPlayer = defaultdict(int)

    # Iterating through the lineup dataframe to convert each player to indexes
    for _, row in filteredDF.iterrows():

        home = [row[f'home_{j}'] for j in range(5)]

        # calculating how many times the player appears
        for player in home:
            freqOfEachPlayer[player] += 1

    synergyMatrix = EngFeatures.getSynergyMatrix(filteredDF)
    winRateMatrix = EngFeatures.getWinRateMatrix(filteredDF)


    return synergyMatrix, winRateMatrix, freqOfEachPlayer

# ------------------------- Creating Models For Prediction -------------------- #
# Combining all the matchup data into one dataset
matchupData = [
    "./NBAData/matchups-2007.csv",
    "./NBAData/matchups-2008.csv",
    "./NBAData/matchups-2009.csv",
    "./NBAData/matchups-2010.csv",
    "./NBAData/matchups-2011.csv",
    "./NBAData/matchups-2012.csv",
    "./NBAData/matchups-2013.csv",
    "./NBAData/matchups-2014.csv",
    "./NBAData/matchups-2015.csv"
]

combinedMatchupDF = pd.concat((pd.read_csv(f) for f in matchupData), ignore_index=True)

# Saving combined DataFrame into one CSV
combinedMatchupDF.to_csv("./NBAData/matchupsUltimate.csv", index=False)

x, y, rostersMega, playerToIndexMega, teamToIndexMega = aquireData("./NBAData/matchupsUltimate.csv")
modelMega = trainModel(x,y)

synergyMatrix07, winRateMatrix07, freqOfEachPlayer07 = getPostProcData("./NBAData/matchups-2007.csv")
synergyMatrix08, winRateMatrix08, freqOfEachPlayer08 = getPostProcData("./NBAData/matchups-2008.csv")
synergyMatrix09, winRateMatrix09, freqOfEachPlayer09 = getPostProcData("./NBAData/matchups-2009.csv")
synergyMatrix10, winRateMatrix10, freqOfEachPlayer10 = getPostProcData("./NBAData/matchups-2010.csv")
synergyMatrix11, winRateMatrix11, freqOfEachPlayer11 = getPostProcData("./NBAData/matchups-2011.csv")
synergyMatrix12, winRateMatrix12, freqOfEachPlayer12 = getPostProcData("./NBAData/matchups-2012.csv")
synergyMatrix13, winRateMatrix13, freqOfEachPlayer13 = getPostProcData("./NBAData/matchups-2013.csv")
synergyMatrix14, winRateMatrix14, freqOfEachPlayer14 = getPostProcData("./NBAData/matchups-2014.csv")
synergyMatrix15, winRateMatrix15, freqOfEachPlayer15 = getPostProcData("./NBAData/matchups-2015.csv")

# ----------------------------------------- Gathering Tests ------------------------------------------------- #

# getting test data and seperating it by year
testDF = pd.read_csv("./NBAData/NBA_test.csv")
testDFResults = pd.read_csv("./NBAData/NBA_test_labels.csv")

# Storing each test data and test result data in a 2d array
tests = {year: [] for year in range(2007, 2017)} 
testsResults = {year: [] for year in range(2007, 2017)} 

# Iterating through the rows and classifying them by season
for index, row in testDF.iterrows():
    season = row['season']
    if season in tests:
        tests[season].append(row) # stores all the test data
        testsResults[season].append(testDFResults.iloc[index].values[0]) # Stores the result of the test data

predData = {
    'Game_ID': [],
    'Home_Team': [],
    'Fifth_Player': []
}

print("Making Predictions...")

totalResults = 0

# Used for testing...
# correctResults = predictTests(rosters07, tests[2008], testsResults[2008], playerToIndex07, teamToIndex07, freqOfEachPlayer07, synergyMatrix07, winRateMatrix07,  modelMega, predData)
# print(f"2008 TESTS - accuracy: {correctResults/(len(tests[2008]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2008]) + 1} correct results")
# totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2007], testsResults[2007], playerToIndexMega, teamToIndexMega, freqOfEachPlayer07, synergyMatrix07, winRateMatrix07, modelMega, predData)
print(f"2007 TESTS - accuracy: {correctResults/(len(tests[2007]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2007]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2008], testsResults[2008], playerToIndexMega, teamToIndexMega, freqOfEachPlayer08, synergyMatrix08, winRateMatrix08, modelMega, predData)
print(f"2008 TESTS - accuracy: {correctResults/(len(tests[2008]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2008]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2009], testsResults[2009], playerToIndexMega, teamToIndexMega, freqOfEachPlayer09, synergyMatrix09, winRateMatrix09, modelMega, predData)
print(f"2009 TESTS - accuracy: {correctResults/(len(tests[2009]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2009]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2010], testsResults[2010], playerToIndexMega, teamToIndexMega, freqOfEachPlayer10, synergyMatrix10, winRateMatrix10, modelMega, predData)
print(f"2010 TESTS - accuracy: {correctResults/(len(tests[2010]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2010]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2011], testsResults[2011], playerToIndexMega, teamToIndexMega, freqOfEachPlayer11, synergyMatrix11, winRateMatrix11, modelMega, predData)
print(f"2011 TESTS - accuracy: {correctResults/(len(tests[2011]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2011]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2012], testsResults[2012], playerToIndexMega, teamToIndexMega, freqOfEachPlayer12, synergyMatrix12, winRateMatrix12, modelMega, predData)
print(f"2012 TESTS - accuracy: {correctResults/(len(tests[2012]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2012]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2013], testsResults[2013], playerToIndexMega, teamToIndexMega, freqOfEachPlayer13, synergyMatrix13, winRateMatrix13, modelMega, predData)
print(f"2013 TESTS - accuracy: {correctResults/(len(tests[2013]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2013]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2014], testsResults[2014], playerToIndexMega, teamToIndexMega, freqOfEachPlayer14, synergyMatrix14, winRateMatrix14, modelMega, predData)
print(f"2014 TESTS - accuracy: {correctResults/(len(tests[2014]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2014]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2015], testsResults[2015], playerToIndexMega, teamToIndexMega, freqOfEachPlayer15, synergyMatrix15, winRateMatrix15, modelMega, predData)
print(f"2015 TESTS - accuracy: {correctResults/(len(tests[2015]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2015]) + 1} correct results")
totalResults = totalResults + correctResults

correctResults = predictTests(rostersMega, tests[2016], testsResults[2016], playerToIndexMega, teamToIndexMega, freqOfEachPlayer15, synergyMatrix15, winRateMatrix15, modelMega, predData)
print(f"2016 TESTS - accuracy: {correctResults/(len(tests[2016]) + 1)*100:.4f}%\nThere are {correctResults}/{len(tests[2016]) + 1} correct results")
totalResults = totalResults + correctResults

totalTests = (len(tests[2007]) + 1 + len(tests[2008]) + 1 + len(tests[2009]) + 
              1 + len(tests[2010]) + 1 + len(tests[2011]) + 1 + len(tests[2012]) + 1
               + len(tests[2013]) + 1 + len(tests[2014]) + 1 + len(tests[2015]) + 1
                + len(tests[2016]) + 1)
print(f"\n\n The average of correct results is:\naccuracy: {(totalResults/totalTests):.4f}")

predDF = pd.DataFrame(predData)

# Writing the results to a csv file
predDF.to_csv('./PredictionData/output.csv', index=False)
