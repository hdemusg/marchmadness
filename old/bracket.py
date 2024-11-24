import pandas as pd
import sklearn
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()

def testing():
    pass

def generate_bracket(data_file, output_file):
    '''
    training data: 2022 tournament
    test data: 2023 base bracket
    '''
    template = "2024_template.xlsx"
    #final_training = pd.read_excel("2023_data.xlsx", sheet_name="train")
    final_training = pd.read_csv(data_file, index_col=0)
    final_test = pd.read_csv(template, index_col=0)
    #print(training)
    #final_test = pd.read_excel("2023_data.xlsx", sheet_name="test")
    #print(final_test)

    # create models on an 80-20 split of training data, select the best performing predictor to use on 2023 data
    from sklearn.linear_model import LinearRegression  # Lasso
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    #from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    lscore = 0
    #kscore = 0
    #dscore = 0
    #ascore = 0
    #rscore = 0

    for run in range(1, 6):
        # create training and test split of the pre-2022 data
        #print(final_training.columns)
        data_x = final_training[['Round', 'Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
        data_y = final_training[['Score 1', 'Score 2']]
        ft_train_x, ft_test_x, ft_train_y, ft_test_y = train_test_split(data_x, data_y, train_size = 0.75)

        print("Run", run)
        lr = LinearRegression()
        train_x = ft_train_x.drop('Round', axis=1)
        lr.fit(train_x, ft_train_y)

        lr_correct = 0
        lr_game_value = 0
        values = {'First Four': 0.5, 'R1': 1, 'R2': 2, 'Sweet 16': 4, 'Elite 8': 8, 'Final 4': 16, 'Championship': 32}
        for i in range(len(ft_test_x)):
            data = np.asarray(ft_test_x.iloc[i]).reshape(1, -1)
            round = data[0][0]
            rv = values[round]
            #print(data)
            pred = lr.predict([data[0][1:]])[0]
            res = ft_test_y.iloc[i].values
            #print(pred, res)
            if pred[0] < pred[1] and res[0] < res[1]:
                lr_correct += rv
            elif pred[0] >= pred[1] and res[0] >= res[1]:
                lr_correct += rv
            lr_game_value += rv
        print("Linear Regressor accuracy: ", float(lr_correct / lr_game_value))
        lscore += float(lr_correct / lr_game_value)

 
        kn = KNeighborsRegressor()
        train_x = ft_train_x.drop('Round', axis=1)
        kn.fit(train_x, ft_train_y)

        kn_correct = 0
        kn_game_value = 0
        for i in range(len(ft_test_x)):
            data = np.asarray(ft_test_x.iloc[i]).reshape(1, -1)
            #print(data)
            round = data[0][0]
            rv = values[round]
            pred = kn.predict([data[0][1:]])[0]
            res = ft_test_y.iloc[i].values
            #print(pred, res)
            if pred[0] < pred[1] and res[0] < res[1]:
                kn_correct += rv
            elif pred[0] >= pred[1] and res[0] >= res[1]:
                kn_correct += rv
            kn_game_value += rv
        print("K Neighbors Regressor accuracy: ", float(kn_correct / kn_game_value))
        kscore += float(kn_correct / kn_game_value) 

        dt = DecisionTreeRegressor()
        train_x = ft_train_x.drop('Round', axis=1)
        dt.fit(train_x, ft_train_y)

        dt_correct = 0
        dt_game_value = 0
        for i in range(len(ft_test_x)):
            data = np.asarray(ft_test_x.iloc[i]).reshape(1, -1)
            round = data[0][0]
            rv = values[round]
            pred = dt.predict([data[0][1:]])[0]
            res = ft_test_y.iloc[i].values
            #print(pred, res)
            if pred[0] < pred[1] and res[0] < res[1]:
                dt_correct += rv
            elif pred[0] >= pred[1] and res[0] >= res[1]:
                dt_correct += rv
            dt_game_value += rv
        print("K Neighbors Regressor accuracy: ", float(dt_correct / dt_game_value))
        dscore += float(dt_correct / dt_game_value)
        
        '''
        la = Lasso(alpha = 0.5)
        train_x = ft_train_x.drop('Round', axis=1)
        la.fit(train_x, ft_train_y)

        la_correct = 0
        la_game_value = 0
        for i in range(len(ft_test_x)):
            data = np.asarray(ft_test_x.iloc[i]).reshape(1, -1)
            round = data[0][0]
            rv = values[round]
            pred = dt.predict([data[0][1:]])[0]
            res = ft_test_y.iloc[i].values
            #print(pred, res)
            if pred[0] < pred[1] and res[0] < res[1]:
                la_correct += rv
            elif pred[0] >= pred[1] and res[0] >= res[1]:
                la_correct += rv
            la_game_value += rv
        print("K Neighbors Regressor accuracy: ", float(la_correct / la_game_value))
        ascore += float(la_correct / la_game_value)

        rf = RandomForestRegressor()
        train_x = ft_train_x.drop('Round', axis=1)
        rf.fit(train_x, ft_train_y)

        rf_correct = 0
        rf_game_value = 0
        for i in range(len(ft_test_x)):
            data = np.asarray(ft_test_x.iloc[i]).reshape(1, -1)
            round = data[0][0]
            rv = values[round]
            pred = dt.predict([data[0][1:]])[0]
            res = ft_test_y.iloc[i].values
            #print(pred, res)
            if pred[0] < pred[1] and res[0] < res[1]:
                rf_correct += rv
            elif pred[0] >= pred[1] and res[0] >= res[1]:
                rf_correct += rv
            rf_game_value += rv
        print("K Neighbors Regressor accuracy: ", float(rf_correct / rf_game_value))
        ascore += float(rf_correct / rf_game_value)
        '''

    print("Linear Regressor average accuracy: ", float(lscore/5))
    print("K Neighbors Regressor average accuracy: ", float(kscore/5))
    print("Decision Tree Regressor average accuracy: ", float(dscore/5))
    # print("Lasso Regressor average accuracy:", float(ascore/5))
    # print("Random Forest average accuracy:", float(rscore/5))

    if max(dscore, kscore, lscore) == lscore:
        print("Now using linear regression on the 2024 bracket!")
        model = LinearRegression()
    elif max(dscore, kscore, lscore) == kscore:
        print("Now using k neighbors regression on the 2024 bracket!")
        model = KNeighborsRegressor()
    else: 
        print("Now using decision tree regression on the 2024 bracket!")
        model = DecisionTreeRegressor()

    data_x = data_x.drop('Round', axis=1)
    model.fit(data_x, data_y)
    bracket = final_test

    '''
    # First Four - Texas Southern v Fairleigh Dickinson
    firstfour_3 = bracket.iloc[2][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
    ff3_pred = model.predict(np.asarray(firstfour_3).reshape(1, -1))[0]
    ff3_team1 = bracket.iloc[2][['Team 1']]
    ff3_team2 = bracket.iloc[2][['Team 2']]
    #print(ff3_team1.values[0], ff3_pred[0])
    #print(ff3_team2.values[0], ff3_pred[1])
    bracket.iat[2, bracket.columns.get_loc('Score 1')] = int(ff3_pred[0])
    bracket.iat[2, bracket.columns.get_loc('Score 2')] = int(ff3_pred[1])
    if ff3_pred[0] < ff3_pred[1]:
        wins = int(bracket.iloc[2][['Wins 2']].values[0]) + 1
        gp = int(bracket.iloc[2][['GP 2']].values[0]) + 1
        winpct = float(bracket.iloc[2][['Wins 2']].values[0] / bracket.iloc[2][['GP 2']].values[0])
        bracket.iat[2, bracket.columns.get_loc('Winner')] = ff3_team2.values[0]
        bracket.iat[12, bracket.columns.get_loc('Team 2')] = ff3_team2.values[0]
        bracket.iat[12, bracket.columns.get_loc('Offense 2')] = bracket.iloc[2][['Offense 2']].values[0]
        bracket.iat[12, bracket.columns.get_loc('Defense 2')] = bracket.iloc[2][['Defense 2']].values[0]
        bracket.iat[12, bracket.columns.get_loc('Wins 2')] = wins
        bracket.iat[12, bracket.columns.get_loc('GP 2')] = gp
        bracket.iat[12, bracket.columns.get_loc('Win Pct 2')] = winpct
        if int(ff3_pred[0]) == int(ff3_pred[1]):
            bracket.iat[2, bracket.columns.get_loc('Score 2')] = int(ff3_pred[1]) + 1
    else:
        wins = int(bracket.iloc[2][['Wins 1']].values[0]) + 1
        gp = int(bracket.iloc[2][['GP 1']].values[0]) + 1
        winpct = float(bracket.iloc[2][['Wins 1']].values[0] / bracket.iloc[2][['GP 1']].values[0])
        bracket.iat[2, bracket.columns.get_loc('Winner')] = ff3_team1.values[0]
        if int(ff3_pred[0]) == int(ff3_pred[1]):
            bracket.iat[2, bracket.columns.get_loc('Score 1')] = int(ff3_pred[0]) + 1
        bracket.iat[12, bracket.columns.get_loc('Team 2')] = ff3_team1.values[0]
        bracket.iat[12, bracket.columns.get_loc('Offense 2')] = bracket.iloc[2][['Offense 1']].values[0]
        bracket.iat[12, bracket.columns.get_loc('Defense 2')] = bracket.iloc[2][['Defense 1']].values[0]        
        bracket.iat[12, bracket.columns.get_loc('Wins 2')] = wins
        bracket.iat[12, bracket.columns.get_loc('GP 2')] = gp
        bracket.iat[12, bracket.columns.get_loc('Win Pct 2')] = winpct
    #print(bracket.iloc[2])
    #print(bracket.iloc[20])

     # First Four - Arizona State v Nevada
    firstfour_4 = bracket.iloc[3][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
    ff4_pred = model.predict(np.asarray(firstfour_4).reshape(1, -1))[0]
    ff4_team1 = bracket.iloc[3][['Team 1']]
    ff4_team2 = bracket.iloc[3][['Team 2']]
    #print(ff4_team1.values[0], ff4_pred[0])
    #print(ff4_team2.values[0], ff4_pred[1])
    bracket.iat[3, bracket.columns.get_loc('Score 1')] = int(ff4_pred[0])
    bracket.iat[3, bracket.columns.get_loc('Score 2')] = int(ff4_pred[1])
    if ff4_pred[0] < ff4_pred[1]:
        wins = int(bracket.iloc[3][['Wins 2']].values[0]) + 1
        gp = int(bracket.iloc[3][['GP 2']].values[0]) + 1
        winpct = float(bracket.iloc[3][['Wins 2']].values[0] / bracket.iloc[3][['GP 2']].values[0])
        bracket.iat[3, bracket.columns.get_loc('Winner')] = ff4_team2.values[0]
        bracket.iat[32, bracket.columns.get_loc('Team 2')] = ff4_team2.values[0]
        bracket.iat[32, bracket.columns.get_loc('Offense 2')] = bracket.iloc[3][['Offense 2']].values[0]
        bracket.iat[32, bracket.columns.get_loc('Defense 2')] = bracket.iloc[3][['Defense 2']].values[0]
        bracket.iat[32, bracket.columns.get_loc('Wins 2')] = wins
        bracket.iat[32, bracket.columns.get_loc('GP 2')] = gp
        bracket.iat[32, bracket.columns.get_loc('Win Pct 2')] = winpct
        if int(ff4_pred[0]) == int(ff4_pred[1]):
            bracket.iat[3, bracket.columns.get_loc('Score 2')] = int(ff4_pred[1]) + 1
    else:
        wins = int(bracket.iloc[3][['Wins 1']].values[0]) + 1
        gp = int(bracket.iloc[3][['GP 1']].values[0]) + 1
        winpct = float(bracket.iloc[3][['Wins 1']].values[0] / bracket.iloc[3][['GP 1']].values[0])
        bracket.iat[3, bracket.columns.get_loc('Winner')] = ff4_team1.values[0]
        if int(ff4_pred[0]) == int(ff4_pred[1]):
            bracket.iat[3, bracket.columns.get_loc('Score 1')] = int(ff4_pred[0]) + 1
        bracket.iat[32, bracket.columns.get_loc('Team 2')] = ff4_team1.values[0]
        bracket.iat[32, bracket.columns.get_loc('Offense 2')] = bracket.iloc[3][['Offense 1']].values[0]
        bracket.iat[32, bracket.columns.get_loc('Defense 2')] = bracket.iloc[3][['Defense 1']].values[0]
        bracket.iat[32, bracket.columns.get_loc('Wins 2')] = wins
        bracket.iat[32, bracket.columns.get_loc('GP 2')] = gp
        bracket.iat[32, bracket.columns.get_loc('Win Pct 2')] = winpct
    '''

    #print(bracket.iloc[3])
    #print(bracket.iloc[8])

    #Round of 64
    r1s = 4
    r2s = 36
    for a in range(32):
        source = a + r1s
        r1data = bracket.iloc[source][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
        #print(r1data)
        r1pred = model.predict(np.asarray(r1data).reshape(1, -1))[0]
        r1_team1 = bracket.iloc[source][['Team 1']]
        r1_team2 = bracket.iloc[source][['Team 2']]
        bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(r1pred[0])
        bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(r1pred[1])
        if r1pred[0] < r1pred[1]:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = r1_team2.values[0]
            teamname = r1_team2.values[0]
            wins = int(bracket.iloc[source][['Wins 2']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 2']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 2']].values[0]
            offense = bracket.iloc[source][['Offense 2']].values[0]
            defense = bracket.iloc[source][['Defense 2']].values[0]
            winpct = float(bracket.iloc[source][['Wins 2']].values[0] / bracket.iloc[source][['GP 2']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 2']].values[0]
            if int(r1pred[0]) == int(r1pred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(r1pred[1]) + 1
        else:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = r1_team1.values[0]
            teamname = r1_team1.values[0]
            wins = int(bracket.iloc[source][['Wins 1']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 1']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 1']].values[0]
            offense = bracket.iloc[source][['Offense 1']].values[0]
            defense = bracket.iloc[source][['Defense 1']].values[0]
            winpct = float(bracket.iloc[source][['Wins 1']].values[0] / bracket.iloc[source][['GP 1']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 1']].values[0]
            if int(r1pred[0]) == int(r1pred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(r1pred[0]) + 1
        target = int(a / 2) + r2s
        oe = a % 2
        if oe:
            bracket.iat[target, bracket.columns.get_loc('Wins 2')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 2')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 2')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 2')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 2')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 2')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 2')] = winpct
        else:
            bracket.iat[target, bracket.columns.get_loc('Wins 1')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 1')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 1')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 1')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 1')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 1')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 1')] = winpct
        #print(bracket.iloc[target])

    #Round of 32
    sss = 52
    for b in range(16):
        source = b + r2s
        r2data = bracket.iloc[source][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
        r2pred = model.predict(np.asarray(r2data).reshape(1, -1))[0]
        r2_team1 = bracket.iloc[source][['Team 1']]
        r2_team2 = bracket.iloc[source][['Team 2']]
        bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(r2pred[0])
        bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(r2pred[1])
        if r2pred[0] < r2pred[1]:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = r2_team2.values[0]
            teamname = r2_team2.values[0]
            wins = int(bracket.iloc[source][['Wins 2']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 2']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 2']].values[0]
            offense = bracket.iloc[source][['Offense 2']].values[0]
            defense = bracket.iloc[source][['Defense 2']].values[0]
            winpct = float(bracket.iloc[source][['Wins 2']].values[0] / bracket.iloc[source][['GP 2']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 2']].values[0]
            if int(r2pred[0]) == int(r2pred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(r2pred[1]) + 1
        else:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = r2_team1.values[0]
            teamname = r2_team1.values[0]
            wins = int(bracket.iloc[source][['Wins 1']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 1']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 1']].values[0]
            offense = bracket.iloc[source][['Offense 1']].values[0]
            defense = bracket.iloc[source][['Defense 1']].values[0]
            winpct = float(bracket.iloc[source][['Wins 1']].values[0] / bracket.iloc[source][['GP 1']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 1']].values[0]
            if int(r2pred[0]) == int(r2pred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(r2pred[0]) + 1
        target = int(b / 2) + sss
        oe = b % 2
        if oe:
            bracket.iat[target, bracket.columns.get_loc('Wins 2')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 2')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 2')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 2')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 2')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 2')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 2')] = winpct
        else:
            bracket.iat[target, bracket.columns.get_loc('Wins 1')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 1')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 1')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 1')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 1')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 1')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 1')] = winpct
        #print(bracket.iloc[target])
    '''
    '''
    # Sweet 16
    ees = 60
    for c in range(8):
        source = c + sss
        ssdata = bracket.iloc[source][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
        sspred = model.predict(np.asarray(ssdata).reshape(1, -1))[0]
        ss_team1 = bracket.iloc[source][['Team 1']]
        ss_team2 = bracket.iloc[source][['Team 2']]
        bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(sspred[0])
        bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(sspred[1])
        if sspred[0] < sspred[1]:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = ss_team2.values[0]
            teamname = ss_team2.values[0]
            wins = int(bracket.iloc[source][['Wins 2']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 2']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 2']].values[0]
            offense = bracket.iloc[source][['Offense 2']].values[0]
            defense = bracket.iloc[source][['Defense 2']].values[0]
            winpct = float(bracket.iloc[source][['Wins 2']].values[0] / bracket.iloc[source][['GP 2']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 2']].values[0]
            if int(sspred[0]) == int(sspred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(sspred[1]) + 1
        else:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = ss_team1.values[0]
            teamname = ss_team1.values[0]
            wins = int(bracket.iloc[source][['Wins 1']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 1']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 1']].values[0]
            offense = bracket.iloc[source][['Offense 1']].values[0]
            defense = bracket.iloc[source][['Defense 1']].values[0]
            winpct = float(bracket.iloc[source][['Wins 1']].values[0] / bracket.iloc[source][['GP 1']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 1']].values[0]
            if int(sspred[0]) == int(sspred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(sspred[0]) + 1
        target = int(c / 2) + ees
        oe = c % 2
        if oe:
            bracket.iat[target, bracket.columns.get_loc('Wins 2')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 2')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 2')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 2')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 2')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 2')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 2')] = winpct
        else:
            bracket.iat[target, bracket.columns.get_loc('Wins 1')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 1')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 1')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 1')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 1')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 1')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 1')] = winpct
        #print(bracket.iloc[target])

    # Elite 8
    ffs = 64
    for d in range(4):
        source = d + ees
        eedata = bracket.iloc[source][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
        eepred = model.predict(np.asarray(eedata).reshape(1, -1))[0]
        ee_team1 = bracket.iloc[source][['Team 1']]
        ee_team2 = bracket.iloc[source][['Team 2']]
        bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(eepred[0])
        bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(eepred[1])
        if eepred[0] < eepred[1]:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = ee_team2.values[0]
            teamname = ee_team2.values[0]
            wins = int(bracket.iloc[source][['Wins 2']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 2']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 2']].values[0]
            offense = bracket.iloc[source][['Offense 2']].values[0]
            defense = bracket.iloc[source][['Defense 2']].values[0]
            winpct = float(bracket.iloc[source][['Wins 2']].values[0] / bracket.iloc[source][['GP 2']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 2']].values[0]
            if int(eepred[0]) == int(eepred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(eepred[1]) + 1
        else:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = ee_team1.values[0]
            teamname = ee_team1.values[0]
            wins = int(bracket.iloc[source][['Wins 1']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 1']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 1']].values[0]
            offense = bracket.iloc[source][['Offense 1']].values[0]
            defense = bracket.iloc[source][['Defense 1']].values[0]
            winpct = float(bracket.iloc[source][['Wins 1']].values[0] / bracket.iloc[source][['GP 1']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 1']].values[0]
            if int(eepred[0]) == int(eepred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(eepred[0]) + 1
        target = int(d / 2) + ffs
        oe = d % 2
        if oe:
            bracket.iat[target, bracket.columns.get_loc('Wins 2')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 2')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 2')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 2')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 2')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 2')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 2')] = winpct
        else:
            bracket.iat[target, bracket.columns.get_loc('Wins 1')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 1')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 1')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 1')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 1')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 1')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 1')] = winpct
        #print(bracket.iloc[target])

    # Final Four
    cs = 66
    for e in range(2):
        source = e + ffs
        ffdata = bracket.iloc[source][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
        ffpred = model.predict(np.asarray(ffdata).reshape(1, -1))[0]
        ff_team1 = bracket.iloc[source][['Team 1']]
        ff_team2 = bracket.iloc[source][['Team 2']]
        bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(ffpred[0])
        bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(ffpred[1])
        if ffpred[0] < ffpred[1]:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = ff_team2.values[0]
            teamname = ff_team2.values[0]
            wins = int(bracket.iloc[source][['Wins 2']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 2']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 2']].values[0]
            offense = bracket.iloc[source][['Offense 2']].values[0]
            defense = bracket.iloc[source][['Defense 2']].values[0]
            winpct = float(bracket.iloc[source][['Wins 2']].values[0] / bracket.iloc[source][['GP 2']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 2']].values[0]
            if int(ffpred[0]) == int(ffpred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(ffpred[1]) + 1
        else:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = ff_team1.values[0]
            teamname = ff_team1.values[0]
            wins = int(bracket.iloc[source][['Wins 1']].values[0]) + 1
            gp = int(bracket.iloc[source][['GP 1']].values[0]) + 1
            seed = bracket.iloc[source][['Seed 1']].values[0]
            offense = bracket.iloc[source][['Offense 1']].values[0]
            defense = bracket.iloc[source][['Defense 1']].values[0]
            winpct = float(bracket.iloc[source][['Wins 1']].values[0] / bracket.iloc[source][['GP 1']].values[0])
            #winpct = bracket.iloc[source][['Win Pct 1']].values[0]
            if int(ffpred[0]) == int(ffpred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(ffpred[0]) + 1
        target = int(e / 2) + cs
        oe = e % 2
        if oe:
            bracket.iat[target, bracket.columns.get_loc('Wins 2')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 2')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 2')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 2')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 2')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 2')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 2')] = winpct
        else:
            bracket.iat[target, bracket.columns.get_loc('Wins 1')] = wins
            bracket.iat[target, bracket.columns.get_loc('GP 1')] = gp
            bracket.iat[target, bracket.columns.get_loc('Seed 1')] = seed
            bracket.iat[target, bracket.columns.get_loc('Team 1')] = teamname
            bracket.iat[target, bracket.columns.get_loc('Offense 1')] = offense
            bracket.iat[target, bracket.columns.get_loc('Defense 1')] = defense
            bracket.iat[target, bracket.columns.get_loc('Win Pct 1')] = winpct
        #print(bracket.iloc[target])

    #Championship
    for e in range(2):
        source = cs
        cdata = bracket.iloc[source][['Seed 1', 'Seed 2', 'Offense 1', 'Offense 2', 'Defense 1', 'Defense 2', 'Win Pct 1', 'Win Pct 2']]
        cpred = model.predict(np.asarray(cdata).reshape(1, -1))[0]
        c_team1 = bracket.iloc[source][['Team 1']]
        c_team2 = bracket.iloc[source][['Team 2']]
        bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(cpred[0])
        bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(cpred[1])
        if cpred[0] < cpred[1]:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = c_team2.values[0]
            teamname = c_team2.values[0]
            if int(cpred[0]) == int(cpred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 2')] = int(cpred[1]) + 1
        else:
            bracket.iat[source, bracket.columns.get_loc('Winner')] = c_team1.values[0]
            if int(cpred[0]) == int(cpred[1]):
                bracket.iat[source, bracket.columns.get_loc('Score 1')] = int(cpred[0]) + 1
        print(bracket.iloc[source])

    #export results
    bracket.to_excel(output_file)

def main():
    parser.add_argument("-f", "--datafile", help="Training Data Filename")
    parser.add_argument("-o", "--outputfile", help="Output Filename")
    parser.add_argument("-y", "--years", help="Number of years used to train", type=int)
    parser.add_argument('-b', '--bracket', help="Whether to generate a bracket or simply test.")
    args = parser.parse_args()
    if args.datafile == None:
        d = "mm_train.xlsx"
    else:
        d = args.datafile
    if args.outputfile == None:
        o = "mm_preds.xlsx"
    else:
        o = args.outputfile
    if args.years == None:
        y = 1
    else:
        y = args.years
    generate_bracket()
    
if __name__ == "__main__":
    main()


