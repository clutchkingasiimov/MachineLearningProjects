#Dependencies
import pandas as pd
import sklearn as sk
import numpy as np 
import matplotlib.pyplot as plt
import keras
import seaborn as sns
import gc
import xgboost
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

#Import the datasets
df_train = pd.read_csv('../input/train_V2.csv')
df_test = pd.read_csv('../input/test_V2.csv')

#Fix the missing value in the winPlacePerc column 
df_train['winPlacePerc'].fillna(0, inplace=True)

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    #Credits to Hyun Woo Kim's notebook 
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)



ids = df_test['Id'].copy()
df_test = df_test.drop(['Id'],axis=1)
#df_test = df_test.drop(['groupId','matchId'],axis=1)

#Some new features 
df_train['totalDistance'] = df_train['walkDistance'] + df_train['rideDistance'] + df_train['swimDistance']
df_test['totalDistance'] = df_test['walkDistance'] + df_test['rideDistance'] + df_test['swimDistance']
df_train['Logistics'] = df_train['heals'] + df_train['boosts']
df_test['Logistics'] = df_test['heals'] + df_test['boosts']
"""
XGBoost can easily work around sparse or missing data, hence the current format will not affect boosting in any way, but since 
there might be players with 0 headshots and 0 kills in total, the division can not happen and lead to the infinity bound. Hence
to fix the problem, an extra 1 is added in the denominator to negate the infinity bound.
"""
df_train['HeadshotRate'] = (df_train['headshotKills'] /(df_train['kills'] + 1)) + 1
df_test['HeadshotRate'] = (df_test['headshotKills'] / (df_test['kills'] + 1)) + 1


df_train['AvgKS'] = (df_train['killStreaks'] / (df_train['kills']))
df_test['AvgKS'] = (df_test['killStreaks'] / (df_test['kills']))

df_train['AvgKS'].fillna(0, inplace=True)
df_test['AvgKS'].fillna(0, inplace=True)


df_train['KillsPerSecond'] = df_train['kills'] / df_train['matchDuration']
df_test['KillsPerSecond'] = df_test['kills'] / df_test['matchDuration']

"""
In groups, each player will have a different standing since not all players in a group play on equal footing, but since the groups will be the 
same for some players, the scoring will also be the same. Let's find the average group scores.
"""
#First find the group sizes 
df_train_sizes = df_train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
df_test_sizes = df_test.groupby(['matchId','groupId']).size().reset_index(name='group_size')

#Next, obtain the group mean scores 
df_train_means = df_train.groupby(['matchId','groupId']).mean().reset_index()
df_test_means = df_test.groupby(['matchId','groupId']).mean().reset_index()

#The group max and mean 
#df_train_max = df_train.groupby(['matchId','groupId']).max().reset_index()
#df_test_max = df_test.groupby(['matchId','groupId']).max().reset_index()

#df_train_min = df_train.groupby(['matchId','groupId']).min().reset_index()
#df_test_min = df_test.groupby(['matchId','groupId']).min().reset_index()

"""
In the above code line, the goal was to find the average performance of a given group and the number of players in a given group. But because
group players of one particular group will perform differently than the group players of another group, there will be a slight variation in the
scores simply because of this feature also. Hence now to find the
"""

df_train_match_means = df_train.groupby(['matchId']).mean().reset_index()
df_test_match_means = df_test.groupby(['matchId']).mean().reset_index()


#Lastly, merge the datasets together in their respective dataframes

	#Mean merging
df_train = pd.merge(df_train, df_train_means, suffixes=["", "_means"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_means, suffixes=["", "_means"], how='left', on=['matchId', 'groupId'])
del df_train_means
del df_test_means

	#Match mean merging
df_train = pd.merge(df_train, df_train_match_means, suffixes=["", "_match_means"], how='left', on=['matchId'])
df_test = pd.merge(df_test, df_test_match_means, suffixes=["", "_match_means"], how='left', on=['matchId'])
del df_train_match_means
del df_test_match_means

	#Group size merging 
df_train = pd.merge(df_train, df_train_sizes, how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_sizes, how='left', on=['matchId', 'groupId'])
del df_train_sizes
del df_test_sizes

	#Max merging 
#df_train = pd.merge(df_train, df_train_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
#df_test = pd.merge(df_test, df_test_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
#del df_train_max
#del df_test_max

	#Min merging 
#df_train = pd.merge(df_train, df_train_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
#df_test = pd.merge(df_test, df_test_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
#del df_train_min
#del df_test_min


df_train = df_train.drop(['Id', 'groupId','matchId','matchType','rankPoints'],axis=1)
df_test = df_test.drop(['groupId','matchId','matchType','rankPoints'],axis=1)

#df_train['winPlacePerc_match_means'].fillna(0, inplace=True)
#df_train['winPlacePerc_means'].fillna(0, inplace=True)


#Models 
from xgboost import XGBRegressor
import lightgbm as lgb 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train = df_train.drop(['winPlacePerc','winPlacePerc_means','winPlacePerc_match_means'],axis=1)
y_train = df_train['winPlacePerc']
X_test = df_test
#Convert all features to same dtype
#X_train = np.array(X_train, dtype=np.float64)
#X_test = np.array(X_test, dtype=np.float64)
#y_train = np.array(y_train, dtype=np.float64)


scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train*2 - 1



from keras import Sequential
from keras.layers import Dense
from keras import regularizers 

#Define the layers and the parameters (ReLU for regression)
model = Sequential()
model.add(Dense(35, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(30, activation='relu'))
model.add(Dense(22, activation='relu'))
#model.add(Dense(17, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mse']) #Adam 
model.fit(X_train, y_train, epochs=350, batch_size=741161)
predictions = model.predict(X_test)

#linreg = LinearRegression(fit_intercept=True)
#linreg.fit(X_train, y_train)
#predictions = linreg.predict(X_test)
"""
params = {
    'boosting_type':'gbdt',
    'learning_rate': 0.05,
    'max_depth': -1,
    'num_leaves' : 40,
    'subsample' : 0.8,
    'objective' : 'regression',
    'metric' : 'mae'}
trainer = lgb.Dataset(X_train, y_train)
lgb_model = lgb.train(params, train_set = trainer, num_boost_round=10000)
predictions = lgb_model.predict(X_test)
"""
predictions[predictions > 1] = 1
predictions[predictions < 0] = 0 


submission = pd.DataFrame(columns=['Id','winPlacePerc'])
submission['Id'] = ids
submission['winPlacePerc'] = predictions
submission.to_csv('submission.csv',index=False)