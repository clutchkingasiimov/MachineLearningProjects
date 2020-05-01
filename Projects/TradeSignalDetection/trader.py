from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.metrics import plot_confusion_matrix, classification_report
import ta
import numpy as np 
import pandas as pd 



class PCARF():
    commission = 0.03 
    tax_rate = 0.025 

    def __init__(self, data):
        self.data = data 
        self.indicators = {
            "MA": self.data["Adj Close"].rolling(20).mean(),
            "EMA": ta.trend.EMAIndicator(self.data["Adj Close"], n=20).ema_indicator(),
            "MACD_line": ta.trend.MACD(self.data["Adj Close"], n_slow=26, n_fast=12,
                                n_sign=9).macd(),
            "MACD_signal": ta.trend.MACD(self.data["Adj Close"], n_slow=26, n_fast=12,
                                n_sign=9).macd_signal(),
            "OBV": ta.volume.OnBalanceVolumeIndicator(self.data["Adj Close"], self.data["Volume"]).on_balance_volume(),
            "RSI": ta.momentum.rsi(self.data["Adj Close"]),
            "STOCHASTIC": ta.momentum.stoch(self.data["High"], self.data["Low"], self.data["Adj Close"]),
            "ELDER": ta.volume.force_index(self.data["Adj Close"], self.data["Volume"]),
            "ATR": ta.volatility.average_true_range(self.data["High"], self.data["Low"], self.data["Adj Close"]),
            "CCI": ta.trend.CCIIndicator(self.data["High"], self.data["Low"], self.data["Adj Close"]).cci()
        }

    def generate_data(self, rate_of_change, alpha):
        rules = [0]
        for i in range(len(rate_of_change)):
            if rate_of_change[i] >= alpha: #Strong buy 
                rules.append(1)
            elif 0 <= rate_of_change[i] < alpha: #Normal buy 
                rules.append(2) 
            elif -alpha <= rate_of_change[i] < 0: #Normal sell 
                rules.append(3) 
            elif rate_of_change[i] < -alpha: #Strong sell
                rules.append(4) 

        assert len(rules) == len(self.data) #Assertion check 
        self.data['Signals'] = rules 

        #Adding indicators to the self.data matrix 
        self.data["MA"] = self.indicators["MA"]
        self.data["EMA"] = self.indicators["EMA"]
        self.data["MACD_line"] = self.indicators["MACD_line"]
        self.data["MACD_signal"] = self.indicators["MACD_signal"]
        self.data["OBV"] = self.indicators["OBV"]
        self.data["RSI"] = self.indicators["RSI"]
        self.data["STOCHASTIC"] = self.indicators["STOCHASTIC"]
        self.data["ELDER"] = self.indicators["ELDER"]
        self.data["ATR"] = self.indicators["ATR"]
        self.data["CCI"] = self.indicators["CCI"]

        #Remove NaN rows
        data_new = self.data.dropna(axis=0)
        return data_new

    def train_model(self, data):
        X = data.drop("Signals", axis=1).to_numpy()
        y = data['Signals']

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        pca = PCA()

        pipeline = Pipeline([("scaler", scaler),("pca",pca)])
        pc_components = pipeline.fit_transform(X_train, y_train)
        pc_components_test = pipeline.transform(X_test)

        rfr = RandomForestClassifier(n_estimators=100)
        rfr.fit(pc_components, y_train)
        predictions = rfr.predict(pc_components_test)

        #Obtain imbalanced accuracy and balanced accuracy
        imb_acc = accuracy_score(predictions, y_test)
        bal_acc = balanced_accuracy_score(predictions, y_test)
        print("*"*60)
        print("Imbalanced accuracy score: {}".format(imb_acc))
        print("Balanced accuracy score: {}".format(bal_acc))
        print("*"*60)

        #Prepare the predicted dataset 
        prices = self.data['Close'][len(self.data)-len(y_test):]
        new_df = pd.DataFrame([np.array(prices), predictions, y_test]).T
        new_df = new_df.rename(columns={0:"Close", 1:"preds", 2:"actual"})

        return new_df 

    def generate_trade_strategy(self, signal_data, share_count=int, balance=int):

        """
        Commission applies on buy and sell orders 
        Tax deduction only applies on sell orders 
        """
        if balance < 10000:
            raise ValueError("Balance needs to be more than 10K")
        if share_count < 10:
            raise ValueError("Share trade needs to be 10 for a minimum")

        half_buy = round(share_count/2) #Half shares taken by (2, 3)
        shares = [0]
        equity = [balance]
        transaction_costs = [0]
        gains = []
        close = signal_data['Close']
        signals = signal_data['preds']
        net_gain = [close[i+1] - close[i] for i in range(len(close)-1)]

        for price, signal, index in zip(close, signals, range(len(close)-1)):
            if signal == 1:
                transaction_cost = share_count * (price * (1+self.commission)) #Transaction cost of 10 shares
                transaction_costs.append(transaction_cost) #Transaction cost including commission 
                gain = (net_gain[index]*share_count) - (net_gain[index]*share_count*self.commission) 
                equity.append(equity[index]+gain) #Update equity size 
                shares.append(shares[index]+share_count) #Increment 10 shares in portfolio    
                gains.append(gain)

            if signal == 2: #Normal buy 
                transaction_cost = half_buy * (price * (1+self.commission))
                gain = (net_gain[index]*half_buy) - (net_gain[index]*half_buy*self.commission) 
                equity.append(equity[index] + gain)
                shares.append(shares[index]+half_buy)
                gains.append(gain)

            if signal == 3:
                transaction_cost = half_buy * (price * (1+self.commission)) * (1 + self.tax_rate)
                gain = (net_gain[index]*half_buy) - (net_gain[index]*half_buy*self.commission)  - (net_gain[index]*half_buy*self.tax_rate)
                equity.append(equity[index] + gain)
                shares.append(shares[index]-half_buy)
                gains.append(gain)

            if signal == 4:
                transaction_cost = shares[index] * (price * (1+self.commission)) * (1+self.tax_rate)#Sell all shares
                transaction_costs.append(transaction_cost)
                gain = (net_gain[index]*share_count) - (net_gain[index]*share_count*self.commission) -(net_gain[index]*share_count*self.tax_rate)
                equity.append(equity[index] + (net_gain[index]*shares[index]))
                shares.append(shares[index]-shares[index])
                gains.append(gain)
                
        return equity, shares, gains


class RF(PCARF):

    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def generate_data(self, roc, alpha):
        return super().generate_data(roc, alpha)
    
    def train_model(self, train_data):
        X = train_data.drop("Signals", axis=1).to_numpy()
        y = train_data['Signals']

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        rfr = RandomForestClassifier(n_estimators=100)
        rfr.fit(X_train, y_train)
        predictions = rfr.predict(X_test)

        #Obtain imbalanced accuracy and balanced accuracy
        imb_acc = accuracy_score(predictions, y_test)
        bal_acc = balanced_accuracy_score(predictions, y_test)
        print("Imbalanced accuracy score: {}".format(imb_acc))
        print("Balanced accuracy score: {}".format(bal_acc))

        prices = self.data['Close'][len(self.data)-len(y_test):]
        new_df = pd.DataFrame([np.array(prices), predictions, y_test]).T
        new_df = new_df.rename(columns={0:"Close", 1:"preds", 2:"actual"})

        return new_df 

    def trade_strategy_rf(self, signal_data, share_count=int, balance=int):
        return super().generate_trade_strategy(signal_data, share_count, balance)


class PCABag(PCARF):

    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def generate_data(self, roc, alpha):
        return super().generate_data(roc, alpha)

    def train_model(self, train_data):
        X = train_data.drop("Signals", axis=1).to_numpy()
        y = train_data['Signals']

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        pca = PCA()

        pipeline = Pipeline([("scaler", scaler),("pca",pca)])
        pc_components = pipeline.fit_transform(X_train, y_train)
        pc_components_test = pipeline.transform(X_test)

        bagged_rf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=100),
        n_estimators=30)
        bagged_rf.fit(pc_components, y_train)
        predictions = bagged_rf.predict(pc_components_test)

        imb_acc = accuracy_score(predictions, y_test)
        bal_acc = balanced_accuracy_score(predictions, y_test)
        print("Imbalanced accuracy score: {}".format(imb_acc))
        print("Balanced accuracy score: {}".format(bal_acc))

        prices = self.data['Close'][len(self.data)-len(y_test):]
        new_df = pd.DataFrame([np.array(prices), predictions, y_test]).T
        new_df = new_df.rename(columns={0:"Close", 1:"preds", 2:"actual"})

        return new_df 

    def trade_strategy_svc(self, signal_data, share_count=int, balance=bool):
        return super().generate_trade_strategy(signal_data, share_count, balance)

    
class SVCModel(PCARF):

    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def generate_data(self, roc, alpha):
        return super().generate_data(roc, alpha)

    def train_model(self, train_data):
        X = train_data.drop("Signals", axis=1).to_numpy()
        y = train_data['Signals']

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
        svc = SVC(C=1.0, gamma=0.005)
        svc.fit(X_train, y_train)
        predictions = svc.predict(X_test)

        imb_acc = accuracy_score(predictions, y_test)
        bal_acc = balanced_accuracy_score(predictions, y_test)
        print("Imbalanced accuracy score: {}".format(imb_acc))
        print("Balanced accuracy score: {}".format(bal_acc))

        prices = self.data['Close'][len(self.data)-len(y_test):]
        new_df = pd.DataFrame([np.array(prices), predictions, y_test]).T
        new_df = new_df.rename(columns={0:"Close", 1:"preds", 2:"actual"})

        return new_df 

    def trade_strategy_svc(self, signal_data, share_count=int, balance=bool):
        return super().generate_trade_strategy(signal_data, share_count, balance)





        







