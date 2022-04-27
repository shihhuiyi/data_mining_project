# %%
from datetime import date
import pandas as pd
import numpy as np
from core.data_processor import standard,transform
from sklearn.preprocessing import MinMaxScaler

class DataLoader():

    def __init__(self, cols, cat_cols, split):
        df = pd.read_csv("data/data.csv")
        df = df[df["掛號日期"].notna()]
        df = df[df["發照日期"].notna()]
        df["掛號"] = pd.to_datetime(transform(df["掛號日期"]))
        df["發照"] = pd.to_datetime(transform(df["發照日期"]))
        df["time"] = (df["發照"]-df["掛號"]).dt.days
        IQR_t = df["time"].quantile(0.75)-df["time"].quantile(0.25)
        IQR_t_u = df["time"].quantile(0.75)+1.5*IQR_t
        IQR_t_l = df["time"].quantile(0.25)-1.5*IQR_t
        IQR_p = df["工程造價"].quantile(0.75)-df["工程造價"].quantile(0.25)
        IQR_p_u = df["工程造價"].quantile(0.75)+1.5*IQR_p
        IQR_p_l = df["工程造價"].quantile(0.25)-1.5*IQR_p
        df = df[(df["time"]<=IQR_t_u)&(df["time"]>=IQR_t_l)]
        df = df[(df["工程造價"]<=IQR_p_u)&(df["工程造價"]>=IQR_p_l)]

        self.df = df.get(cols).dropna(axis=0,how="any")

        for index in cat_cols:
            dff = pd.get_dummies(self.df[index],prefix=index)
            self.df = self.df.drop(index,axis=1)
            self.df = pd.concat([dff,self.df],axis=1)

        self.i_split = int(len(self.df)*split)
        self.data_train = self.df.values[:self.i_split]
        self.data_test = self.df.values[self.i_split:]

    def feature_select(self,cols):
        self.df = self.df.get(cols)
        self.data_train = self.df.values[:self.i_split]
        self.data_test = self.df.values[self.i_split:]

    def standard(self, data):
        sc = MinMaxScaler()
        sc.fit_transform(self.data_train[:,:-1])
        a = sc.transform(data)
        return a


        '''
        self.X_train = standard(data=self.data_train[:,:-1])
        self.y_train = self.data_train[:,-1]
        self.X_test = standard(data=self.data_test[:,:-1])
        self.y_test = self.data_test[:,-1]
        '''



    '''
    def creat_dataset(self, data, lookback=1):
        dataX, dataY = [], []
        for i in range(len(data)-lookback):
            x = data[i:i+lookback,:-1]
            y = data[i+lookback]
            dataX.append(x)
            dataY.append(y)


        return np.array(dataX), np.array(dataY).reshape(-1,1)
    '''
# %%
