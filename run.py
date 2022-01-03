# %%
import json
import numpy as np
from core.data_loader import DataLoader
import xgboost as xgb
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

def train_xgb(config):
    print("."*10+"start train xgb"+"."*10)

    data = DataLoader(config["data"]["cols"],config["data"]["cat_cols"],config["training"]["split"])
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]

    reg = xgb.XGBRegressor(n_estimators = 100,max_depth=6)
    reg.fit(
        X_train,y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )
    pred = reg.predict(X_test)
    #print(pred)

    mape = MAPE(y_test, pred)
    rmse = RMSE(y_test, pred)
    print("mape:",mape)
    print("rmse:",rmse)

    plot_result(y_test.flatten(),pred,rmse)

def MAPE(true, pred):
    return round(np.mean(np.abs((true - pred) / true)) * 100,2)

def RMSE(true, pred):
    return round(sqrt(mean_squared_error(true, pred)))

def plot_result(true,pred,rmse):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    fig = plt.figure(facecolor="white",figsize=(12,8))
    ax = fig.add_subplot(111)
    plt.title("RMSE:"+str(rmse))
    plt.plot(true,label="True data", marker=".")
    plt.plot(pred,label="Predict data", marker=".")
    plt.legend()
    plt.show()

if __name__=="__main__":
    configs = json.load(open("config.json","r",encoding="utf-8-sig"))
    train_xgb(configs)
# %%
