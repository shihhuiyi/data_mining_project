# %%
import json
import numpy as np
import pandas as pd
from core.data_loader import DataLoader
import xgboost as xgb
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression

def train_xgb(config):
    print("."*10+"start select feature"+"."*10)

    data = DataLoader(config["data"]["cols"],config["data"]["cat_cols"],config["training"]["split"])
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)
    # print("x train: \n", X_train)
    # print("x test: \n", X_test)
    # a = pd.DataFrame(X_train)
    # b = pd.DataFrame(X_test)
    # c = pd.concat([a,b],axis=0)
    # c.columns = data.df.columns[:-1]
    # c.to_csv("standard_data.csv", encoding="utf_8_sig")

    reg = xgb.XGBRegressor(n_estimators = 100)
    reg.fit(
        X_train,y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )

    feature = plot_feature(reg.feature_importances_, data.df.columns[:-1])
    feature = np.append(feature["feature"].values,"time").flatten()
    #print("特徵值：\n",feature)

    print("."*10+"start train xgb"+"."*10)

    data.feature_select(feature)
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)

    reg = xgb.XGBRegressor(n_estimators = 100)
    reg.fit(
        X_train,y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )
    pred = reg.predict(X_test)

    mape = MAPE(y_test, pred)
    rmse = RMSE(y_test, pred)
    print("mape:",mape)
    print("rmse:",rmse)

    plot_result("xgboost",y_test.flatten(),pred,rmse,mape)

def train_rf(config):
    print("."*10+"start select feature"+"."*10)

    data = DataLoader(config["data"]["cols"],config["data"]["cat_cols"],config["training"]["split"])
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)

    reg = xgb.XGBRegressor(n_estimators = 100)
    reg.fit(
        X_train,y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )

    feature = plot_feature(reg.feature_importances_, data.df.columns[:-1])
    feature = np.append(feature["feature"].values,"time").flatten()
    #print("特徵值：\n",feature)

    print("."*10+"start train random forest"+"."*10)

    data.feature_select(feature)
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)

    reg = RandomForestRegressor(n_estimators=100,max_depth=6)
    reg.fit(X_train,y_train)
    pred = reg.predict(X_test)

    mape = MAPE(y_test, pred)
    rmse = RMSE(y_test, pred)
    print("mape:",mape)
    print("rmse:",rmse)

    plot_result("random forest",y_test.flatten(),pred,rmse,mape)

def train_knn(config):
    print("."*10+"start select feature"+"."*10)

    data = DataLoader(config["data"]["cols"],config["data"]["cat_cols"],config["training"]["split"])
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)

    reg = xgb.XGBRegressor(n_estimators = 100)
    reg.fit(
        X_train,y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )

    feature = plot_feature(reg.feature_importances_, data.df.columns[:-1])
    feature = np.append(feature["feature"].values,"time").flatten()
    #print("特徵值：\n",feature)

    print("."*10+"start train knn"+"."*10)

    data.feature_select(feature)
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)

    reg = KNeighborsRegressor()
    reg.fit(X_train,y_train)
    pred = reg.predict(X_test)

    mape = MAPE(y_test, pred)
    rmse = RMSE(y_test, pred)
    print("mape:",mape)
    print("rmse:",rmse)

    plot_result("knn",y_test.flatten(),pred,rmse,mape)

def train_lr(config):
    print("."*10+"start select feature"+"."*10)

    data = DataLoader(config["data"]["cols"],config["data"]["cat_cols"],config["training"]["split"])
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)

    reg = xgb.XGBRegressor(n_estimators = 100)
    reg.fit(
        X_train,y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )

    feature = plot_feature(reg.feature_importances_, data.df.columns[:-1])
    feature = np.append(feature["feature"].values,"time").flatten()
    #print("特徵值：\n",feature)

    print("."*10+"start train logistic regression"+"."*10)

    data.feature_select(feature)
    X_train, y_train = data.data_train[:,:-1],data.data_train[:,-1]
    X_test, y_test = data.data_test[:,:-1],data.data_test[:,-1]
    X_train = data.standard(X_train)
    X_test = data.standard(X_test)

    reg = LogisticRegression()
    reg.fit(X_train,y_train)
    pred = reg.predict(X_test)

    mape = MAPE(y_test, pred)
    rmse = RMSE(y_test, pred)
    print("mape:",mape)
    print("rmse:",rmse)

    plot_result("logistic regression",y_test.flatten(),pred,rmse,mape)


def MAPE(true, pred):
    return round(np.mean(np.abs((true - pred) / true)) * 100,2)

def RMSE(true, pred):
    return round(sqrt(mean_squared_error(true, pred)))

def plot_result(model,true,pred,rmse,mape):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    fig = plt.figure(facecolor="white",figsize=(12,8))
    #ax = fig.add_subplot(111)
    plt.title(str(model)+" RMSE:"+str(rmse)+" MAPE:"+str(mape))
    plt.plot(true,label="True data", marker=".")
    plt.plot(pred,label="Predict data", marker=".")
    plt.xlabel("樣本")
    plt.ylabel("發照天數")
    plt.legend()
    #plt.show()
    plt.savefig("plot/{}.png".format(str(model)))

def plot_feature(feature_importance,label):
    print("特徵值數量：%d\n" %len(feature_importance))
    feature = pd.DataFrame()
    feature["feature"] = label
    feature["importance"] = feature_importance
    feature = feature.sort_values("importance",ascending=False)
    feature.to_csv("feature/feature_importance.csv", index_label=0)
    feature = feature[feature["importance"]!=0]
    print("選取特徵值數量：%d\n" %len(feature))
    feature.to_csv("feature/feature.csv", index_label=0)
    #fig = plt.figure(figsize=(100,20))
    #plt.bar(range(len(feature_importance)),feature_importance)
    #plt.xticks(range(len(feature_importance)),label)
    #plt.show
    return feature

if __name__=="__main__":
    configs = json.load(open("config.json","r",encoding="utf-8-sig"))
    train_xgb(configs)
    train_rf(configs)
    train_knn(configs)
    train_lr(configs)
# %%
