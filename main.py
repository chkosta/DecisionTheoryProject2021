import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def func(x):
    if not x.Outlet_Size:
        if x.Outlet_Type == 'Grocery Store':
            x.Outlet_Size == 'Small'
        elif x.Outlet_Location_Type == 'Tier 1' and x.Outlet_Type == 'Supermarket Type1':
            x.Outlet_Size == 'Medium'
        elif x.Outlet_Location_Type == 'Tier 2' and x.Outlet_Type == 'Supermarket Type1':
            x.Outlet_Size == 'Small'
        elif x.Outlet_Location_Type == 'Tier 3' and x.Outlet_Type == 'Supermarket Type1':
            x.Outlet_Size == 'High'
        elif x.Outlet_Type == 'Supermarket Type2':
            x.Outlet_Size == 'Medium'
        elif x.Outlet_Type == 'Supermarket Type3':
            x.Outlet_Size == 'Medium'
    return(x)


def train_model(model, model_name):
    accuracy = {}
    rmse = {}
    explained_variance = {}
    max_error = {}
    MAE = {}

    print(model_name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = metrics.r2_score(y_test, pred) * 100
    print('R2_Score', acc)
    accuracy[model_name] = acc

    met = np.sqrt(metrics.mean_squared_error(y_test, pred))
    print('RMSE : ', met)
    rmse[model_name] = met

    var = (metrics.explained_variance_score(y_test, pred))
    print('Explained_Variance : ', var)
    explained_variance[model_name] = var

    error = (metrics.max_error(y_test, pred))
    print('Max_Error : ', error)
    max_error[model_name] = error

    err = metrics.mean_absolute_error(y_test, pred)
    print("Mean Absolute Error", err)
    MAE[model_name] = err


desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)



if __name__ == '__main__':

    # Fortonoume ta dedomena
    data = pd.read_csv("Train_UWu5bXk.csv")
    data.iloc[:,5:12].head()
    x = data.info()
    print("\n")
    x = data.describe(include='all').transpose()


    # Check for categorical attributes
    # cat_col = []
    # for x in data.dtypes.index:
    #     if data.dtypes[x] == 'object':
    #         cat_col.append(x)
    # print(cat_col)
    # print("\n")

    # Convert column type into their correct type
    # data.Item_Identifier = data.Item_Identifier.astype('category')
    # data.Item_Fat_Content = data.Item_Fat_Content.astype('category')
    # data.Item_Type = data.Item_Type.astype('category')
    # data.Outlet_Identifier = data.Outlet_Identifier.astype('category')
    # data.Outlet_Establishment_Year = data.Outlet_Establishment_Year.astype('int64')
    # data.Outlet_Type = data.Outlet_Type.astype('category')
    # data.Outlet_Location_Type = data.Outlet_Location_Type.astype('category')
    # data.Outlet_Size = data.Outlet_Size.astype('category')
    # x = data.info()
    # print("\n")


    # Check for missing values
    missing = 100*(data.isna().sum())/len(data)


    # Gemizoume tis elipis times tou Item_Weight me tin mesi timi
    data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

    missing = 100 * (data.isna().sum()) / len(data)

    print(data.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].value_counts())
    data.Outlet_Size = data.apply(func, axis=1)


    # Diorthwnoume tis times stin stili Item_Fat_Content
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('LF', 'Low Fat')
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('low fat', 'Low Fat')
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('reg', 'Regular')


    # Dimiourgoume ta dianusmata
    # X = data.drop(["Item_Outlet_Sales"], axis=1).values.reshape((-1, 1))
    # Y = data["Item_Outlet_Sales"].values.reshape((-1, 1))
    # Y = [float(x) for x in Y]
    X = data.select_dtypes(include=np.number).drop(["Item_Outlet_Sales"], axis=1)
    y = data["Item_Outlet_Sales"]

    # Use only one feature
    # X = data["Item_MRP"].values.reshape((-1, 1))

    # Kanoume ena arxiko plot twn dianismatwn
    # plt.figure(figsize=(16, 8))
    # plt.scatter(X, y, c='black')
    # plt.xlabel("Item_MRP")
    # plt.ylabel("SALES")
    # plt.show()


    # Coolerelation Matrix
    corr = data.select_dtypes(include=[np.number]).corr()
    print(sns.heatmap(corr, vmax=8, square=True))


    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


    # Model Training
    reg = LinearRegression()
    dtr = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
    rfc = ensemble.RandomForestRegressor(n_estimators=400, bootstrap=True, min_samples_leaf=10)
    gbr = GradientBoostingRegressor()

    # Train the model
    # reg.fit(X_train, y_train)
    #
    # # Predicting the test set result
    # pred = reg.predict(X_test)
    #
    # # Plot outputs
    # plt.figure(figsize=(16, 8))
    # plt.scatter(X_test, y_test, c='black')
    # plt.plot(X_test, pred, c='blue', linewidth=2)
    # plt.xlabel("Item MRP")
    # plt.ylabel("SALES")
    # plt.show()


    print("\n")
    train_model(reg, "Linear Regression")
    print("-" * 30)
    train_model(dtr, "Decision Tree")
    print("-" * 30)
    train_model(rfc, "Random Forest")
    print("-" * 30)
    train_model(gbr, "Gradient Boosting Regression")
