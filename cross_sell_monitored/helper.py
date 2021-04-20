import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from xgboost import XGBClassifier
import statistics
import joblib

from a2ml.api.a2ml import A2ML, Context

target = "Response"

def split_to_train_test():
    df = pd.read_csv('../files/cross_sell.csv.gz')
    df.drop(['id'], inplace=True, axis=1)
    print(len(df))

    df = pd.get_dummies(df, prefix_sep="__", columns=['Gender','Vehicle_Age','Vehicle_Damage'], dummy_na=False)
    #print(df.columns)
    df.rename(columns={'Vehicle_Age__< 1 Year':'Vehicle_Age__more 1 Year',
        'Vehicle_Age__> 2 Years':'Vehicle_Age__less 2 Years'}, inplace=True)
    #print(df.columns)

    df_train, df_test = train_test_split(df, train_size=0.5, random_state=123,
        stratify=df[[target]], shuffle=True)

    df_train.to_csv('../files/cross_sell_train.csv.gz', index=False, encoding='utf-8', compression='gzip')
    df_test.to_csv('../files/cross_sell_test.csv.gz', index=False, encoding='utf-8', compression='gzip')

def build_model(train_path, df_data=None):
    df = pd.read_csv(train_path)
    if df_data is not None:
        df.append(df_data)

    df_train, df_test = train_test_split(df, train_size=0.8, random_state=123,
        stratify=df[[target]], shuffle=True)

    y_train = df_train[target]
    X_train = df_train
    del X_train[target]

    y_test = df_test[target]
    X_test = df_test
    del X_test[target]

    model = XGBClassifier(
        max_depth=8,
        n_estimators=200,
        min_child_weight=30, 
        colsample_bytree=0.8, 
        subsample=0.8, 
        eta=0.3,    
        random_state=42
    )
    model.fit(X_train, y_train)    

    fi = model.feature_importances_
    print(fi)

    predictions = model.predict(X_test)
    score = precision_score(y_test, predictions)
    print(score)

    y_train = df[target]
    X_train = df
    del X_train[target]

    model.fit(X_train, y_train)

    joblib.dump(model, "../models/cross_sell_xgboost.pkl")

    return "XGBClassifier", score

def deploy_monitored_model(name, algorithm, score):
    ctx = Context()
    a2ml = A2ML(ctx)

    return a2ml.deploy(model_id=None, name=name, algorithm=algorithm, score=score)

