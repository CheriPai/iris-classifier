import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


VAL_PROP = 0.10

def process_data(file_path):
    df = pd.read_csv(file_path).drop("Id", 1)
    df = df.replace("Iris-setosa", 0)
    df = df.replace("Iris-versicolor", 1)
    df = df.replace("Iris-virginica", 2)

    X = df.drop("Species", 1).as_matrix()
    y = df["Species"].as_matrix()
    X, y = shuffle(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_PROP)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return dtrain, dval




if __name__ == "__main__":
    dtrain, dval = process_data("data/Iris.csv")
    param = {"bst:max_depth":2, "bst:eta":1, "silent":1, "objective":"multi:softmax", "num_class":3}
    plst = param.items()
    evallist = [(dval, "eval"), (dtrain, "train")]
    bst = xgb.train(plst, dtrain, 10, evallist)
    bst.dump_model("data/model.txt")
