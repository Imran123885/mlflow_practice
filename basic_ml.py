import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e

def eval_function(ground_truth, preds):
    rmse = np.sqrt(mean_squared_error(ground_truth, preds))

def main(alpha, l1_ratio):
    df = load_data()
    TARGET = 'quality' 
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

    mlflow.set_experiment("ML-Model-1")
    #with mlflow.start_run(run_name='Example Demo'):
    with mlflow.start_run():
        mlflow.set_tag("version","1.0.0")
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio) # key, value

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=6)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        metric = eval(p1 = inp1, p2 = inp2)
        mlflow.log_metric("Eval_Metric",metric)
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(f"Artifact created at {time.asctime()}")
        mlflow.log_artifacts("dummy")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-a", type=float, default=0.2)
    args.add_argument("--l1_ratio","-l1", type=float, default=0.3)
    parsed_args = args.parse_args()
    # parsed_args.param1
    main(parsed_args.alpha, parsed_args.l1_ratio)