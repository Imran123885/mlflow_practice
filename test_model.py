import mlflow
logged_model = 'runs:/<RUN_ID>/RandomForestClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")

### WILL NOT WORK UNLESS I SUCCESSFULLY TRAIN RANDOM FOREST MODEL AND GET SAVED MODEL (SAVED RUN ID), CAN'T RN BECAUSE THIS COMPUTER IS RLLY SLOW ###