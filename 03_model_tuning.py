#Update this VERSION_NUM to version your features, models etc!
VERSION_NUM = 'TEST_ML_JOB'
DB = "E2E_SNOW_MLOPS_DB" 
SCHEMA = "MLOPS_SCHEMA" 
COMPUTE_WAREHOUSE = "E2E_SNOW_MLOPS_WH" 


import pandas as pd
import numpy as np
import sklearn
import math
import pickle
import shap
from datetime import datetime
import streamlit as st
from xgboost import XGBClassifier

# Snowpark ML
from snowflake.ml.registry import Registry
from snowflake.ml.modeling.tune import get_tuner_context
from snowflake.ml.modeling import tune
from entities import search_algorithm
import snowflake.ml.modeling.preprocessing as snowml


#Snowflake feature store
from snowflake.ml.feature_store import FeatureStore, FeatureView, Entity, CreationMode

# Snowpark session
from snowflake.snowpark import DataFrame
from snowflake.snowpark.functions import col, to_timestamp, min, max, month, dayofweek, dayofyear, avg, date_add, sql_expr
from snowflake.snowpark.types import IntegerType, StringType
from snowflake.snowpark import Window


#set up snowpark session
from snowflake.snowpark.context import get_active_session
session = get_active_session()


X_train = train.drop("MORTGAGERESPONSE", "TIMESTAMP", "LOAN_ID")
y_train = train.select("MORTGAGERESPONSE")
X_test = test.drop("MORTGAGERESPONSE","TIMESTAMP", "LOAN_ID")
y_test = test.select("MORTGAGERESPONSE")


from snowflake.ml.data import DataConnector
from snowflake.ml.modeling.tune import get_tuner_context
from snowflake.ml.modeling import tune
from entities import search_algorithm

#Define dataset map
dataset_map = {
    "x_train": DataConnector.from_dataframe(X_train),
    "y_train": DataConnector.from_dataframe(y_train),
    "x_test": DataConnector.from_dataframe(X_test),
    "y_test": DataConnector.from_dataframe(y_test)
    }


# Define a training function, with any models you choose within it.
def train_func():
    # A context object provided by HPO API to expose data for the current HPO trial
    tuner_context = get_tuner_context()
    config = tuner_context.get_hyper_params()
    dm = tuner_context.get_dataset_map()

    model = XGBClassifier(**config, random_state=42)
    model.fit(dm["x_train"].to_pandas().sort_index(), dm["y_train"].to_pandas().sort_index())
    f1_metric = f1_score(
        dm["y_train"].to_pandas().sort_index(), model.predict(dm["x_train"].to_pandas().sort_index())
    )
    tuner_context.report(metrics={"f1_score": f1_metric}, model=model)

tuner = tune.Tuner(
    train_func=train_func,
    search_space={
        "max_depth": tune.randint(1, 10),
        "learning_rate": tune.uniform(0.01, 0.1),
        "n_estimators": tune.randint(50, 100),
    },
    tuner_config=tune.TunerConfig(
        metric="f1_score",
        mode="max",
        search_alg=search_algorithm.RandomSearch(random_state=101),
        num_trials=8, #run 8 trial runs
        max_concurrent_trials=4, #run 4 trials at a time
    ),
)


#Train several model candidates (note this may take 1-2 minutes)
tuner_results = tuner.run(dataset_map=dataset_map)


#Select best model results and inspect configuration
tuned_model = tuner_results.best_model
tuned_model


#Generate predictions
xgb_opt_preds = tuned_model.predict(train_pd.drop(["TIMESTAMP", "LOAN_ID", "MORTGAGERESPONSE"],axis=1))

#Generate performance metrics
f1_opt_train = round(f1_score(train_pd.MORTGAGERESPONSE, xgb_opt_preds),4)
precision_opt_train = round(precision_score(train_pd.MORTGAGERESPONSE, xgb_opt_preds),4)
recall_opt_train = round(recall_score(train_pd.MORTGAGERESPONSE, xgb_opt_preds),4)

print(f'Train Results: \nF1: {f1_opt_train} \nPrecision {precision_opt_train} \nRecall: {recall_opt_train}')


#Generate test predictions
xgb_opt_preds_test = tuned_model.predict(test_pd.drop(["TIMESTAMP", "LOAN_ID", "MORTGAGERESPONSE"],axis=1))

#Generate performance metrics on test data
f1_opt_test = round(f1_score(test_pd.MORTGAGERESPONSE, xgb_opt_preds_test),4)
precision_opt_test = round(precision_score(test_pd.MORTGAGERESPONSE, xgb_opt_preds_test),4)
recall_opt_test = round(recall_score(test_pd.MORTGAGERESPONSE, xgb_opt_preds_test),4)

print(f'Test Results: \nF1: {f1_opt_test} \nPrecision {precision_opt_test} \nRecall: {recall_opt_test}')

 
# # Here we see the HPO model has a more modest train accuracy than our base model - but the peformance doesn't drop off during testing


#Log the optimized model to the model registry (if not already there)
optimized_version_name = 'XGB_Optimized'

try:
    #Check for existing model
    mv_opt = model_registry.get_model(model_name).version(optimized_version_name)
    print("Found existing model version!")
except:
    #Log model to registry
    print("Logging new model version...")
    mv_opt = model_registry.log_model(
        model_name=model_name,
        model=tuned_model, 
        version_name=optimized_version_name,
        sample_input_data = train.drop(["TIMESTAMP", "LOAN_ID", "MORTGAGERESPONSE"]).limit(100),
        comment = f"""HPO ML model for predicting loan approval likelihood.
            This model was trained using XGBoost classifier.
            Optimized hyperparameters used were:
            max_depth={tuned_model.max_depth}, 
            n_estimators={tuned_model.n_estimators}, 
            learning_rate = {tuned_model.learning_rate}, 
            """,
        target_platforms= ["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        options= {"enable_explainability": True}

        

    )
    #Set metrics
    mv_opt.set_metric(metric_name="Train_F1_Score", value=f1_opt_train)
    mv_opt.set_metric(metric_name="Train_Precision_Score", value=precision_opt_train)
    mv_opt.set_metric(metric_name="Train_Recall_score", value=recall_opt_train)

    mv_opt.set_metric(metric_name="Test_F1_Score", value=f1_opt_test)
    mv_opt.set_metric(metric_name="Test_Precision_Score", value=precision_opt_test)
    mv_opt.set_metric(metric_name="Test_Recall_score", value=recall_opt_test)


#Here we see the BASE version is our default version
model_registry.get_model(model_name).default


#Now we'll set the optimized model to be the default model version going forward
model_registry.get_model(model_name).default = optimized_version_name


#Now we see our optimized version we have now recently promoted to our DEFAULT model version
model_registry.get_model(model_name).default


#we'll now update the PROD tagged model to be the optimized model version rather than our overfit base version
m.unset_tag("PROD")
m.set_tag("PROD", optimized_version_name)
m.show_tags()