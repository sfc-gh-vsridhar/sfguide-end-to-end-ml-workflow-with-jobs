# # ❄️ End-to-end ML Demo ❄️
# 
# In this worfklow we will work through the following elements of a typical tabular machine learning pipeline.
# 
# ### 1. Use Feature Store to track engineered features
# * Store feature defintions in feature store for reproducible computation of ML features
#       
# ### 2. Train two Models using the Snowflake ML APIs
# * Baseline XGboost
# * XGboost with optimal hyper-parameters identified via Snowflake ML distributed HPO methods
# 
# ### 3. Register both models in Snowflake model registry
# * Explore model registry capabilities such as **metadata tracking, inference, and explainability**
# * Compare model metrics on train/test set to identify any issues of model performance or overfitting
# * Tag the best performing model version as 'default' version
# ### 4. Set up Model Monitor to track 1 year of predicted and actual loan repayments
# * **Compute performance metrics** such a F1, Precision, Recall
# * **Inspect model drift** (i.e. how much has the average predicted repayment rate changed day-to-day)
# * **Compare models** side-by-side to understand which model should be used in production
# * Identify and understand **data issues**
# 
# ### 5. Track data and model lineage throughout
# * View and understand
#   * The **origin of the data** used for computed features
#   * The **data used** for model training
#   * The **available model versions** being monitored


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

#set up feature store 
fs = FeatureStore(
    session=session, 
    database=DB, 
    name=SCHEMA, 
    default_warehouse=COMPUTE_WAREHOUSE,
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST
)

ds = fs.retrieve_dataset(
    name=f"MORTGAGE_DATASET_EXTENDED_FEATURES_{VERSION_NUM}"
    )

ds_sp = ds.read.to_snowpark_dataframe()


OHE_COLS = ds_sp.select([col.name for col in ds_sp.schema if col.datatype ==StringType()]).columns
OHE_POST_COLS = [i+"_OHE" for i in OHE_COLS]

# Encode categoricals to numeric columns
snowml_ohe = snowml.OneHotEncoder(input_cols=OHE_COLS, output_cols = OHE_COLS, drop_input_cols=True)
ds_sp_ohe = snowml_ohe.fit(ds_sp).transform(ds_sp)

#Rename columns to avoid double nested quotes and white space chars
rename_dict = {}
for i in ds_sp_ohe.columns:
    if '"' in i:
        rename_dict[i] = i.replace('"','').replace(' ', '_')

ds_sp_ohe = ds_sp_ohe.rename(rename_dict)
ds_sp_ohe.columns


train, test = ds_sp_ohe.random_split(weights=[0.70, 0.30], seed=0)


train = train.fillna(0)
test = test.fillna(0)


train_pd = train.to_pandas()
test_pd = test.to_pandas()

 
# ## Model Training
# ### Below we will define and fit an xgboost classifier as our baseline model and evaluate the performance
# ##### Note this is all done with OSS frameworks


#Define model config
xgb_base = XGBClassifier(
    max_depth=50,
    n_estimators=3,
    learning_rate = 0.75,
    booster = 'gbtree')


#Split train data into X, y
X_train_pd = train_pd.drop(["TIMESTAMP", "LOAN_ID", "MORTGAGERESPONSE"],axis=1) #remove
y_train_pd = train_pd.MORTGAGERESPONSE

#train model
xgb_base.fit(X_train_pd,y_train_pd)


from sklearn.metrics import f1_score, precision_score, recall_score
train_preds_base = xgb_base.predict(X_train_pd) #update this line with correct ata

f1_base_train = round(f1_score(y_train_pd, train_preds_base),4)
precision_base_train = round(precision_score(y_train_pd, train_preds_base),4)
recall_base_train = round(recall_score(y_train_pd, train_preds_base),4)

print(f'F1: {f1_base_train} \nPrecision {precision_base_train} \nRecall: {recall_base_train}')

 
# # Model Registry
# 
# - Log models with important metadata
# - Manage model lifecycles
# - Serve models from Snowflake runtimes


#Create a snowflake model registry object 
from snowflake.ml.registry import Registry

# Define model name
model_name = f"MORTGAGE_LENDING_MLOPS_{VERSION_NUM}"

# Create a registry to log the model to
model_registry = Registry(session=session, 
                          database_name=DB, 
                          schema_name=SCHEMA,
                          options={"enable_monitoring": True})


#Log the base model to the model registry (if not already there)
base_version_name = 'XGB_BASE'

try:
    #Check for existing model
    mv_base = model_registry.get_model(model_name).version(base_version_name)
    print("Found existing model version!")
except:
    print("Logging new model version...")
    #Log model to registry
    mv_base = model_registry.log_model(
        model_name=model_name,
        model=xgb_base, 
        version_name=base_version_name,
        sample_input_data = train.drop(["TIMESTAMP", "LOAN_ID", "MORTGAGERESPONSE"]).limit(100), #using snowpark df to maintain lineage
        comment = f"""ML model for predicting loan approval likelihood.
                    This model was trained using XGBoost classifier.
                    Hyperparameters used were:
                    max_depth={xgb_base.max_depth}, 
                    n_estimators={xgb_base.n_estimators}, 
                    learning_rate = {xgb_base.learning_rate}, 
                    algorithm = {xgb_base.booster}
                    """,
        target_platforms= ["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
        options= {"enable_explainability": True}

    )
    
    #set metrics
    mv_base.set_metric(metric_name="Train_F1_Score", value=f1_base_train)
    mv_base.set_metric(metric_name="Train_Precision_Score", value=precision_base_train)
    mv_base.set_metric(metric_name="Train_Recall_score", value=recall_base_train)


#Create tag for PROD model
session.sql("CREATE OR REPLACE TAG PROD")


#Apply prod tag 
m = model_registry.get_model(model_name)
m.comment = "Loan approval prediction models" #set model level comment
m.set_tag("PROD", base_version_name)
m.show_tags()


model_registry.show_models()
model_registry.get_model(model_name).show_versions()
mv_base.show_functions()

reg_preds = mv_base.run(test, function_name = "predict").rename(col('"output_feature_0"'), "MORTGAGE_PREDICTION")
reg_preds.show(10)

preds_pd = reg_preds.select(["MORTGAGERESPONSE", "MORTGAGE_PREDICTION"]).to_pandas()
f1_base_test = round(f1_score(preds_pd.MORTGAGERESPONSE, preds_pd.MORTGAGE_PREDICTION),4)
precision_base_test = round(precision_score(preds_pd.MORTGAGERESPONSE, preds_pd.MORTGAGE_PREDICTION),4)
recall_base_test = round(recall_score(preds_pd.MORTGAGERESPONSE, preds_pd.MORTGAGE_PREDICTION),4)

#log metrics to model registry model
mv_base.set_metric(metric_name="Test_F1_Score", value=f1_base_test)
mv_base.set_metric(metric_name="Test_Precision_Score", value=precision_base_test)
mv_base.set_metric(metric_name="Test_Recall_score", value=recall_base_test)

print(f'F1: {f1_base_test} \nPrecision {precision_base_test} \nRecall: {recall_base_test}')

# What's a good way to save the train dataset and restore it in the tuning step?