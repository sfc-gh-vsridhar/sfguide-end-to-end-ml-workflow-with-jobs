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

#setup snowpark session
from snowflake.snowpark.context import get_active_session
session = get_active_session()


try:
    print("Reading table data...")
    df = session.table("MORTGAGE_LENDING_DEMO_DATA")
    df.show(5)
except:
    print("Table not found! Uploading data to snowflake table")
    df_pandas = pd.read_csv("MORTGAGE_LENDING_DEMO_DATA.csv.zip")
    session.write_pandas(df_pandas, "MORTGAGE_LENDING_DEMO_DATA", auto_create_table=True)
    df = session.table("MORTGAGE_LENDING_DEMO_DATA")

#Get current date and time
current_time = datetime.now()
df_max_time = datetime.strptime(str(df.select(max("TS")).collect()[0][0]), "%Y-%m-%d %H:%M:%S.%f")

#Find delta between latest existing timestamp and today's date
timedelta = current_time- df_max_time

#Update timestamps to represent last ~1 year from today's date
df.select(min(date_add(to_timestamp("TS"), timedelta.days-1)), max(date_add(to_timestamp("TS"), timedelta.days-1)))

# ## Feature Engineering with Snowpark APIs
#Create a dict with keys for feature names and values containing transform code

feature_eng_dict = dict()

#Timstamp features
feature_eng_dict["TIMESTAMP"] = date_add(to_timestamp("TS"), timedelta.days-1)
feature_eng_dict["MONTH"] = month("TIMESTAMP")
feature_eng_dict["DAY_OF_YEAR"] = dayofyear("TIMESTAMP") 
feature_eng_dict["DOTW"] = dayofweek("TIMESTAMP")

# df= df.with_columns(feature_eng_dict.keys(), feature_eng_dict.values())

#Income and loan features
feature_eng_dict["LOAN_AMOUNT"] = col("LOAN_AMOUNT_000s")*1000
feature_eng_dict["INCOME"] = col("APPLICANT_INCOME_000s")*1000
feature_eng_dict["INCOME_LOAN_RATIO"] = col("INCOME")/col("LOAN_AMOUNT")

county_window_spec = Window.partition_by("COUNTY_NAME")
feature_eng_dict["MEAN_COUNTY_INCOME"] = avg("INCOME").over(county_window_spec)
feature_eng_dict["HIGH_INCOME_FLAG"] = (col("INCOME")>col("MEAN_COUNTY_INCOME")).astype(IntegerType())

feature_eng_dict["AVG_THIRTY_DAY_LOAN_AMOUNT"] =  sql_expr("""AVG(LOAN_AMOUNT) OVER (PARTITION BY COUNTY_NAME ORDER BY TIMESTAMP  
                                                            RANGE BETWEEN INTERVAL '30 DAYS' PRECEDING AND CURRENT ROW)""")

df = df.with_columns(feature_eng_dict.keys(), feature_eng_dict.values())
 
# ## Create a Snowflake Feature Store
fs = FeatureStore(
    session=session, 
    database=DB, 
    name=SCHEMA, 
    default_warehouse=COMPUTE_WAREHOUSE,
    creation_mode=CreationMode.CREATE_IF_NOT_EXIST
)

#First try to retrieve an existing entity definition, if not define a new one and register
try:
    #retrieve existing entity
    loan_id_entity = fs.get_entity('LOAN_ENTITY') 
    print('Retrieved existing entity')

except:
#define new entity
    loan_id_entity = Entity(
        name = "LOAN_ENTITY",
        join_keys = ["LOAN_ID"],
        desc = "Features defined on a per loan level")
    #register
    fs.register_entity(loan_id_entity)
    print("Registered new entity")


#Create a dataframe with just the ID, timestamp, and engineered features. We will use this to define our feature view
feature_df = df.select(["LOAN_ID"]+list(feature_eng_dict.keys()))
feature_df.show(5)


#define and register feature view
loan_fv = FeatureView(
    name="Mortgage_Feature_View",
    entities=[loan_id_entity],
    feature_df=feature_df,
    timestamp_col="TIMESTAMP",
    refresh_freq="1 day")

#add feature level descriptions

loan_fv = loan_fv.attach_feature_desc(
    {
        "MONTH": "Month of loan",
        "DAY_OF_YEAR": "Day of calendar year of loan",
        "DOTW": "Day of the week of loan",
        "LOAN_AMOUNT": "Loan amount in $USD",
        "INCOME": "Household income in $USD",
        "INCOME_LOAN_RATIO": "Ratio of LOAN_AMOUNT/INCOME",
        "MEAN_COUNTY_INCOME": "Average household income aggregated at county level",
        "HIGH_INCOME_FLAG": "Binary flag to indicate whether household income is higher than MEAN_COUNTY_INCOME",
        "AVG_THIRTY_DAY_LOAN_AMOUNT": "Rolling 30 day average of LOAN_AMOUNT"
    }
)

loan_fv = fs.register_feature_view(loan_fv, version=VERSION_NUM, overwrite=True)

#Generate a dataset to pickup later for ML Modeling
ds = fs.generate_dataset(
    name=f"MORTGAGE_DATASET_EXTENDED_FEATURES_{VERSION_NUM}",
    spine_df=df.select("LOAN_ID", "TIMESTAMP", "LOAN_PURPOSE_NAME","MORTGAGERESPONSE"), #only need the features used to fetch rest of feature view
    features=[loan_fv],
    spine_timestamp_col="TIMESTAMP",
    spine_label_cols=["MORTGAGERESPONSE"]
)