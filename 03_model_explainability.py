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

#Split train data into X, y
X_train_pd = train_pd.drop(["TIMESTAMP", "LOAN_ID", "MORTGAGERESPONSE"],axis=1) #remove
y_train_pd = train_pd.MORTGAGERESPONSE

#Create a snowflake model registry object 
from snowflake.ml.registry import Registry

# Define model name
model_name = f"MORTGAGE_LENDING_MLOPS_{VERSION_NUM}"

# Create a registry to log the model to
model_registry = Registry(session=session, 
                          database_name=DB, 
                          schema_name=SCHEMA)


#Check for existing base model
base_version_name = 'XGB_BASE'
mv_base = model_registry.get_model(model_name).version(base_version_name)
print("Found existing base model version!")




#Check for existing optimized model
optimized_version_name = 'XGB_Optimized'
mv_opt = model_registry.get_model(model_name).version(optimized_version_name)
print("Found existing optimized model version!")

 
# ## Now that we've deployed some model versions and tested inference... 
# # Let's explain our models!
# - ### Snowflake offers built in explainability capabilities on top of models logged to the model registry
# - ### In the below section we'll generate shapley values using these built in functions to understand how input features impact our model's behavior


#create a sample of 1000 records
test_pd_sample=test_pd.rename(columns=rename_dict).sample(n=2500, random_state = 100).reset_index(drop=True)

#Compute shapley values for each model
base_shap_pd = mv_base.run(test_pd_sample, function_name="explain")
opt_shap_pd = mv_opt.run(test_pd_sample, function_name="explain")


import shap 

shap.summary_plot(np.array(base_shap_pd.astype(float)), 
                  test_pd_sample.drop(["LOAN_ID","MORTGAGERESPONSE", "TIMESTAMP"], axis=1), 
                  feature_names = test_pd_sample.drop(["LOAN_ID","MORTGAGERESPONSE", "TIMESTAMP"], axis=1).columns)


shap.summary_plot(np.array(opt_shap_pd.astype(float)), 
                  test_pd_sample.drop(["LOAN_ID","MORTGAGERESPONSE", "TIMESTAMP"], axis=1), 
                  feature_names = test_pd_sample.drop(["LOAN_ID","MORTGAGERESPONSE", "TIMESTAMP"], axis=1).columns)


#Merge shap vals and actual vals together for easier plotting below
all_shap_base = test_pd_sample.merge(base_shap_pd, right_index=True, left_index=True, how='outer')
all_shap_opt = test_pd_sample.merge(opt_shap_pd, right_index=True, left_index=True, how='outer')


import seaborn as sns
import matplotlib.pyplot as plt

#filter data down to strip outliers
asb_filtered = all_shap_base[(all_shap_base.INCOME>0) & (all_shap_base.INCOME<250000)]
aso_filtered = all_shap_opt[(all_shap_opt.INCOME>0) & (all_shap_opt.INCOME<250000)]

# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle("INCOME EXPLANATION")
# Plot side-by-side boxplots
sns.scatterplot(data = asb_filtered, x ='INCOME', y = 'INCOME_explanation', ax=axes[0])
sns.regplot(data = asb_filtered, x ="INCOME", y = 'INCOME_explanation', scatter=False, color='red', line_kws={"lw":2},ci =100, lowess=False, ax =axes[0])

axes[0].set_title('Base Model')
sns.scatterplot(data = aso_filtered, x ='INCOME', y = 'INCOME_explanation',color = "orange", ax = axes[1])
sns.regplot(data = aso_filtered, x ="INCOME", y = 'INCOME_explanation', scatter=False, color='blue', line_kws={"lw":2},ci =100, lowess=False, ax =axes[1])
axes[1].set_title('Opt Model')

# Customize and show the plot
for ax in axes:
    ax.set_xlabel("Income")
    ax.set_ylabel("Influence")
plt.tight_layout()
plt.show()



#filter data down to strip outliers
asb_filtered = all_shap_base[all_shap_base.LOAN_AMOUNT<2000000]
aso_filtered = all_shap_opt[all_shap_opt.LOAN_AMOUNT<2000000]


# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle("LOAN_AMOUNT EXPLANATION")
# Plot side-by-side boxplots
sns.scatterplot(data = asb_filtered, x ='LOAN_AMOUNT', y = 'LOAN_AMOUNT_explanation', ax=axes[0])
sns.regplot(data = asb_filtered, x ="LOAN_AMOUNT", y = 'LOAN_AMOUNT_explanation', scatter=False, color='red', line_kws={"lw":2},ci =100, lowess=True, ax =axes[0])
axes[0].set_title('Base Model')

sns.scatterplot(data = aso_filtered, x ='LOAN_AMOUNT', y = 'LOAN_AMOUNT_explanation',color = "orange", ax = axes[1])
sns.regplot(data = aso_filtered, x ="LOAN_AMOUNT", y = 'LOAN_AMOUNT_explanation', scatter=False, color='blue', line_kws={"lw":2},ci =100, lowess=True, ax =axes[1])
axes[1].set_title('Opt Model')

# Customize and show the plot
for ax in axes:
    ax.set_xlabel("LOAN_AMOUNT")
    ax.set_ylabel("Influence")
    # ax.set_xlim((0,10000))
plt.tight_layout()
plt.show()



# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle("HOME PURCHASE LOAN EXPLANATION")
# Plot side-by-side boxplots
sns.boxplot(data = all_shap_base, x ='LOAN_PURPOSE_NAME_HOME_PURCHASE', y = 'LOAN_PURPOSE_NAME_HOME_PURCHASE_explanation',
            hue='LOAN_PURPOSE_NAME_HOME_PURCHASE', width=0.8, ax=axes[0])
axes[0].set_title('Base Model')
sns.boxplot(data = all_shap_opt, x ='LOAN_PURPOSE_NAME_HOME_PURCHASE', y = 'LOAN_PURPOSE_NAME_HOME_PURCHASE_explanation',
            hue='LOAN_PURPOSE_NAME_HOME_PURCHASE', width=0.4, ax = axes[1])
axes[1].set_title('Opt Model')

# Customize and show the plot
for ax in axes:
    ax.set_xlabel("Home PURCHASE Loan (1 = True)")
    ax.set_ylabel("Influence")
    ax.legend(loc='upper right')

plt.show()



# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle("HOME IMPROVEMENT LOAN EXPLANATION")
# Plot side-by-side boxplots
sns.boxplot(data = all_shap_base, x ='LOAN_PURPOSE_NAME_HOME_IMPROVEMENT', y = 'LOAN_PURPOSE_NAME_HOME_IMPROVEMENT_explanation',
            hue='LOAN_PURPOSE_NAME_HOME_IMPROVEMENT', width=0.8, ax=axes[0])
axes[0].set_title('Base Model')
sns.boxplot(data = all_shap_opt, x ='LOAN_PURPOSE_NAME_HOME_IMPROVEMENT', y = 'LOAN_PURPOSE_NAME_HOME_IMPROVEMENT_explanation',
            hue='LOAN_PURPOSE_NAME_HOME_IMPROVEMENT', width=0.4, ax = axes[1])
axes[1].set_title('Opt Model')

# Customize and show the plot
for ax in axes:
    ax.set_xlabel("Home Improvement Loan (1 = True)")
    ax.set_ylabel("Influence")
    ax.legend(loc='upper right')

plt.show()
 
# ## Conclusion 
# 
# #### 🛠️ Snowflake Feature Store tracks feature definitions and maintains lineage of sources and destinations 🛠️
# #### 🚀 Snowflake Model Registry gives users a secure and flexible framework to log models, tag candidates for production, and run inference and explainability jobs 🚀
# #### 📈 ML observability in Snowflake allows users to montior model performance over time and detect model, feature, and concept drift 📈
# #### 🔮 All models logged in the Model Registry can be accessed for inference, explainability, lineage tracking, visibility and more 🔮
# 
