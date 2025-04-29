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

 
# # Oh no! Our model's performance seems to have dropped off significantly from training to our test set. 
# ## This is evidence that our model is overfit - can we fix this with Distributed Hyperparameter Optimization??


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


 
# # Model Monitoring setup


train.write.save_as_table(f"DEMO_MORTGAGE_LENDING_TRAIN_{VERSION_NUM}", mode="overwrite")
test.write.save_as_table(f"DEMO_MORTGAGE_LENDING_TEST_{VERSION_NUM}", mode="overwrite")


session.sql("CREATE stage IF NOT EXISTS ML_STAGE").collect()


from snowflake import snowpark

def demo_inference_sproc(session: snowpark.Session, table_name: str, modelname: str, modelversion: str) -> str:

    reg = Registry(session=session)
    m = reg.get_model(model_name)  # Fetch the model using the registry
    mv = m.version(modelversion)
    
    input_table_name=table_name
    pred_col = f'{modelversion}_PREDICTION'

    # Read the input table to a dataframe
    df = session.table(input_table_name)
    results = mv.run(df, function_name="predict").select("LOAN_ID",'"output_feature_0"').withColumnRenamed('"output_feature_0"', pred_col)
    # 'results' is the output DataFrame with predictions

    final = df.join(results, on="LOAN_ID", how="full")
    # Write results back to Snowflake table
    final.write.save_as_table(table_name, mode='overwrite',enable_schema_evolution=True)

    return "Success"

# Register the stored procedure
session.sproc.register(
    func=demo_inference_sproc,
    name="model_inference_sproc",
    replace=True,
    is_permanent=True,
    stage_location="@ML_STAGE",
    packages=['joblib', 'snowflake-snowpark-python', 'snowflake-ml-python'],
    return_type=StringType()
)



CALL model_inference_sproc('DEMO_MORTGAGE_LENDING_TRAIN_{{VERSION_NUM}}','{{model_name}}', '{{base_version_name}}');


CALL model_inference_sproc('DEMO_MORTGAGE_LENDING_TEST_{{VERSION_NUM}}','{{model_name}}', '{{base_version_name}}');


CALL model_inference_sproc('DEMO_MORTGAGE_LENDING_TRAIN_{{VERSION_NUM}}','{{model_name}}', '{{optimized_version_name}}');


CALL model_inference_sproc('DEMO_MORTGAGE_LENDING_TEST_{{VERSION_NUM}}','{{model_name}}', '{{optimized_version_name}}');


select TIMESTAMP, LOAN_ID, INCOME, LOAN_AMOUNT, XGB_BASE_PREDICTION, XGB_OPTIMIZED_PREDICTION, MORTGAGERESPONSE 
FROM DEMO_MORTGAGE_LENDING_TEST_{{VERSION_NUM}} 
limit 20


CREATE OR REPLACE MODEL MONITOR MORTGAGE_LENDING_BASE_MODEL_MONITOR
WITH
    MODEL={{model_name}}
    VERSION={{base_version_name}}
    FUNCTION=predict
    SOURCE=DEMO_MORTGAGE_LENDING_TEST_{{VERSION_NUM}}
    BASELINE=DEMO_MORTGAGE_LENDING_TRAIN_{{VERSION_NUM}}
    TIMESTAMP_COLUMN=TIMESTAMP
    PREDICTION_CLASS_COLUMNS=(XGB_BASE_PREDICTION)  
    ACTUAL_CLASS_COLUMNS=(MORTGAGERESPONSE)
    ID_COLUMNS=(LOAN_ID)
    WAREHOUSE={{COMPUTE_WAREHOUSE}}
    REFRESH_INTERVAL='1 hour'
    AGGREGATION_WINDOW='1 day';


CREATE OR REPLACE MODEL MONITOR MORTGAGE_LENDING_OPTIMIZED_MODEL_MONITOR
WITH
    MODEL={{model_name}}
    VERSION={{optimized_version_name}}
    FUNCTION=predict
    SOURCE=DEMO_MORTGAGE_LENDING_TEST_{{VERSION_NUM}}
    BASELINE=DEMO_MORTGAGE_LENDING_TRAIN_{{VERSION_NUM}}
    TIMESTAMP_COLUMN=TIMESTAMP
    PREDICTION_CLASS_COLUMNS=(XGB_OPTIMIZED_PREDICTION)  
    ACTUAL_CLASS_COLUMNS=(MORTGAGERESPONSE)
    ID_COLUMNS=(LOAN_ID)
    WAREHOUSE={{COMPUTE_WAREHOUSE}}
    REFRESH_INTERVAL='12 hours'
    AGGREGATION_WINDOW='1 day';


#Click the generated link to view your model in the model regsitry and check out the model monitors!
st.write(f'https://app.snowflake.com/{org_name}/{account_name}/#/data/databases/{DB}/schemas/{SCHEMA}/model/{model_name.upper()}')


SELECT * FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
'MORTGAGE_LENDING_BASE_MODEL_MONITOR', -- model monitor to use
'DIFFERENCE_OF_MEANS', -- metric for computing drift
'XGB_BASE_PREDICTION', -- comlumn to compute drift on
'1 DAY',  -- day granularity for drift computation
DATEADD(DAY, -90, CURRENT_DATE()), -- end date
DATEADD(DAY, -60, CURRENT_DATE()) -- start date
)
)

 
# # SPCS Deployment setup (OPTIONAL)
# ## This is disabled by default but uncommenting the below code cells will allow a user to 
# 
# - ### Create a new compute pool with 3 XL CPU nodes
# - ### Create a new image repository to store the container image for conatiner-based model scoring
# - ### Deploys a service on top of our existing HPO model version
# - ### Tests out inference on newly created container service
# 


image_repo_name = "MORTGAGE_LENDING_IMAGE_REPO_LLM"
cp_name = "MORTGAGE_LENDING_INFERENCE_CP"
num_spcs_nodes = '3'
spcs_instance_family = 'CPU_X64_L'
service_name = 'MORTGAGE_LENDING_PREDICTION_SERVICE'

current_database = session.get_current_database().replace('"', '')
current_schema = session.get_current_schema().replace('"', '')
extended_image_repo_name = f"{current_database}.{current_schema}.{image_repo_name}"
extended_service_name = f'{current_database}.{current_schema}.{service_name}'


session.sql(f"show image repositories").collect()


session.sql(f"alter compute pool if exists {cp_name} stop all").collect()
session.sql(f"drop compute pool if exists {cp_name}").collect()
session.sql(f"create compute pool {cp_name} min_nodes={num_spcs_nodes} max_nodes={num_spcs_nodes} instance_family={spcs_instance_family} auto_resume=True auto_suspend_secs=300").collect()
session.sql(f"describe compute pool {cp_name}").show()


session.sql(f"create or replace image repository {extended_image_repo_name}").collect()


#note this may take up to 5 minutes to run

mv_opt.create_service(
    service_name=extended_service_name,
    service_compute_pool=cp_name,
    image_repo=extended_image_repo_name,
    ingress_enabled=True,
    max_instances=int(num_spcs_nodes)
    # build_external_access_integration="ALLOW_ALL_INTEGRATION
)


model_registry.get_model(f"MORTGAGE_LENDING_MLOPS_{VERSION_NUM}").show_versions()


mv_container = model_registry.get_model(f"MORTGAGE_LENDING_MLOPS_{VERSION_NUM}").default
mv_container.list_services()


mv_container.run(test, function_name = "predict", service_name = "MORTGAGE_LENDING_PREDICTION_SERVICE").rename('"output_feature_0"', 'XGB_PREDICTION')


#Stop the service to save costs
session.sql(f"alter compute pool if exists {cp_name} stop all").collect()

 
# ## Conclusion 
# 
# #### 🛠️ Snowflake Feature Store tracks feature definitions and maintains lineage of sources and destinations 🛠️
# #### 🚀 Snowflake Model Registry gives users a secure and flexible framework to log models, tag candidates for production, and run inference and explainability jobs 🚀
# #### 📈 ML observability in Snowflake allows users to montior model performance over time and detect model, feature, and concept drift 📈
# #### 🔮 All models logged in the Model Registry can be accessed for inference, explainability, lineage tracking, visibility and more 🔮
# 


