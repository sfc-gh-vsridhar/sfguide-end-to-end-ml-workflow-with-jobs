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