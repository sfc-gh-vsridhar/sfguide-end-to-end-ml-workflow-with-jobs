# Quickstart showcasing an end-to-end ML workflow in Snowflake
 - Use Feature Store to track engineered features
     - Store feature definitions in feature store for reproducible computation of ML features
 - Train two SnowML Models
     - Baseline XGboost
     - XGboost with optimal hyper-parameters identified via Snowflake ML distributed HPO methods
 - Register both models in Snowflake model registry
     - Explore model registry capabilities such as metadata tracking, inference, and explainability
     - Compare model metrics on train/test set to identify any issues of model performance or overfitting
     - Tag the best performing model version as 'default' version
 - Set up Model Monitor to track 1 year of predicted and actual loan repayments
     - Compute performance metrics such a F1, Precision, Recall
     - Inspect model drift (i.e. how much has the average predicted repayment rate changed day-to-day)
     - Compare models side-by-side to understand which model should be used in production
     - Identify and understand data issues
 - Track data and model lineage throughout
     - View and understand
       - The origin of the data used for computed features
       - The data used for model training
       - The available model versions being monitored
 - Additional components also include
     - Distributed GPU model training example
     - SPCS deployment for inference
         - [WIP] REST API scoring example 
 
 
 INSTRUCTIONS:
## Step-by-Step Guide
For prerequisites, environment setup, step-by-step guide and instructions, please refer to the [QuickStart Guide](https://quickstarts.snowflake.com/guide/end-to-end-ml-workflow).
 
