from snowflake.ml.jobs import submit_file

compute_pool = "MY_COMPUTE_POOL"
# Upload and run a single script
job1 = submit_file(
    "./01_feature_eng.py",
    compute_pool,
    stage_name="payload_stage"
)