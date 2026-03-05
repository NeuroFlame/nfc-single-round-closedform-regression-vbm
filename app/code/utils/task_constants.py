# app/code/utils/task_constants.py

# Contributors only
TASK_CLIENT_STEP1_VALIDATE_INPUTS_GENERATE_MASK = "step1_validate_inputs_generate_mask"
TASK_CLIENT_STEP2_COMPUTE_LOCAL_MODEL = "step2_compute_local_model"
TASK_CLIENT_STEP3_COMPUTE_METRICS = "step3_compute_metrics"

# Everyone (broadcast-only lifecycle task)
TASK_CLIENT_RECEIVE_RESULTS = "receive_results"

# Aggregator component ID
SRCFR_AGGREGATOR_ID = "srcfr_aggregator"