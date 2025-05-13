#!/bin/bash

# Define local models to test
# Ensure these model names are valid Hugging Face identifiers
# or paths to local checkpoints that nli_test.py can load.
LOCAL_MODELS=(
    "Qwen/Qwen3-4B"  # Example model name, replace with actual if different
    "Qwen/Qwen3-8B"  # Example model name, replace with actual if different
)
# Note: As of my last update, Qwen3 models might not be widely released or might have specific names.
# Please use the exact Hugging Face model identifiers.
# For example, Qwen1.5 models are available: "Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-4B-Chat"
# If "Qwen/Qwen3-8B" and "Qwen/Qwen3-4B" are correct, keep them.

DATASETS=("dev_matched" "dev_mismatched") # Add paths to .jsonl files if needed

# Common parameters
BATCH_SIZE=16       # Adjust based on your GPU memory for local models
SAMPLE_SIZE=50      # Adjust for quicker tests or remove for full evaluation
NUM_PROC=4          # Number of workers for Hugging Face map()
BASE_OUTPUT_DIR="test_results/local_models"
PYTHON_SCRIPT_PATH="/home/shurui/projects/NLP-Individual/nli_test.py" # Adjusted path

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Check if the python script exists
if [ ! -f "${PYTHON_SCRIPT_PATH}" ]; then
    echo "Error: Python script ${PYTHON_SCRIPT_PATH} not found."
    exit 1
fi

# Loop through datasets
for dataset_identifier in "${DATASETS[@]}"; do
    echo "=================================================="
    echo "Testing Dataset: ${dataset_identifier}"
    echo "=================================================="

    SPLIT_ARG="${dataset_identifier}"
    DATASET_NAME_FOR_FILE="${dataset_identifier}"

    # If dataset_identifier is a path to a .jsonl file, extract a base name for the output file
    if [[ "${dataset_identifier}" == *.jsonl ]]; then
        if [ ! -f "${dataset_identifier}" ]; then
            echo "Error: Dataset file ${dataset_identifier} not found."
            continue # Skip to the next dataset
        fi
        # Use the filename without extension for the output name
        DATASET_NAME_FOR_FILE=$(basename "${dataset_identifier}" .jsonl)
    else
        # For Hugging Face dataset splits like "dev_matched", create a dummy SPLIT_FILE for naming consistency
        # or ensure your nli_test.py handles these names directly for output.
        # Here, we'll just use the identifier for naming.
        # If you have corresponding .jsonl files for these, adjust logic like in test_api_models.sh
        # e.g., SPLIT_FILE="${dataset_identifier}_sampled-1.jsonl" and check its existence.
        # For simplicity, assuming nli_test.py handles HF dataset names directly for loading.
        : # No specific file check needed if it's a HF dataset name
    fi


    # Loop through models
    for model_name_or_path in "${LOCAL_MODELS[@]}"; do
        # Sanitize model name for use in filenames (replace / with _)
        sanitized_model_name=$(echo "${model_name_or_path}" | tr '/' '_')

        echo "=================================================="
        echo "Testing Model: ${model_name_or_path}"
        echo "=================================================="

        # Create a unique name for log and prediction files
        output_name="${sanitized_model_name}_${DATASET_NAME_FOR_FILE}"
        log_file="${BASE_OUTPUT_DIR}/${output_name}.log"
        preds_file="${BASE_OUTPUT_DIR}/${output_name}_preds.tsv"

        # Construct the command
        command="python ${PYTHON_SCRIPT_PATH} \
            --model-name \"${model_name_or_path}\" \
            --split \"${SPLIT_ARG}\" \
            --batch-size ${BATCH_SIZE} \
            --num-proc ${NUM_PROC} \
            --save-preds-path \"${preds_file}\""

        if [ ! -z "${SAMPLE_SIZE}" ]; then
            command="${command} --sample-size ${SAMPLE_SIZE}"
        fi

        # Print the command
        echo "Running command:"
        echo "${command}"
        echo ""

        # Execute the command and save output to log file
        ${command} > "${log_file}" 2>&1

        # Check exit status
        if [ $? -eq 0 ]; then
            echo "Successfully completed. Log: ${log_file}, Predictions: ${preds_file}"
        else
            echo "Error during execution. Check log: ${log_file}"
        fi
        echo "--------------------------------------------------"
        echo ""
    done
    echo "=================================================="
    echo ""
done

echo "All local model tests completed. Results are in the '${BASE_OUTPUT_DIR}' directory."