export OPENAI_API_KEY="sk-xxx"
export OPENAI_API_BASE="xxx"
# Define models and prompts to test
MODELS=("gpt-4o-mini" "gpt-4.1-nano-2025-04-14")
PROMPTS=(
    "system_default"
    "1_direct_instruction"
    "2_chain_of_thought"
    "3_summarize_compare_label"
    "6_genre_aware_few_shot"
)
DATASETS=("dev_matched" "dev_mismatched")

# Common parameters
# SPLIT="dev_matched" # Or use "dev_mismatched" or a path to a .jsonl file
# SPLIT_FILE="./${SPLIT}_sampled-1.jsonl" # Adjust as needed
BATCH_SIZE=50
# SAMPLE_SIZE=1000 # Adjust for quicker tests or remove for full evaluation
BASE_OUTPUT_DIR="test_results/api_models"

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

# Ensure OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it before running the script, e.g., export OPENAI_API_KEY='your_key_here'"
    exit 1
fi

# Loop through datasets
for dataset in "${DATASETS[@]}"; do
    echo "=================================================="
    echo "Testing Dataset: ${dataset}"
    echo "=================================================="

    SPLIT_FILE="${dataset}_sampled-1.jsonl"
    # Check if the dataset file exists
    if [ ! -f "${SPLIT_FILE}" ]; then
        echo "Error: Dataset file ${SPLIT_FILE} not found."
        exit 1
    fi

    # Set the split file for the current dataset
    

    # Loop through models
    for model_name in "${MODELS[@]}"; do
        echo "=================================================="
        echo "Testing Model: ${model_name}"
        echo "=================================================="

        # Loop through prompts
        for prompt_template in "${PROMPTS[@]}"; do
            echo "--------------------------------------------------"
            echo "Using Prompt Template: ${prompt_template}"
            echo "--------------------------------------------------"

            # Create a unique name for log and prediction files
            output_name="${model_name}_${prompt_template}_${dataset}"
            log_file="${BASE_OUTPUT_DIR}/${output_name}.log"
            preds_file="${BASE_OUTPUT_DIR}/${output_name}_preds.tsv"

            # Construct the command
            command="python /home/shurui/projects/NLP-Individual/nli_test.py \
                --model-name ${model_name} \
                --split ${SPLIT_FILE} \
                --batch-size ${BATCH_SIZE} \
                --prompt-template ${prompt_template} \
                --save-preds-path ${preds_file}"

            if [ ! -z "${SAMPLE_SIZE}" ]; then
                command="${command} --sample-size ${SAMPLE_SIZE}"
            fi

            # Print the command
            echo "Running command:"
            echo "${command}"
            echo ""

            # Execute the command and save output to log file
            # Use 'script -q -c "command" /dev/null | tee log_file' to capture color codes if needed
            # For simplicity, direct stdout and stderr to log file
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
    echo "=================================================="
    echo ""
done

echo "All tests completed. Results are in the '${BASE_OUTPUT_DIR}' directory."