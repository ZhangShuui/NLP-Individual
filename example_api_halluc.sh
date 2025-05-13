#!/usr/bin/env bash
# ------------------------------------------------------------
# batch_halluc_test.sh – Evaluate GPT models on hallucination task
# ------------------------------------------------------------

# === Required environment variables (please export before running the script) ===
export OPENAI_API_KEY="sk-xx"
export OPENAI_API_BASE="xxx"

MODELS=(
    "gpt-4o-mini"
    "gpt-4.1-nano-2025-04-14"
)

# Prompt 6 requires the genre field, hallucination dataset doesn't have it → not included
PROMPTS=(
    "system_default"
    "1_direct_instruction"
    "2_chain_of_thought"
    "3_summarize_compare_label"
)

# WikiBio-GPT3-Hallucination splits
SPLITS=("validation" "test")

BATCH_SIZE=25              # Concurrent request size
BASE_OUT="test_results/api_models_halluc"
mkdir -p "${BASE_OUT}"

# -------- Execute loop --------
for split in "${SPLITS[@]}"; do
    echo "========== Split: ${split} =========="
    for model in "${MODELS[@]}"; do
        echo "  ---- Model: ${model}"
        for prompt in "${PROMPTS[@]}"; do
            echo "    >> Prompt: ${prompt}"

            name="${model}_${prompt}_${split}"
            log="${BASE_OUT}/${name}.log"
            preds="${BASE_OUT}/${name}_preds.tsv"

            cmd="
python /home/shurui/projects/NLP-Individual/nli_test.py \
    --task halluc \
    --model-name ${model} \
    --split ${split} \
    --batch-size ${BATCH_SIZE} \
    --prompt-template ${prompt} \
    --save-preds-path ${preds}
"
            # Run and redirect logs
            eval ${cmd} > \"${log}\" 2>&1
            if [ $? -eq 0 ]; then
                echo \"      ✓ Completed, log: ${log}\"
            else
                echo \"      ✗ Error, please check log: ${log}\"
            fi
        done
    done
done

echo \"All hallucination tests completed, results are in '${BASE_OUT}'.\" 
