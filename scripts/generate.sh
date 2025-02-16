set -ex


MODEL_NAME_OR_PATH=$1
iteration_dir_path=$2
OUTPUT_DIR=$iteration_dir_path

NUM_TEST_SAMPLE=-1
SEED=0
cd ./qwen_math

# English open datasets
DATA_NAME=$3
response_num=$4
suffix=$5


if [ -z "$6" ];then
  device=0,1,2,3
else
  device=$6
fi

if [ -z "$7" ];then
  SPLIT="train"
else
  SPLIT=$7
fi

if [ -z "$8" ];then
  PROMPT_TYPE=qwen25-math-cot
else
  PROMPT_TYPE=$8
fi

out_path=$OUTPUT_DIR/$DATA_NAME/${SPLIT}_${PROMPT_TYPE}_-1_seed0_t1.0_s0_e-1.jsonl$suffix


if [ ! -f "$out_path" ]; then
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=${device} \
python -u math_generate.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed $SEED \
    --temperature 1 \
    --n_sampling $response_num \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --suffix $suffix
fi