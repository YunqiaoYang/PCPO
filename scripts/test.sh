set -ex

cd ./qwen_math


# PROMPT_TYPE="qwen-boxed"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$MODEL_NAME_OR_PATH


NUM_TEST_SAMPLE=-1

# English open datasets
DATA_NAME=$2
if [ -z "$3" ];then
  device=0,1
else
  device=$3
fi
if [ -z "$4" ];then
  PROMPT_TYPE="qwen25-math-cot" 
else
  PROMPT_TYPE=$4
fi
if [ -z "$5" ];then
  SEED=0
else
  SEED=$5
fi
if [ -z "$6" ];then
  num=1
else
  num=$6
fi
if [ -z "$7" ];then
  SPLIT="test"
else
  SPLIT=$7
fi
if [ -z "$8" ];then
  max_tokens=2048
else
  max_tokens=$8
fi

out_path=$OUTPUT_DIR/$DATA_NAME/${SPLIT}_${PROMPT_TYPE}_-1_seed${SEED}_t0.0_s0_e-1.jsonl


if [ ! -f "$out_path" ]; then
echo ${out_path} not exists
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=$device \
python -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature 0 \
    --max_tokens_per_call $max_tokens \
    --n_sampling $num \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite
fi