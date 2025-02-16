set -ex

cd ./qwen_math


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
temperature=0.95
if [ -z "$5" ];then
  SEED=0
else
  SEED=$5
fi
if [ -z "$6" ];then
  num=8
else
  num=$6
fi
if [ -z "$7" ];then
  SPLIT="test"
else
  SPLIT=$7
fi

out_path=$OUTPUT_DIR/$DATA_NAME/${SPLIT}_${PROMPT_TYPE}_-1_seed${SEED}_t${temperature}_maj@${num}_s0_e-1.jsonl


if [ ! -f "$out_path" ]; then
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES=$device \
python -u math_eval_maj.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature $temperature \
    --n_sampling $num \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --maj
fi