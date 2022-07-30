#!/bin/sh

## uncomment for slurm
##SBATCH -p gpu
##SBATCH --gres=gpu:8
##SBATCH -c 80

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
conda activate real-time
PYTHON=python

dataset=$1
exp_name=$2
suffix=$3
exp_dir=exp/${dataset}/${exp_name}/${suffix}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
local_config=${exp_dir}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/train_loop.sh tool/train_loop.py tool/evaluate.py ${config} ${exp_dir}

export PYTHONPATH=./
$PYTHON -u ${exp_dir}/train_loop.py \
  --config=${local_config} \
  2>&1 | tee ${model_dir}/train-$now.log

## evaluate for last iteration model
$PYTHON -u ${exp_dir}/evaluate.py \
  --config=${local_config} \
  model_path ${model_dir}/last.pth \
  2>&1 | tee ${result_dir}/eval_last-$now.log

# ## evaluate for best val mIoU model
# $PYTHON -u ${exp_dir}/evaluate.py \
#   --config=${local_config} \
#   model_path ${model_dir}/best.pth \
#   2>&1 | tee -a ${result_dir}/eval_best-$now.log