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
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/dist_train.sh tool/dist_train.py tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.launch --nproc_per_node=2 ${exp_dir}/dist_train.py \
  --config=${config} \
  2>&1 | tee ${model_dir}/train-$now.log

# $PYTHON -u ${exp_dir}/test.py \
#   --config=${config} \
#   2>&1 | tee ${result_dir}/test-$now.log
