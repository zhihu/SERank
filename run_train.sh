#!/bin/bash

pip install -r requirements.txt
output_dir=outputs
rm -r $output_dir
mkdir $output_dir

python serank.py \
  --train_path=data/train.txt \
  --vali_path=data/vali.txt \
  --test_path=data/test.txt \
  --output_dir=$output_dir \
  --num_features=136 \
  --serank=True \
  --query_label_weight=True
