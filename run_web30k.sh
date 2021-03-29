#!/bin/bash

pip install -r requirements.txt
output_dir=serank-b
#rm -r $output_dir
mkdir $output_dir

python serank.py \
  --train_path=web30k/train.tfrecord \
  --vali_path=web30k/vali.tfrecord \
  --test_path=web30k/test.tfrecord \
  --output_dir=$output_dir \
  --num_features=136 \
  --serank=true \
  --shrink_first=True \
  --query_label_weight=False \
  --num_train_steps=10000 \
  --tfrecord=True
