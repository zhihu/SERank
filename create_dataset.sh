mkdir web30k
python create_dataset.py -tfrecord Fold1/train.txt web30k/train.tfrecord 200 136 &
python create_dataset.py -tfrecord Fold1/vali.txt web30k/vali.tfrecord 800 136 &
python create_dataset.py -tfrecord Fold1/test.txt web30k/test.tfrecord 800 136 &
wait %1
wait %2
wait %3
