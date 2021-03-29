#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
import logging

import numpy as np
import pandas as pd
import six
import tensorflow as tf

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )

parser = argparse.ArgumentParser()
parser.add_argument('-np', action='store_true')
parser.add_argument('-tfrecord', action='store_true')
parser.add_argument('-parquet', action='store_true')
parser.add_argument('src')
parser.add_argument('des')
parser.add_argument('list_size', type=int)
parser.add_argument('num_features', type=int)
args = parser.parse_args()
info = logging.info


def create_float_feature(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(np.reshape(v, [-1]))))


def load_libsvm_data(path, list_size, num_features):
    """Returns features and labels in numpy.array."""

    def discard_fn(arr):
        if not discards:
            return arr
        arr_ = []
        for i, x in enumerate(arr):
            if i in discards:
                continue
            arr_.append(x)
        return arr_

    def _parse_line(line):
        """Parses a single line in LibSVM format."""
        tokens = line.split("#")[0].split()
        assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
        label = float(tokens[0])
        qid = tokens[1]
        kv_pairs = [kv.split(":") for kv in tokens[2:]]
        features = {k: float(v) for (k, v) in kv_pairs}
        return qid, features, label

    info("Loading data from {}".format(path))

    # The 0-based index assigned to a query.
    qid_to_index = {}
    # The number of docs seen so far for a query.
    qid_to_ndoc = {}
    # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
    # a shape of [num_queries, list_size]. We use list for each of them due to the
    # unknown number of quries.
    feature_map = {str(k): [] for k in range(1, num_features + 1)}
    label_list = []
    total_docs = 0
    discarded_docs = 0
    with open(path, "rt") as f:
        for line in f:
            qid, features, label = _parse_line(line)
            if qid not in qid_to_index:
                # Create index and allocate space for a new query.
                qid_to_index[qid] = len(qid_to_index)
                qid_to_ndoc[qid] = 0
                for k in feature_map:
                    feature_map[k].append(np.zeros([list_size, 1], dtype=np.float32))
                label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
            total_docs += 1
            batch_idx = qid_to_index[qid]
            doc_idx = qid_to_ndoc[qid]
            qid_to_ndoc[qid] += 1
            # Keep the first 'list_size' docs only.
            if doc_idx >= list_size:
                discarded_docs += 1
                continue
            for k, v in six.iteritems(features):
                assert k in feature_map, "Key {} not found in features.".format(k)
                feature_map[k][batch_idx][doc_idx, 0] = v
            label_list[batch_idx][doc_idx] = label

    discards = set()
    for i, label in enumerate(label_list):
        if np.all(label <= 0):
            discards.add(i)
    info('num queries to discard: {}'.format(len(discards)))
    label_list = discard_fn(label_list)
    for k in feature_map:
        feature_map[k] = discard_fn(feature_map[k])

    info("Number of queries: {}".format(len(qid_to_index)))
    info(
        "Number of documents in total: {}".format(total_docs))
    info(
        "Number of documents discarded: {}".format(discarded_docs))

    # Convert everything to np.array.
    for k in feature_map:
        feature_map[k] = np.array(feature_map[k])

    label_list = np.array(label_list)
    return feature_map, label_list


if __name__ == '__main__':
    feature_map, label_list = load_libsvm_data(args.src, args.list_size, args.num_features)
    if args.np:
        for k, arr in feature_map:
            np.save('{}.{}.npy'.format(args.des, k), arr)
        np.save('{}.{}.npy'.format(args.des, 'label'), label_list)

    if args.tfrecord:
        writer = tf.io.TFRecordWriter(args.des)
        for i in range(len(label_list)):
            features = {'label': create_float_feature(label_list[i])}
            for name in feature_map:
                features[name] = create_float_feature(feature_map[name][i])
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
        writer.close()

    if args.parquet:
        features = feature_map
        for k in feature_map:
            features[k] = np.reshape(feature_map[k], -1)
        n = len(label_list)
        index = np.expand_dims(np.arange(n), -1) * np.ones_like(label_list)
        features['qid'] = np.reshape(index, -1)
        features['label_list'] = np.reshape(label_list, -1)
        df = pd.DataFrame(features)
        df.to_parquet(args.des)
