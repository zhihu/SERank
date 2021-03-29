# Copyright 2020 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""TF Ranking sample code for LETOR datasets in LibSVM format.

WARNING: All data sets are loaded into memory in this sample code. It is
for small data sets whose sizes are < 10G.

A note on the LibSVM format:
--------------------------------------------------------------------------
Due to the sparse nature of features utilized in most academic datasets for
learning to rank such as LETOR datasets, data points are represented in the
LibSVM format. In this setting, every line encapsulates features and a (graded)
relevance judgment of a query-document pair. The following illustrates the
general structure:

<relevance int> qid:<query_id int> [<feature_id int>:<feature_value float>]

For example:

1 qid:10 32:0.14 48:0.97  51:0.45
0 qid:10 1:0.15  31:0.75  32:0.24  49:0.6
2 qid:10 1:0.71  2:0.36   31:0.58  51:0.12
0 qid:20 4:0.79  31:0.01  33:0.05  35:0.27
3 qid:20 1:0.42  28:0.79  35:0.30  42:0.76

In the above example, the dataset contains two queries. Query "10" has 3
documents, two of which relevant with grades 1 and 2. Similarly, query "20"
has 1 relevant document. Note that query-document pairs may have different
sets of zero-valued features and as such their feature vectors may only
partly overlap or not at all.
--------------------------------------------------------------------------

You can use TensorBoard to display the training results stored in $OUTPUT_DIR.

Notes:
  * Use --alsologtostderr if the output is not printed into screen.
  * In addition, you can enable multi-objective learning by adding the following
  flags: --secondary_loss=<the secondary loss key>.
"""

from absl import flags

import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr

flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for validation.")
flags.DEFINE_string("test_path", None, "Input file path used for testing.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")

flags.DEFINE_integer("train_batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", 5000, "Number of steps for training.")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "Number of steps for save ckpt.")

flags.DEFINE_float("learning_rate", 0.1, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["256", "128", "64"],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("num_features", 136, "Number of features per document.")
flags.DEFINE_integer("train_list_size", 200, "List size used for training.")
flags.DEFINE_integer("valid_list_size", 800, "List size used for valid and test.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")

flags.DEFINE_string("loss", "softmax_loss",
                    "The RankingLossKey for the primary loss function.")
flags.DEFINE_string(
    "secondary_loss", None, "The RankingLossKey for the secondary loss for "
                            "multi-objective learning.")
flags.DEFINE_float(
    "secondary_loss_weight", 0.5, "The weight for the secondary loss in "
                                  "multi-objective learning.")

flags.DEFINE_bool("serank", False, "serank")
flags.DEFINE_bool("query_label_weight", False, "use query label weight")
flags.DEFINE_float('shrinkage', 2.0, 'se block shrinkage')
flags.DEFINE_bool("shrink_first", False, "se block with shrink first")
flags.DEFINE_bool("without_squeeze", False, "se block without squeeze operation")
flags.DEFINE_bool("without_excite", False, "se block without excite operation")

flags.DEFINE_bool("tfrecord", False, "use tfrecord input")

FLAGS = flags.FLAGS

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"


def _use_multi_head():
    """Returns True if using multi-head."""
    return FLAGS.secondary_loss is not None


class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        if self.iterator_initializer_fn is not None:
            self.iterator_initializer_fn(session)


def example_feature_columns(with_mask=False):
    """Returns the example feature columns."""
    feature_names = ["{}".format(i + 1) for i in range(FLAGS.num_features)]
    if with_mask:
        feature_names.append('mask')
    return {
        name:
            tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
        for name in feature_names
    }


def load_libsvm_data(path, list_size):
    """Returns features and labels in numpy.array."""

    def _parse_line(line):
        """Parses a single line in LibSVM format."""
        tokens = line.split("#")[0].split()
        assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
        label = float(tokens[0])
        qid = tokens[1]
        kv_pairs = [kv.split(":") for kv in tokens[2:]]
        features = {k: float(v) for (k, v) in kv_pairs}
        return qid, features, label

    tf.compat.v1.logging.info("Loading data from {}".format(path))

    # The 0-based index assigned to a query.
    qid_to_index = {}
    # The number of docs seen so far for a query.
    qid_to_ndoc = {}
    # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
    # a shape of [num_queries, list_size]. We use list for each of them due to the
    # unknown number of quries.
    feature_map = {k: [] for k in example_feature_columns()}
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

    tf.compat.v1.logging.info("Number of queries: {}".format(len(qid_to_index)))
    tf.compat.v1.logging.info(
        "Number of documents in total: {}".format(total_docs))
    tf.compat.v1.logging.info(
        "Number of documents discarded: {}".format(discarded_docs))

    # Convert everything to np.array.
    for k in feature_map:
        feature_map[k] = np.array(feature_map[k])
    return feature_map, np.array(label_list)


def get_train_inputs(features, labels, batch_size):
    """Set up training input in batches."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _train_input_fn():
        """Defines training input fn."""
        # mask out invalid samples for serank
        features['mask'] = np.where(labels >= 0, 1.0, 0.0).astype(labels.dtype)

        valid_label = np.where(labels > 0.0, labels, 0.0)
        features['query_label_weight'] = np.sum(valid_label, axis=1, keepdims=True)

        features_placeholder = {
            k: tf.compat.v1.placeholder(v.dtype, v.shape)
            for k, v in six.iteritems(features)
        }

        if _use_multi_head():
            placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
            labels_placeholder = {
                _PRIMARY_HEAD: placeholder,
                _SECONDARY_HEAD: placeholder,
            }
        else:
            labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices(
            (features_placeholder, labels_placeholder))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        if _use_multi_head():
            feed_dict = {
                labels_placeholder[head_name]: labels
                for head_name in labels_placeholder
            }
        else:
            feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k] for k in features_placeholder})

        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _train_input_fn, iterator_initializer_hook


def get_eval_inputs(features, labels):
    """Set up eval inputs in a single batch."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _eval_input_fn():
        """Defines eval input fn."""
        # mask out invalid samples for serank
        features['mask'] = np.where(labels >= 0, 1.0, 0.0).astype(labels.dtype)

        features_placeholder = {
            k: tf.compat.v1.placeholder(v.dtype, v.shape)
            for k, v in six.iteritems(features)
        }

        if _use_multi_head():
            placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
            labels_placeholder = {
                _PRIMARY_HEAD: placeholder,
                _SECONDARY_HEAD: placeholder,
            }
        else:
            labels_placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensors(
            (features_placeholder, labels_placeholder))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        if _use_multi_head():
            feed_dict = {
                labels_placeholder[head_name]: labels
                for head_name in labels_placeholder
            }
        else:
            feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k] for k in features_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _eval_input_fn, iterator_initializer_hook


def make_serving_input_fn():
    """Returns serving input fn to receive tf.Example."""
    feature_spec = tf.feature_column.make_parse_example_spec(
        example_feature_columns().values())
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)


def get_inputs(path, list_size, is_train=False):
    def _tfrecord_input_fn():
        def _parse_fn(proto):
            features = {k: tf.io.FixedLenFeature([list_size], tf.float32) for k in example_feature_columns()}
            features['label'] = tf.io.FixedLenFeature([list_size], tf.float32)
            example = tf.io.parse_single_example(proto, features)
            example['mask'] = tf.cast(example['label'] >= tf.zeros_like(example['label']), tf.float32)
            return example, example['label']

        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(_parse_fn, num_parallel_calls=6)
        if is_train:
            dataset = dataset.repeat().shuffle(1000)
        dataset = dataset.batch(FLAGS.train_batch_size).prefetch(8)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator.get_next()

    return _tfrecord_input_fn, IteratorInitializerHook()


def make_transform_fn():
    """Returns a transform_fn that converts features to dense Tensors."""

    def _transform_fn(features, mode):
        """Defines transform_fn."""
        if mode == tf.estimator.ModeKeys.PREDICT:
            # We expect tf.Example as input during serving. In this case, group_size
            # must be set to 1.
            if FLAGS.group_size != 1:
                raise ValueError(
                    "group_size should be 1 to be able to export model, but get %s" %
                    FLAGS.group_size)
            context_features, example_features = (
                tfr.feature.encode_pointwise_features(
                    features=features,
                    context_feature_columns=None,
                    example_feature_columns=example_feature_columns(),
                    mode=mode,
                    scope="transform_layer"))
        else:
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=None,
                example_feature_columns=example_feature_columns(with_mask=True),
                mode=mode,
                scope="transform_layer")

        return context_features, example_features

    return _transform_fn


def make_se_block_fn(shrinkage=1.0, shrink_first=False, without_squeeze=False, without_excite=False):
    def squeeze(cur_layer, last_dim, mask=None, list_size=1):
        # output shape: [batch_size, 1, last_dim]
        cur_layer = tf.reshape(cur_layer, [-1, list_size, last_dim])
        if mask is None:
            cur_layer = tf.reduce_mean(cur_layer, axis=1)
        else:
            # when training & eval, mask out padding records
            mask = tf.reshape(mask, [-1, list_size, 1])
            cur_layer = tf.reduce_sum(cur_layer * mask, axis=1) / tf.reduce_sum(mask + 1e-6, axis=1)
        return cur_layer

    def se_block_fn(input_layer, layer_width, mask=None, list_size=1):
        # input_layer: [batch_size * list_size, dim]
        # mask: [batch_size * list_size, 1]
        dim = int(layer_width / shrinkage)
        if shrink_first:
            cur_layer = tf.compat.v1.layers.dense(input_layer, units=dim)
            cur_layer = tf.nn.relu(cur_layer)
            if not without_squeeze:
                cur_layer = squeeze(cur_layer, dim, mask, list_size)
                cur_layer = tf.reshape(tf.tile(cur_layer, [1, list_size]), [-1, list_size, dim])
        else:
            cur_layer = input_layer
            if not without_squeeze:
                cur_layer = squeeze(cur_layer, layer_width, mask, list_size)
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=dim)
            cur_layer = tf.nn.relu(cur_layer)
            cur_layer = tf.reshape(tf.tile(cur_layer, [1, list_size]), [-1, list_size, dim])
        cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
        if without_excite:
            cur_layer = tf.concat([input_layer, cur_layer], axis=-1)
        else:
            excitation = tf.reshape(tf.nn.sigmoid(cur_layer), [-1, layer_width])
            cur_layer = input_layer * excitation
        return cur_layer

    return se_block_fn


def make_score_fn(se_block_fn=None):
    """Returns a groupwise score fn to build `EstimatorSpec`."""

    def _score_fn(unused_context_features, group_features, mode, params,
                  unused_config):
        """Defines the network to score a group of documents."""
        with tf.compat.v1.name_scope("input_layer"):
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(example_feature_columns())
            ]
            input_layer = tf.concat(group_input, 1)
            tf.compat.v1.summary.scalar("input_sparsity",
                                        tf.nn.zero_fraction(input_layer))
            tf.compat.v1.summary.scalar("input_max",
                                        tf.reduce_max(input_tensor=input_layer))
            tf.compat.v1.summary.scalar("input_min",
                                        tf.reduce_min(input_tensor=input_layer))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = tf.compat.v1.layers.batch_normalization(
            input_layer, training=is_training)
        for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                        tf.nn.zero_fraction(cur_layer))
            if se_block_fn:
                if is_training:
                    list_size = params['train_list_size']
                else:
                    list_size = params.get('list_size', 1)
                cur_layer = se_block_fn(cur_layer, layer_width, group_features.get('mask'), list_size)
        cur_layer = tf.compat.v1.layers.dropout(
            cur_layer, rate=FLAGS.dropout_rate, training=is_training)
        logits = tf.compat.v1.layers.dense(cur_layer, units=FLAGS.group_size)
        if _use_multi_head():
            # Duplicate the logits for both heads.
            return {_PRIMARY_HEAD: logits, _SECONDARY_HEAD: logits}
        else:
            return logits

    return _score_fn


def get_eval_metric_fns():
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
            tfr.metrics.RankingMetricKey.ARP,
            tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
        ]
    })
    metric_fns.update({
        "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [1, 3, 5, 10]
    })
    return metric_fns


def train_and_eval():
    """Train and Evaluate."""
    if FLAGS.tfrecord:
        train_input_fn, train_hook = get_inputs(FLAGS.train_path, FLAGS.train_list_size, is_train=True)
        vali_input_fn, vali_hook = get_inputs(FLAGS.vali_path, FLAGS.valid_list_size)
        test_input_fn, test_hook = get_inputs(FLAGS.test_path, FLAGS.valid_list_size)
    else:
        features, labels = load_libsvm_data(FLAGS.train_path, FLAGS.train_list_size)
        train_input_fn, train_hook = get_train_inputs(features, labels,
                                                      FLAGS.train_batch_size)

        features_vali, labels_vali = load_libsvm_data(FLAGS.vali_path,
                                                      FLAGS.valid_list_size)
        vali_input_fn, vali_hook = get_eval_inputs(features_vali, labels_vali)

        features_test, labels_test = load_libsvm_data(FLAGS.test_path,
                                                      FLAGS.valid_list_size)
        test_input_fn, test_hook = get_eval_inputs(features_test, labels_test)

    optimizer = tf.compat.v1.train.AdagradOptimizer(
        learning_rate=FLAGS.learning_rate)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([minimize_op, update_ops])
        return train_op

    weights_feature_name = 'query_label_weight' if FLAGS.query_label_weight else None

    if _use_multi_head():
        primary_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(FLAGS.loss, weights_feature_name=weights_feature_name),
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn,
            name=_PRIMARY_HEAD)
        secondary_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(FLAGS.secondary_loss, weights_features_name=weights_feature_name),
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn,
            name=_SECONDARY_HEAD)
        ranking_head = tfr.head.create_multi_ranking_head(
            [primary_head, secondary_head], [1.0, FLAGS.secondary_loss_weight])
    else:
        ranking_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn)

    se_block_fn = None
    if FLAGS.serank:
        se_block_fn = make_se_block_fn(shrinkage=FLAGS.shrinkage,
                                       shrink_first=FLAGS.shrink_first,
                                       without_squeeze=FLAGS.without_squeeze,
                                       without_excite=FLAGS.without_excite)

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(se_block_fn),
            group_size=FLAGS.group_size,
            transform_fn=make_transform_fn(),
            ranking_head=ranking_head),
        config=tf.estimator.RunConfig(
            FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps),
        params={'train_list_size': FLAGS.train_list_size,
                'list_size': FLAGS.valid_list_size})

    early_stopping_hook = tf.estimator.experimental.stop_if_no_increase_hook(
        estimator=estimator,
        metric_name='metric/ndcg@5',
        max_steps_without_increase=FLAGS.save_checkpoints_steps * 3)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        hooks=[train_hook, early_stopping_hook],
        max_steps=FLAGS.num_train_steps)

    # Export model to accept tf.Example when group_size = 1.
    if FLAGS.group_size == 1:
        vali_spec = tf.estimator.EvalSpec(
            input_fn=vali_input_fn,
            hooks=[vali_hook],
            steps=1,
            exporters=tf.estimator.LatestExporter(
                "latest_exporter",
                serving_input_receiver_fn=make_serving_input_fn()),
            start_delay_secs=0,
            throttle_secs=30)
    else:
        vali_spec = tf.estimator.EvalSpec(
            input_fn=vali_input_fn,
            hooks=[vali_hook],
            steps=1,
            start_delay_secs=0,
            throttle_secs=30)

    # Train and validate
    tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

    # Evaluate on the test data.
    from tensorflow.python.training import checkpoint_management
    latest_path = checkpoint_management.latest_checkpoint(FLAGS.output_dir)
    path, step = latest_path.rsplit('-', 1)
    step = int(step) - FLAGS.save_checkpoints_steps * 3
    best_path = path + '-' + str(step)
    estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook], checkpoint_path=best_path)


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    train_and_eval()


if __name__ == "__main__":
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("vali_path")
    flags.mark_flag_as_required("test_path")
    flags.mark_flag_as_required("output_dir")
    tf.random.set_random_seed(0)
    tf.compat.v1.app.run()
