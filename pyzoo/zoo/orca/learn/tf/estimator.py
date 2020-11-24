#
# Copyright 2018 Analytics Zoo Authors.
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
#
from pyspark.sql import DataFrame

from bigdl.optim.optimizer import MaxEpoch

from zoo.tfpark.tf_dataset import TFNdarrayDataset
from zoo.tfpark.model import _standarize_feature_label_dataset

from zoo.common.utils import load_from_file
from zoo.orca.data.tf.data import Dataset, TFDataDataset2
from zoo.orca.learn.tf.utils import *
from zoo.orca.learn.trigger import Trigger
from zoo.orca.learn.utils import find_latest_checkpoint, convert_predict_to_xshard
from zoo.tfpark import KerasModel
from zoo.tfpark import TFOptimizer, TFNet, ZooOptimizer
from zoo.tfpark.tf_optimizer import StatelessMetric
from zoo.tfpark.utils import evaluate_metrics
from zoo.util import nest
from zoo.util.tf import save_tf_checkpoint
from zoo.orca.learn.spark_estimator import Estimator as SparkEstimator


class Estimator(object):
    @staticmethod
    def from_graph(*, inputs, outputs=None,
                   labels=None, loss=None, optimizer=None,
                   clip_norm=None, clip_value=None,
                   metrics=None, updates=None,
                   sess=None, model_dir=None, backend="bigdl"):
        """
        Create an Estimator for tesorflow graph.
        :param inputs: input tensorflow tensors.
        :param outputs: output tensorflow tensors.
        :param labels: label tensorflow tensors.
        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optimizer: tensorflow optimization method.
        :param clip_norm: float >= 0. Gradients will be clipped when their L2 norm exceeds
        this value.
        :param clip_value:  a float >= 0 or a tuple of two floats.
        If clip_value is a float, gradients will be clipped when their absolute value
        exceeds this value.
        If clip_value is a tuple of two floats, gradients will be clipped when their value less
        than clip_value[0] or larger than clip_value[1].
        :param metrics: metric tensor.
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model,
        you should use the Session to load the pre-trained variables and pass it to estimator
        :param model_dir: location to save model checkpoint and summaries.
        :param backend: backend for estimator. Now it only can be "bigdl".
        :return: an Estimator object.
        """
        import warnings
        warnings.warn("This method will be deprecated, please "
                      "from zoo.orca.learn.spark_estimator import Estimator and use "
                      "Estimator.from_tf_graph instead", DeprecationWarning)
        assert backend == "bigdl", "only bigdl backend is supported for now"
        return TFOptimizerWrapper(inputs=inputs,
                                  outputs=outputs,
                                  labels=labels,
                                  loss=loss,
                                  optimizer=optimizer,
                                  clip_norm=clip_norm,
                                  clip_value=clip_value,
                                  metrics=metrics, updates=updates,
                                  sess=sess,
                                  model_dir=model_dir
                                  )

    @staticmethod
    def from_keras(keras_model, metrics=None, model_dir=None, optimizer=None, backend="bigdl"):
        """
        Create an Estimator from a tensorflow.keras model. The model must be compiled.
        :param keras_model: the tensorflow.keras model, which must be compiled.
        :param metrics: user specified metric.
        :param model_dir: location to save model checkpoint and summaries.
        :param optimizer: an optional bigdl optimMethod that will override the optimizer in
                          keras_model.compile
        :param backend: backend for estimator. Now it only can be "bigdl".
        :return: an Estimator object.
        """
        import warnings
        warnings.warn("This method will be deprecated, please "
                      "from zoo.orca.learn.spark_estimator import Estimator and use "
                      "Estimator.from_keras instead", DeprecationWarning)
        assert backend == "bigdl", "only bigdl backend is supported for now"
        return TFKerasWrapper(keras_model, metrics, model_dir, optimizer)

    @staticmethod
    def load_keras_model(path):
        """
        Create Estimator by loading an existing keras model (with weights) from HDF5 file.

        :param path: String. The path to the pre-defined model.
        :return: Orca TF Estimator.
        """
        import warnings
        warnings.warn("This method will be deprecated, please "
                      "from zoo.orca.learn.spark_estimator import Estimator and use "
                      "Estimator.load_keras_model instead", DeprecationWarning)
        from tensorflow.python.keras import models

        def load_func(file_path):
            return models.load_model(file_path)

        model = load_from_file(load_func, path)
        return Estimator.from_keras(keras_model=model)


def is_tf_data_dataset(data):
    is_dataset = isinstance(data, tf.data.Dataset)
    is_dataset_v2 = isinstance(data, tf.python.data.ops.dataset_ops.DatasetV2)
    return is_dataset or is_dataset_v2


def to_dataset(data, batch_size, batch_per_thread, validation_data,
               feature_cols, labels_cols, hard_code_batch_size,
               sequential_order, shuffle, auto_shard_files):
    # todo wrap argument into kwargs
    if validation_data:
        if isinstance(data, SparkXShards):
            assert isinstance(validation_data, SparkXShards), \
                "train data and validation data should be both SparkXShards"
        if isinstance(data, Dataset):
            assert isinstance(validation_data, Dataset), \
                "train data and validation data should be both orca.data.tf.Dataset"
        if isinstance(data, DataFrame):
            assert isinstance(validation_data, DataFrame), \
                "train data and validation data should be both Spark DataFrame"
        if isinstance(data, tf.data.Dataset):
            assert isinstance(validation_data, tf.data.Dataset), \
                "train data and validation data should be both tf.data.Dataset"

    if isinstance(data, SparkXShards):
        dataset = xshards_to_tf_dataset(data,
                                        batch_size,
                                        batch_per_thread,
                                        validation_data,
                                        hard_code_batch_size=hard_code_batch_size,
                                        sequential_order=sequential_order,
                                        shuffle=shuffle)
    elif isinstance(data, Dataset):
        dataset = TFDataDataset2(data, batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 validation_dataset=validation_data)
    elif isinstance(data, DataFrame):
        dataset = TFDataset.from_dataframe(data, feature_cols, labels_cols,
                                           batch_size,
                                           batch_per_thread,
                                           hard_code_batch_size,
                                           validation_data,
                                           sequential_order,
                                           shuffle
                                           )
    elif is_tf_data_dataset(data):
        dataset = TFDataset.from_tf_data_dataset(data,
                                                 batch_size,
                                                 batch_per_thread,
                                                 hard_code_batch_size,
                                                 validation_data,
                                                 sequential_order,
                                                 shuffle, auto_shard_files=auto_shard_files)
    else:
        raise ValueError("data must be SparkXShards or orca.data.tf.Dataset or "
                         "Spark DataFrame or tf.data.Dataset")

    return dataset


class TFOptimizerWrapper(SparkEstimator):
    def __init__(self, *, inputs, outputs, labels, loss,
                 optimizer, clip_norm, clip_value,
                 metrics,
                 updates, sess,
                 model_dir
                 ):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        self.use_bigdl_optim = False
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        if optimizer is not None:
            from zoo.orca.learn.optimizers import Optimizer
            if isinstance(optimizer, Optimizer):
                self.train_op = None
                self.optimizer = optimizer.get_optimizer()
                self.use_bigdl_optim = True
            else:
                assert isinstance(optimizer, tf.train.Optimizer), \
                    "optimizer is of type {}, ".format(type(optimizer)) + \
                    "it should be an instance of tf.train.Optimizer"
                self.optimizer = ZooOptimizer(optimizer)
                if clip_norm or clip_value:
                    gvs = self.optimizer.compute_gradients(self.loss)
                    if clip_norm:
                        gvs = [(tf.clip_by_norm(g_v[0], clip_norm), g_v[1]) for g_v in gvs]
                    if clip_value:
                        if isinstance(clip_value, tuple):
                            assert len(clip_value) == 2 and clip_value[0] < clip_value[1], \
                                "clip value should be (clip_min, clip_max)"
                            gvs = [(tf.clip_by_value(g_v[0], clip_value[0], clip_value[1]), g_v[1])
                                   for g_v in gvs]
                        if isinstance(clip_value, (int, float)):
                            assert clip_value > 0, "clip value should be larger than 0"
                            gvs = [(tf.clip_by_value(g_v[0], -clip_value, clip_value), g_v[1])
                                   for g_v in gvs]
                        else:
                            raise Exception("clip_value should be a tuple or one number")
                    self.train_op = self.optimizer.apply_gradients(gvs)
                else:
                    self.train_op = self.optimizer.minimize(self.loss)
        else:
            self.optimizer = None
            self.train_op = None
        self.metrics = metrics
        self.updates = updates
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.model_dir = model_dir
        self.load_checkpoint = False
        self.tf_optimizer = None
        self.log_dir = None
        self.app_name = None

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            labels_cols=None,
            validation_data=None,
            hard_code_batch_size=False,
            auto_shard_files=True,
            session_config=None,
            feed_dict=None,
            checkpoint_trigger=None
            ):
        """
        Train this graph model with train data.
        :param data: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        If data is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
        If data is tf.data.Dataset, each element is a tuple of input tensors.
        :param epochs: number of epochs to train.
        :param batch_size: total batch size for each iteration.
        :param feature_cols: feature column names if train data is Spark DataFrame.
        :param labels_cols: label column names if train data is Spark DataFrame.
        :param validation_data: validation data. Validation data type should be the same
        as train data.
        :param hard_code_batch_size: whether hard code batch size for training. Default is False.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
        and apply sharding on files, otherwise sharding on records. Default is True.
        :param session_config: tensorflow session configuration for training.
        Should be object of tf.ConfigProto
        :param feed_dict: a dictionary. The key is TensorFlow tensor, usually a
        placeholder, the value of the dictionary is a tuple of two elements. The first one of
        the tuple is the value to feed to the tensor in training phase and the second one
        is the value to feed to the tensor in validation phase.
        :param checkpoint_trigger: when to trigger checkpoint during training.
        Should be a zoo.orca.learn.trigger, like EveryEpoch(), SeveralIteration(num_iterations),etc.
        """

        assert self.labels is not None, \
            "labels is None; it should not be None in training"
        assert self.loss is not None, \
            "loss is None; it should not be None in training"
        assert self.optimizer is not None, \
            "optimizer is None; it should not be None in training"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in training"

        if checkpoint_trigger is not None:
            checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=False, shuffle=True,
                             auto_shard_files=auto_shard_files
                             )

        if feed_dict is not None:
            tensor_with_value = {key: (value[0], value[1]) for key, value in feed_dict.items()}
        else:
            tensor_with_value = None

        if self.use_bigdl_optim:
            self.tf_optimizer = TFOptimizer.from_loss(
                self.loss, self.optimizer,
                session=self.sess, inputs=(self.inputs, self.labels), dataset=dataset,
                clip_norm=self.clip_norm, clip_value=self.clip_value, metrics=self.metrics,
                tensor_with_value=tensor_with_value, session_config=session_config,
                model_dir=self.model_dir, updates=self.updates)
        else:

            self.tf_optimizer = TFOptimizer.from_train_op(
                train_op=self.train_op,
                loss=self.loss,
                inputs=self.inputs,
                labels=self.labels,
                dataset=dataset,
                metrics=self.metrics,
                updates=self.updates, sess=self.sess,
                tensor_with_value=tensor_with_value,
                session_config=session_config,
                model_dir=self.model_dir)

        if self.load_checkpoint:
            self.tf_optimizer.load_checkpoint(self.checkpoint_path, self.checkpoint_version)

        if self.log_dir and self.app_name:
            self.tf_optimizer.estimator.set_tensorboad(self.log_dir, self.app_name)

        self.tf_optimizer.optimize(end_trigger=MaxEpoch(epochs),
                                   checkpoint_trigger=checkpoint_trigger)
        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                hard_code_batch_size=False,
                auto_shard_files=True,
                ):
        """
        Predict input data
        :param data: data to be predicted. It can be XShards, Spark DataFrame.
        If data is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays}.
        :param batch_size: batch size per thread
        :param feature_cols: list of feature column names if input data is Spark DataFrame.
        :param hard_code_batch_size: whether to hard code batch size for prediction.
         The default value is False.
        :return: predicted result.
         If input data is XShards or tf.data.Dataset, the predict result is a XShards,
         and the schema for each result is: {'prediction': predicted numpy array or
          list of predicted numpy arrays}.
         If input data is Spark DataFrame, the predict result is a DataFrame which includes original
         columns plus 'prediction' column. The 'prediction' column can be FloatType, VectorUDT
         or Array of VectorUDT depending on model outputs shape.
        """

        assert self.outputs is not None, \
            "output is None, it should not be None in prediction"
        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"

        assert not is_tf_data_dataset(data), "tf.data.Dataset currently cannot be used for" \
                                             "estimator prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=None,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True,
                             shuffle=False,
                             auto_shard_files=auto_shard_files,
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_outputs = nest.flatten(self.outputs)
        tfnet = TFNet.from_session(sess=self.sess, inputs=flat_inputs, outputs=flat_outputs)
        predicted_rdd = tfnet.predict(dataset)
        if isinstance(data, DataFrame):
            return convert_predict_to_dataframe(data, predicted_rdd)
        elif isinstance(data, SparkXShards) or isinstance(data, tf.data.Dataset):
            return convert_predict_to_xshard(predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=32,
                 feature_cols=None,
                 labels_cols=None,
                 hard_code_batch_size=False,
                 auto_shard_files=True,
                 ):
        """
        Evaluate model.
        :param data: evaluation data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        If data is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
        If data is tf.data.Dataset, each element is a tuple of input tensors.
        :param batch_size: batch size per thread.
        :param feature_cols: feature_cols: feature column names if train data is Spark DataFrame.
        :param labels_cols: label column names if train data is Spark DataFrame.
        :param hard_code_batch_size: whether to hard code batch size for evaluation.
        :return: evaluation result as a dictionary of {'metric name': metric value}
        """

        assert self.metrics is not None, \
            "metrics is None, it should not be None in evaluate"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True,
                             shuffle=False,
                             auto_shard_files=auto_shard_files,
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_labels = nest.flatten(self.labels)

        return evaluate_metrics(flat_inputs + flat_labels,
                                sess=self.sess,
                                dataset=dataset, metrics=self.metrics)

    def save_tf_checkpoint(self, path):
        save_tf_checkpoint(self.sess, path)

    def get_model(self):
        pass

    def save(self, model_path):
        self.save_tf_checkpoint(model_path)

    def load(self, checkpoint, **kwargs):
        self.load_latest_orca_checkpoint(checkpoint)

    def set_tensorboard(self, log_dir, app_name):
        self.log_dir = log_dir
        self.app_name = app_name

    def clear_gradient_clipping(self):
        raise NotImplementedError

    def set_constant_gradient_clipping(self, min, max):
        raise NotImplementedError

    def set_l2_norm_gradient_clipping(self, clip_norm):
        raise NotImplementedError

    def get_train_summary(self, tag=None):
        if self.tf_optimizer:
            return self.tf_optimizer.estimator.get_train_summary(tag)

        return None

    def get_validation_summary(self, tag=None):
        if self.tf_optimizer:
            for val_method in self.tf_optimizer.tf_model.val_methods:
                if isinstance(val_method, StatelessMetric):
                    if tag == val_method.name:
                        return self.tf_optimizer.estimator.get_validation_summary(tag)
                else:
                    if tag == str(val_method.val_method):
                        return self.tf_optimizer.estimator.\
                            get_validation_summary("{} {}".format(val_method.name, tag))
                continue
        return None

    def load_orca_checkpoint(self, path, version):
        self.load_checkpoint = True
        self.checkpoint_path = path
        self.checkpoint_version = version

    def load_latest_orca_checkpoint(self, path):
        ckpt_path, _, version = find_latest_checkpoint(path, model_type="tf")
        if ckpt_path is None:
            raise Exception("Cannot find checkpoint")
        self.load_orca_checkpoint(ckpt_path, version)


class TFKerasWrapper(SparkEstimator):
    def __init__(self, keras_model, metrics, model_dir, optimizer):
        self.model = KerasModel(keras_model, model_dir)
        self.load_checkpoint = False
        self.metrics = metrics
        self.tf_optimizer = None
        self.optimizer = optimizer
        from zoo.orca.learn.optimizers import Optimizer
        if self.optimizer is not None and isinstance(self.optimizer, Optimizer):
            self.optimizer = self.optimizer.get_optimizer()
        self.log_dir = None
        self.app_name = None
        self.clip_norm = None
        self.clip_min = None
        self.clip_max = None

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            labels_cols=None,
            validation_data=None,
            hard_code_batch_size=False,
            session_config=None,
            checkpoint_trigger=None,
            auto_shard_files=True,
            ):
        """
        Train this keras model with train data.
        :param data: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        If data is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
        If data is tf.data.Dataset, each element is [feature tensor tuple, label tensor tuple]
        :param epochs: number of epochs to train.
        :param batch_size: total batch size for each iteration.
        :param feature_cols: feature column names if train data is Spark DataFrame.
        :param labels_cols: label column names if train data is Spark DataFrame.
        :param validation_data: validation data. Validation data type should be the same
        as train data.
        :param hard_code_batch_size: whether hard code batch size for training. Default is False.
        :param session_config: tensorflow session configuration for training.
        Should be object of tf.ConfigProto
        :param checkpoint_trigger: when to trigger checkpoint during training.
        Should be a zoo.orca.learn.trigger, like EveryEpoch(), SeveralIteration(num_iterations),etc.
        """

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in training"

        if isinstance(data, tf.data.Dataset):
            assert isinstance(data.element_spec, tuple), \
                "If data is tf.data.Dataset, each element should be " \
                "(feature tensors, label tensor), where each feature/label tensor can be " \
                "either a single tensor or a tuple of tensors"
            if validation_data is not None:
                assert isinstance(validation_data, tf.data.Dataset), \
                    "train data and validation data should be both tf.data.Dataset"
                assert isinstance(validation_data.element_spec, tuple), \
                    "If validation_data is tf.data.Dataset, each element should be " \
                    "(feature tensors, label tensor), where each feature/label tensor can be " \
                    "either a single tensor or a tuple of tensors"

        if checkpoint_trigger is not None:
            checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=False, shuffle=True,
                             auto_shard_files=auto_shard_files)
        if isinstance(dataset, TFNdarrayDataset):
            dataset = _standarize_feature_label_dataset(dataset, self.model.model)

        self.tf_optimizer = TFOptimizer.from_keras(self.model.model, dataset,
                                                   model_dir=self.model.model_dir,
                                                   session_config=session_config,
                                                   metrics=self.metrics,
                                                   optimizer=self.optimizer)

        if self.clip_norm:
            self.tf_optimizer.set_gradient_clipping_by_l2_norm(clip_norm=self.clip_norm)
        if self.clip_min and self.clip_max:
            self.tf_optimizer.set_constant_gradient_clipping(self.clip_min, self.clip_max)

        if self.load_checkpoint:
            self.tf_optimizer.load_checkpoint(self.checkpoint_path, self.checkpoint_version)

        if self.log_dir and self.app_name:
            self.tf_optimizer.estimator.set_tensorboad(self.log_dir, self.app_name)

        self.tf_optimizer.optimize(MaxEpoch(epochs), checkpoint_trigger=checkpoint_trigger)

        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                hard_code_batch_size=False,
                auto_shard_files=True,
                ):
        """
        Predict input data
        :param data: data to be predicted.
        It can be XShards, Spark DataFrame, or tf.data.Dataset.
        If data is XShard, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays}.
        If data is tf.data.Dataset, each element is feature tensor tuple
        :param batch_size: batch size per thread
        :param feature_cols: list of feature column names if input data is Spark DataFrame.
        :param hard_code_batch_size: if require hard code batch size for prediction.
         The default value is False.
        :return: predicted result.
         If input data is XShards or tf.data.Dataset, the predict result is also a XShards,
         and the schema for each result is: {'prediction': predicted numpy array or
          list of predicted numpy arrays}.
         If input data is Spark DataFrame, the predict result is a DataFrame which includes
         original columns plus 'prediction' column. The 'prediction' column can be FloatType,
         VectorUDT or Array of VectorUDT depending on model outputs shape.
        """

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=None,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True, shuffle=False,
                             auto_shard_files=auto_shard_files,
                             )

        predicted_rdd = self.model.predict(dataset, batch_size)
        if isinstance(data, DataFrame):
            return convert_predict_to_dataframe(data, predicted_rdd)
        elif isinstance(data, SparkXShards) or isinstance(data, tf.data.Dataset):
            return convert_predict_to_xshard(predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=4,
                 feature_cols=None,
                 labels_cols=None,
                 hard_code_batch_size=False,
                 auto_shard_files=True
                 ):
        """
        Evaluate model.
        :param data: evaluation data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        If data is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
        If data is tf.data.Dataset, each element is [feature tensor tuple, label tensor tuple]
        :param batch_size: batch size per thread.
        :param feature_cols: feature_cols: feature column names if train data is Spark DataFrame.
        :param labels_cols: label column names if train data is Spark DataFrame.
        :param hard_code_batch_size: whether to hard code batch size for evaluation.
        :return: evaluation result as a dictionary of {'metric name': metric value}
        """

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert labels_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, labels_cols=labels_cols,
                             hard_code_batch_size=hard_code_batch_size,
                             sequential_order=True, shuffle=False,
                             auto_shard_files=auto_shard_files
                             )

        return self.model.evaluate(dataset, batch_per_thread=batch_size)

    def save_keras_model(self, path, overwrite=True):
        self.model.save_model(path, overwrite=overwrite)

    def get_model(self):
        pass

    def save(self, model_path, overwrite=True):
        self.save_keras_model(model_path, overwrite=True)

    def load(self, checkpoint, **kwargs):
        self.load_latest_orca_checkpoint(checkpoint)

    def set_tensorboard(self, log_dir, app_name):
        self.log_dir = log_dir
        self.app_name = app_name

    def clear_gradient_clipping(self):
        self.clip_norm = None
        self.clip_min = None
        self.clip_max = None

    def set_constant_gradient_clipping(self, min, max):
        assert min > 0, "clip value should be larger than 0"
        assert min < max, "clip max should be larger than clip min"
        self.clip_min = min
        self.clip_max = max

    def set_l2_norm_gradient_clipping(self, clip_norm):
        self.clip_norm = clip_norm

    def get_train_summary(self, tag=None):
        if self.tf_optimizer:
            return self.tf_optimizer.estimator.get_train_summary(tag)

        return None

    def get_validation_summary(self, tag=None):
        if self.tf_optimizer:
            for val_method in self.tf_optimizer.tf_model.val_methods:
                if isinstance(val_method, StatelessMetric):
                    if tag == val_method.name:
                        return self.tf_optimizer.estimator.get_validation_summary(tag)
                else:
                    if tag == str(val_method.val_method):
                        return self.tf_optimizer.estimator.\
                            get_validation_summary("{} {}".format(val_method.name, tag))
                continue
        return None

    def load_orca_checkpoint(self, path, version):
        self.load_checkpoint = True
        self.checkpoint_path = path
        self.checkpoint_version = version

    def load_latest_orca_checkpoint(self, path):
        ckpt_path, _, version = find_latest_checkpoint(path, model_type="tf")
        if ckpt_path is None:
            raise Exception("Cannot find checkpoint")
        self.load_orca_checkpoint(ckpt_path, version)

    def save_keras_weights(self, filepath, overwrite=True, save_format=None):
        self.model.save_weights(filepath, overwrite, save_format)

    def load_keras_weights(self, filepath, by_name=False):
        self.model.load_weights(filepath, by_name)
