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
from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.learn.pytorch.pytorch_horovod_estimator import PyTorchHorovodEstimator


class Estimator(object):
    def fit(self, data, epochs, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    @staticmethod
    def from_model_creator(*,
                           model_creator,
                           optimizer_creator,
                           loss_creator=None,
                           scheduler_creator=None,
                           training_operator_cls=TrainingOperator,
                           initialization_hook=None,
                           config=None,
                           scheduler_step_freq="batch",
                           backend="ray"):
        assert backend == "ray", "only ray backend is supported for now"
        return PyTorchHorovodEstimatorWrapper(model_creator=model_creator,
                                              optimizer_creator=optimizer_creator,
                                              loss_creator=loss_creator,
                                              scheduler_creator=scheduler_creator,
                                              training_operator_cls=training_operator_cls,
                                              initialization_hook=initialization_hook,
                                              config=config,
                                              scheduler_step_freq=scheduler_step_freq)


class PyTorchHorovodEstimatorWrapper(Estimator):
    def __init__(self,
                 *,
                 model_creator,
                 optimizer_creator,
                 loss_creator=None,
                 scheduler_creator=None, 
                 training_operator_cls=TrainingOperator,
                 initialization_hook=None,
                 config=None,
                 scheduler_step_freq="batch"):
        self.estimator = PyTorchHorovodEstimator(model_creator=model_creator,
                                                 optimizer_creator=optimizer_creator,
                                                 loss_creator=loss_creator,
                                                 scheduler_creator=scheduler_creator,
                                                 training_operator_cls=training_operator_cls,
                                                 initialization_hook=initialization_hook,
                                                 config=config,
                                                 scheduler_step_freq=scheduler_step_freq)

    def fit(self, data, epochs=1, num_steps=None, profile=False, reduce_results=True, info=None):
        """

        :param data: (callable) a funtion that takes a config dict as input and return a data
            loader containing the training data.
        :param epochs: (int) Number of epochs to train the model
        :param num_steps: (int) Number of batches to compute update steps on.
            This corresponds also to the number of times `TrainingOperator.train_batch`` is called.
        :param profile: (bool) Returns time stats for the training procedure.
        :param reduce_results: (bool) Whether to average all metrics across all workers into one
            dict. If a metric is a non-numerical value (or nested dictionaries), one value will be
            randomly selected among the workers. If False, returns a list of dicts.
        :param info: (dict) Optional dictionary passed to the training operator for ``train_epoch``
            and ``train_batch``.
        :return: (list) A list of stats whose length will be equal to ``epochs``.
                stats is a dictionary of metrics for training.
                    You can provide custom metrics by passing in a custom
                    ``training_operator_cls``. If ``reduce_results=False``,
                    this will return a list of metric dictionaries whose
                    length will be equal to ``num_workers``.
        """
        stats_list = list()
        for i in range(epochs):
            stats = self.estimator.train(data_creator=data, num_steps=num_steps, profile=profile,
                                         reduce_results=reduce_results, info=info)
            stats_list.append(stats)
        return stats_list

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, num_steps=None, profile=False, info=None):
        """

        :param data: (callable) a funtion that takes a config dict as input and return
            a data loader containing the validation data.
        :param num_steps: (int) Number of batches to compute update steps on.
               This corresponds also to the number of times ``TrainingOperator.validate_batch``
               is called.
        :param profile: (bool) Returns time stats for the evaluation procedure.
        :param info: (dict) Optional dictionary passed to the training operator for `validate`
            and `validate_batch`.
        :return: A dictionary of metrics for validation.
            You can provide custom metrics by passing in a custom ``training_operator_cls``.
        """
        return self.estimator.validate(data_creator=data, num_steps=num_steps, profile=profile,
                                       info=info)
