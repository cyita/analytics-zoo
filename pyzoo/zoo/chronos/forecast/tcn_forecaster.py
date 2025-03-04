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

from zoo.chronos.forecast.base_forecaster import BasePytorchForecaster
from zoo.chronos.forecast.utils import set_pytorch_seed
from zoo.chronos.model.tcn import TCNPytorch
from zoo.chronos.model.tcn import model_creator, optimizer_creator, loss_creator


class TCNForecaster(BasePytorchForecaster):
    """
        Example:
            >>> #The dataset is split into x_train, x_val, x_test, y_train, y_val, y_test
            >>> forecaster = TCNForecaster(past_seq_len=24,
                                           future_seq_len=5,
                                           input_feature_num=1,
                                           output_feature_num=1,
                                           ...)
            >>> forecaster.fit((x_train, y_train))
            >>> forecaster.to_local()  # if you set distributed=True
            >>> test_pred = forecaster.predict(x_test)
            >>> test_eval = forecaster.evaluate((x_test, y_test))
            >>> forecaster.save({ckpt_name})
            >>> forecaster.restore({ckpt_name})
    """
    def __init__(self,
                 past_seq_len,
                 future_seq_len,
                 input_feature_num,
                 output_feature_num,
                 num_channels=[30]*7,
                 kernel_size=3,
                 repo_initialization=True,
                 dropout=0.1,
                 optimizer="Adam",
                 loss="mse",
                 lr=0.001,
                 metrics=["mse"],
                 seed=None,
                 distributed=False,
                 workers_per_node=1,
                 distributed_backend="torch_distributed"):
        """
        Build a TCN Forecast Model.

        TCN Forecast may fall into local optima. Please set repo_initialization
        to False to alleviate the issue. You can also change a random seed to
        work around.

        :param past_seq_len: Specify the history time steps (i.e. lookback).
        :param future_seq_len: Specify the output time steps (i.e. horizon).
        :param input_feature_num: Specify the feature dimension.
        :param output_feature_num: Specify the output dimension.
        :param num_channels: Specify the convolutional layer filter number in
               TCN's encoder. This value defaults to [30]*7.
        :param kernel_size: Specify convolutional layer filter height in TCN's
               encoder. This value defaults to 3.
        :param repo_initialization: if to use framework default initialization,
               True to use paper author's initialization and False to use the
               framework's default initialization. The value defaults to True.
        :param dropout: Specify the dropout close possibility (i.e. the close
               possibility to a neuron). This value defaults to 0.1.
        :param optimizer: Specify the optimizer used for training. This value
               defaults to "Adam".
        :param loss: Specify the loss function used for training. This value
               defaults to "mse". You can choose from "mse", "mae" and
               "huber_loss".
        :param lr: Specify the learning rate. This value defaults to 0.001.
        :param metrics: A list contains metrics for evaluating the quality of
               forecasting. You may only choose from "mse" and "mae" for a
               distributed forecaster. You may choose from "mse", "me", "mae",
               "mse","rmse","msle","r2", "mpe", "mape", "mspe", "smape", "mdape"
               and "smdape" for a non-distributed forecaster.
        :param seed: int, random seed for training. This value defaults to None.
        :param distributed: bool, if init the forecaster in a distributed
               fashion. If True, the internal model will use an Orca Estimator.
               If False, the internal model will use a pytorch model. The value
               defaults to False.
        :param workers_per_node: int, the number of worker you want to use.
               The value defaults to 1. The param is only effective when
               distributed is set to True.
        :param distributed_backend: str, select from "torch_distributed" or
               "horovod". The value defaults to "torch_distributed".
        """
        # config setting
        self.data_config = {
            "past_seq_len": past_seq_len,
            "future_seq_len": future_seq_len,
            "input_feature_num": input_feature_num,
            "output_feature_num": output_feature_num
        }
        self.config = {
            "lr": lr,
            "loss": loss,
            "num_channels": num_channels,
            "kernel_size": kernel_size,
            "repo_initialization": repo_initialization,
            "optim": optimizer,
            "dropout": dropout
        }

        # model creator settings
        self.local_model = TCNPytorch
        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator

        # distributed settings
        self.distributed = distributed
        self.distributed_backend = distributed_backend
        self.workers_per_node = workers_per_node

        # other settings
        self.lr = lr
        self.metrics = metrics
        self.seed = seed

        super().__init__()
