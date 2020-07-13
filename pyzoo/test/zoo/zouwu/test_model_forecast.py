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

import pytest
import numpy as np
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
import tensorflow as tf
import pandas as pd

from zoo.zouwu.model.forecast import LSTMForecaster
from zoo.zouwu.model.forecast import MTNetForecaster


class TestZouwuModelForecast(ZooTestCase):

    def setup_method(self, method):
        tf.keras.backend.clear_session()
        # super(TestZouwuModelForecast, self).setup_method(method)
        self.ft = TimeSequenceFeatureTransformer()
        self.create_data()

    def teardown_method(self, method):
        pass

    def create_data(self):
        def gen_train_sample(data, past_seq_len, future_seq_len):
            data = pd.DataFrame(data)
            x, y = self.ft._roll_train(data,
                                       past_seq_len=past_seq_len,
                                       future_seq_len=future_seq_len
                                       )
            return x, y

        def gen_test_sample(data, past_seq_len):
            test_data = pd.DataFrame(data)
            x = self.ft._roll_test(test_data, past_seq_len=past_seq_len)
            return x

        self.long_num = 6
        self.time_step = 2
        look_back = (self.long_num + 1) * self.time_step
        look_forward = 1
        self.x_train, self.y_train = gen_train_sample(data=np.random.randn(
            64, 4), past_seq_len=look_back, future_seq_len=look_forward)
        self.x_val, self.y_val = gen_train_sample(data=np.random.randn(16, 4),
                                                  past_seq_len=look_back,
                                                  future_seq_len=look_forward)
        self.x_test = gen_test_sample(data=np.random.randn(16, 4),
                                      past_seq_len=look_back)

    def test_forecast_lstm(self):
        # TODO hacking to fix a bug
        model = LSTMForecaster(target_dim=1, feature_dim=self.x_train.shape[-1])
        model.fit(self.x_train,
                  self.y_train,
                  validation_data=(self.x_val, self.y_val),
                  batch_size=8,
                  distributed=False)
        model.evaluate(self.x_val, self.y_val)
        model.predict(self.x_test)

    def test_forecast_mtnet(self):
        # TODO hacking to fix a bug
        model = MTNetForecaster(target_dim=1,
                                feature_dim=self.x_train.shape[-1],
                                long_series_num=self.long_num,
                                series_length=self.time_step
                                )
        x_train_long, x_train_short = model.preprocess_input(self.x_train)
        x_val_long, x_val_short = model.preprocess_input(self.x_val)
        x_test_long, x_test_short = model.preprocess_input(self.x_test)

        model.fit([x_train_long, x_train_short],
                  self.y_train,
                  validation_data=([x_val_long, x_val_short], self.y_val),
                  batch_size=32,
                  distributed=False)
        model.evaluate([x_val_long, x_val_short], self.y_val)
        model.predict([x_test_long, x_test_short])

    def test_forecast_tcmf(self):
        from zoo.zouwu.model.forecast import TCMFForecaster
        import tempfile
        model = TCMFForecaster(max_y_iterations=1,
                               init_FX_epoch=1,
                               max_FX_epoch=1,
                               max_TCN_epoch=1,
                               alt_iters=2)
        horizon = np.random.randint(1, 50)
        x = np.random.rand(300, 480)
        with self.assertRaises(Exception) as context:
            model.is_distributed()
        self.assertTrue('You should run fit before call is_distributed()' in str(context.exception))
        model.fit(x, id_as_first_col=False)
        assert not model.is_distributed()
        with self.assertRaises(Exception) as context:
            model.fit(x)
        self.assertTrue('This model has been full trained' in str(context.exception))
        with self.assertRaises(Exception) as context:
            model.fit(x, incremental=True)
        self.assertTrue('NotImplementedError' in context.exception.__class__.__name__)
        with tempfile.TemporaryDirectory() as tempdirname:
            model.save(tempdirname)
            loaded_model = TCMFForecaster.load(tempdirname, distributed=False)
        yhat = model.predict(x=None, horizon=horizon)
        yhat_loaded = loaded_model.predict(x=None, horizon=horizon)
        assert yhat.shape == (300, horizon + 1)
        assert (yhat == yhat_loaded).all()
        target_value = np.random.rand(300, horizon)
        model.evaluate(x=None, target_value=target_value, metric=['mse'])

    def test_forecast_tcmf_xshards(self):
        from zoo.zouwu.model.forecast import TCMFForecaster
        import zoo.orca.data.pandas
        from zoo.orca.data import SharedValue
        import tempfile

        def preprocessing(df):
            idx = df.index.values
            data = np.concatenate((np.expand_dims(idx, axis=1), df.to_numpy().astype(np.float32))
                                  , axis=1)
            return data

        def postprocessing(pred_results, output_dt_col_name):
            final_df = pd.DataFrame(pred_results, columns=["idx"] + output_dt_col_name.value)
            final_df.idx = final_df.idx.astype("int")
            final_df = final_df.set_index("idx")
            final_df.columns.name = "datetime"
            final_df = final_df.unstack().reset_index().rename({0: "kpi"}, axis=1)
            return final_df

        model = TCMFForecaster(max_y_iterations=1,
                               init_FX_epoch=1,
                               max_FX_epoch=1,
                               max_TCN_epoch=1,
                               alt_iters=2)

        with tempfile.NamedTemporaryFile() as temp:
            x = np.random.rand(300, 480)
            df = pd.DataFrame(x)
            df.to_csv(temp.name)
            shard = zoo.orca.data.pandas.read_csv(temp.name)
        shard = shard.transform_shard(preprocessing)
        with self.assertRaises(Exception) as context:
            model.fit(shard, id_as_first_col=False)
        self.assertTrue('id should be the first column of ndarray in xShards' in
                        str(context.exception))
        with self.assertRaises(Exception) as context:
            model.is_distributed()
        self.assertTrue('You should run fit before call is_distributed()' in str(context.exception))
        model.fit(shard)
        assert model.is_distributed()
        with self.assertRaises(Exception) as context:
            model.fit(shard)
        self.assertTrue('This model has been full trained' in str(context.exception))
        with self.assertRaises(Exception) as context:
            model.fit(shard, incremental=True)
        self.assertTrue('NotImplementedError' in context.exception.__class__.__name__)
        with tempfile.TemporaryDirectory() as tempdirname:
            model.save(tempdirname + "/model")
            loaded_model = TCMFForecaster.load(tempdirname + "/model", distributed=True)
        horizon = np.random.randint(1, 50)
        yhat_shard_origin = model.predict(x=None, horizon=horizon)
        yhat_list_origin = yhat_shard_origin.collect()
        yhat_shard = loaded_model.predict(x=None, horizon=horizon)
        yhat_list = yhat_shard.collect()
        yhat_origin = np.concatenate(yhat_list_origin, axis=0)
        yhat = np.concatenate(yhat_list, axis=0)
        assert yhat.shape == (300, horizon + 1)
        assert (yhat == yhat_origin).all()
        output_dt_col_name = pd.date_range(start='2020-05-01', periods=horizon, freq='H').to_list()
        output_dt_col_name_shared_value = SharedValue(output_dt_col_name)
        yhat_df_shards = yhat_shard.transform_shard(postprocessing, output_dt_col_name_shared_value)
        final_df_list = yhat_df_shards.collect()
        final_df = pd.concat(final_df_list)
        final_df.sort_values("datetime", inplace=True)
        assert final_df.shape == (300 * horizon, 3)


if __name__ == "__main__":
    pytest.main([__file__])
