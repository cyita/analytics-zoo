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
import shutil
import errno
import string
from bigdl.nn.criterion import *
from bigdl.nn.layer import *
from bigdl.optim.optimizer import *
from numpy.testing import assert_allclose
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler
from pyspark.sql.types import *

from zoo.common.nncontext import *
from zoo.pipeline.estimator import *
from zoo.pipeline.api.keras.optimizers import Adam as KAdam
from zoo.feature.common import *
from zoo.feature.image import *
from zoo.util.tf import *

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase
from zoo.feature.common import *
from zoo import init_nncontext, init_spark_conf


class TestEstimator(ZooTestCase):

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        # sparkConf = init_spark_conf().setMaster("local[1]").setAppName("testEstimator")
        # self.sc = init_nncontext(sparkConf)
        # self.sqlContext = SQLContext(self.sc)
        # assert(self.sc.appName == "testEstimator")
        sparkConf = init_spark_conf().setMaster("local[4]").setAppName("test feature set")
        self.sc = init_nncontext(sparkConf)


    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_estimator_train_imagefeature(self):
        batch_size = 8
        epoch_num = 5
        images = []
        labels = []
        for i in range(0, 8):
            features = np.random.uniform(0, 1, (200, 200, 3))
            label = np.array([2])
            images.append(features)
            labels.append(label)

        image_frame = DistributedImageFrame(self.sc.parallelize(images),
                                            self.sc.parallelize(labels))

        transformer = Pipeline([BytesToMat(), Resize(256, 256), CenterCrop(224, 224),
                                ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                                MatToTensor(), ImageFrameToSample(target_keys=['label'])])
        data_set = FeatureSet.image_frame(image_frame).transform(transformer)

        model = Sequential()
        model.add(SpatialConvolution(3, 1, 5, 5))
        model.add(View([1 * 220 * 220]))
        model.add(Linear(1 * 220 * 220, 20))
        model.add(LogSoftMax())
        optim_method = SGD(learningrate=0.01)

        estimator = Estimator(model, optim_method, "")
        estimator.set_constant_gradient_clipping(0.1, 1.2)
        estimator.train_imagefeature(train_set=data_set, criterion=ClassNLLCriterion(),
                                        end_trigger=MaxEpoch(epoch_num),
                                        checkpoint_trigger=EveryEpoch(),
                                        validation_set=data_set,
                                        validation_method=[Top1Accuracy()], batch_size=batch_size)
        predict_result = model.predict_image(image_frame.transform(transformer))
        assert(predict_result.get_predict().count(), 8)

    # def test_estimator_train_rdd(self):
    #     dim = 2
    #     data_len = 100
    #
    #     def gen_rand_sample():
    #         features = np.random.uniform(0, 1, dim)
    #         label = np.array((2 * features).sum() + 0.4)
    #         return Sample.from_ndarray(features, label)
    #
    #     data_set = FeatureSet.rdd(self.sc.parallelize(range(0, data_len)).map(lambda i:
    #                                                                           gen_rand_sample()))
    #
    #     model = Sequential().set_name("mse")
    #     model.add(Linear(4, 4).set_name(("fc_1")))
    #     model.add(Sigmoid())
    #     model.add(Linear(4, 1).set_name("fc_2"))
    #     model.add(Sigmoid())
    #
    #     optim_method = SGD(learningrate=0.01)
    #
    #     estimator = Estimator()


if __name__ == "__main__":
    pytest.main([__file__])
