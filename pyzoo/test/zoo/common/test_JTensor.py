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

from zoo.feature.common import JTensor
from zoo import init_nncontext, init_spark_conf
import numpy as np
import pickle
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer


# class TestFeatureCommon(ZooTestCase):
#
#     def setup_method(self, method):
#         """ setup any state tied to the execution of the given method in a
#         class.  setup_method is invoked for every test method of a class.
#         """
#         sparkConf = init_spark_conf().setMaster("local[4]").setAppName("test JTensor")
#         self.sc = init_nncontext(sparkConf)
#
#     def test_JTensor(self):
#         np.random.rand(3, 2)


if __name__ == "__main__":
    # sparkConf = init_spark_conf().setMaster("local[4]").setAppName("test JTensor")
    # sc = init_nncontext(sparkConf)
    # sample = np.random.rand(3, 2)
    # print(sample)
    # print("-----------------------")
    # jtensor = JTensor.from_ndarray(sample)
    # print(jtensor)
    # print("-----------------------")
    # print(jtensor.to_ndarray())
    # print("-----------------------")
    # with open('test.npy', 'wb') as f:
    #     #pickle.dump(jtensor.to_ndarray(), f)
    #     np.save(f, jtensor.to_ndarray())

    with open('testpickle.dat', 'rb') as f:
        x = PickleSerializer().loads(bytes(f.read()), encoding="bytes")
        print(x.to_ndarray())
        # x = np.load(f)
        print(x)
