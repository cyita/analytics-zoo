/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.keras.nn

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.dllib.keras.{Sequential => KSequential}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.dllib.keras.SReLU
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest

import scala.util.Random

class SReLUSpec extends KerasBaseSpec{

  "SReLU" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2, 3])
        |input = np.random.uniform(-1, 1, [1, 2, 3])
        |output_tensor = SReLU('one', 'one')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val srelu = SReLU[Float](sharedAxes = null, inputShape = Shape(2, 3))
    seq.add(srelu)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "SReLU with shared axes" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24])
        |input = np.random.random([2, 3, 24])
        |output_tensor = SReLU(shared_axes=[1, 2])(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val srelu = SReLU[Float](sharedAxes = Array(1, 2), inputShape = Shape(3, 24))
    seq.add(srelu)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class SReLUSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = SReLU[Float](sharedAxes = Array(1, 2), inputShape = Shape(4, 32))
    layer.build(Shape(2, 4, 32))
    val input = Tensor[Float](2, 4, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
