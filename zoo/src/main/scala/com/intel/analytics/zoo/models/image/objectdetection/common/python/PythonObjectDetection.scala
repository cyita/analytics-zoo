package com.intel.analytics.zoo.models.image.objectdetection.common.python

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.models.image.objectdetection.common.loss.{MultiBoxLoss, MultiBoxLossParam}

import scala.reflect.ClassTag


object PythonObjectDetection {
  def ofFloat(): PythonObjectDetection[Float] = new PythonObjectDetection[Float]()

  def ofDouble(): PythonObjectDetection[Double] = new PythonObjectDetection[Double]()
}

class PythonObjectDetection [T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T]{
  def createMultiBoxCriterion(locWeight: Double = 1.0, nClasses: Int = 21,
                              shareLocation: Boolean = true,
                              overlapThreshold: Double = 0.5,
                              bgLabelInd: Int = 0,
                              useDifficultGt: Boolean = true,
                              negPosRatio: Double = 3.0,
                              negOverlap: Double = 0.5): MultiBoxLoss[T] = {
    val param = MultiBoxLossParam(locWeight, nClasses, shareLocation, overlapThreshold, bgLabelInd,
      useDifficultGt, negPosRatio, negOverlap)
    new MultiBoxLoss[T](param)
  }
}
