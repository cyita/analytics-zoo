package com.intel.analytics.zoo.models.image.objectdetection.common.loss

import java.util

import com.intel.analytics.bigdl.nn.{BCECriterion, CrossEntropyCriterion, MSECriterion}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

case class YOLOLossParam(lossMultiple: Double = 10.39,
                         xyLossFraction: Double = 0.1367,
                         whLossFraction: Double = 0.01057,
                         clsLossFraction: Double = 0.01181,
                         confLossFraction: Double = 0.8409,
                         iouTargetAnchorTrainingThreshold: Double = 0.1287,
                         anchors: Array[Array[Float]] = Array(Array(10, 13), Array(16, 30), Array
                         (33, 23), Array(30, 61), Array(62, 45), Array(59, 119), Array(116, 90),
                           Array(156, 198), Array(373, 326)),
                         masks: Array[Array[Int]] = Array(Array(6, 7, 8), Array(3, 4, 5), Array
                         (0, 1, 2)),
                         imgSize: Int = 416,
                         strides: Array[Int] = Array(32, 16, 8)
                         )

class YOLOLoss[T: ClassTag](param: YOLOLossParam)
                           (implicit ev: TensorNumeric[T]) extends AbstractCriterion[Tensor[T],
  Tensor[T], T] {

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
//    val lxy = Tensor[T](0)
//    val lwh = Tensor[T](0)
//    val lcls = Tensor[T](0)
//    val lconf = Tensor[T](0)

    val lxy = 0
    val lwh = 0
    val lcls = 0
    val lconf = 0
    buildTargets(target)


    val MSE = MSECriterion[T]()
    val BCE = BCECriterion[T]()
    val CE = CrossEntropyCriterion[T]()
    val batchSize = input.dim()
    val lossGain = param.lossMultiple * batchSize

    for (i <- 1 to batchSize) {
      val pi0 = input.select(1, i)

    }

    val loss = lxy + lwh + lconf + lcls
    ev.fromType(loss)
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = ???

  def buildTargets(targets: Tensor[T]): Unit = {
    val labelNum = targets.dim()
    var txy = new util.ArrayList[T]()
    var twh = new util.ArrayList[T]()
    var tcls = new util.ArrayList[T]()
    var indices = new util.ArrayList[T]()

    for (i <- 0 to 2) {
      val stride = param.strides(i)
      val ngInt: Int = param.imgSize / stride
      val ng = ev.fromType(ngInt)
      val gwh = targets.narrow(2, 4, 2) * ng
      if (labelNum > 0) {
        val mask = param.masks(i)
        val anchors = for (i <- 0 until mask.length) yield param.anchors(mask(i))
        val anchorVector = for (elem <- anchors) yield Array(elem(0) / stride, elem(1) / stride)
        val iou = for (elem <- anchorVector) yield whIou(elem, gwh)
      }
    }
  }

  def whIou(box1: Array[Float], box2: Tensor[T]): Float = {
    val box2T = box2.t()
    val box1Tensor = Tensor(Storage(box1))
    val w1 = box1(0)
    val h1 = box1(1)
    val w2 = box2T.select(1, 1)
    val h2 = box2T.select(1, 2)

    // Calculate intersection area
    val interArea = 1
    val result: Float = 0.5f
    result
  }
}
