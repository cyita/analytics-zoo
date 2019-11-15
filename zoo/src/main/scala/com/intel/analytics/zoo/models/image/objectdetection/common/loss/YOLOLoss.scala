package com.intel.analytics.zoo.models.image.objectdetection.common.loss

import java.util

import scala.math._
import com.intel.analytics.bigdl.nn.{BCECriterion, CMaxTable, CMinTable, CMulTable, CrossEntropyCriterion, MSECriterion, Sigmoid}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{DenseTensorApply, TensorFunc4}
import com.intel.analytics.zoo.pipeline.api.keras.layers.Max

import scala.collection.mutable.ArrayBuffer

//import com.intel.analytics.bigdl.nn.{BCECriterion, CrossEntropyCriterion, Graph, Input, MSECriterion}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractCriterion, AbstractModule}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import com.intel.analytics.zoo.models.common.KerasZooModel
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Input, MulConstant, Narrow, Permute, Select}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Minimum.minimum

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
                         strides: Array[Int] = Array(32, 16, 8),
                         classNum: Int = 80
                         )

class YOLOLoss[T: ClassTag](param: YOLOLossParam)
                           (implicit ev: TensorNumeric[T]) extends AbstractCriterion[Table,
  Tensor[T], T] {
  val xyMse = MSECriterion[T]()
  val whMes = MSECriterion()
  val ccCe = CrossEntropyCriterion()
  val bce = BCECriterion()

  override def updateOutput(input: Table, target: Tensor[T]): T = {
//    val lxy = Tensor[T](0)
//    val lwh = Tensor[T](0)
//    val lcls = Tensor[T](0)
//    val lconf = Tensor[T](0)

    val yoloOutput1 = input[Tensor[T]](1)
    val yoloOutput2 = input[Tensor[T]](2)
    val yoloOutput3 = input[Tensor[T]](3)


    val inputArr = Array(yoloOutput1, yoloOutput2, yoloOutput3)

    var lxy = 0f
    var lwh = 0f
    var lcls = 0f
    var lconf = 0f
    val targetResults = buildTargets(target)
    val txy = targetResults._1
    val twh = targetResults._2
    val tcls = targetResults._3
    val indices = targetResults._4
    val batchSize = yoloOutput1.size(1)
    val lossGain = batchSize * param.lossMultiple

//    val in = Input()
//    val i1 = Narrow(2, 1, 2).inputs(in)
//    val i2 = Narrow(2, 3, 2).inputs(in)
//    val i3 = Narrow(2, 5, 1).inputs(in)
//    val i4 = Narrow(2, 6, 10).inputs(in)
//    val m = Graph(Array(in), Array(i1, i2, i3, i4))

//    val mseXy = MSECriterion()
//    val MSE = MSECriterion[T]()
//    val BCE = BCECriterion[T]()
//    val CE = CrossEntropyCriterion[T]()

    for (i <- 1 to input.length) {
      val pi0 = input[Tensor[T]](i)
      val idx = indices(i - 1)
      val b = Tensor(Storage(idx(0)))
      val a = Tensor(Storage(idx(1)))
      val gj = idx(2)
      val gi = idx(3)

      val tconf = Tensor(Array(pi0.size(1), pi0.size(2), pi0.size(3), pi0.size(4)))

      if (b.size(1) > 0) {
        val pi = Tensor[T]()
        pi.index(1, b, pi0)
        pi.index(2, a, pi)
        val tmp = "aa"

        val sig = Sigmoid[T]()

        ev.fromType(lossGain * param.xyLossFraction) * xyMse.forward(sig.forward(pi.narrow(2, 1, 2)), txy(i - 1))
      }

//      val o = m.forward(pi0).toTable
//      lxy += xyMse.forward()


    }

    val loss = lxy + lwh + lconf + lcls
    ev.fromType(loss)
  }

  override def updateGradInput(input: Table, target: Tensor[T]): Table = ???

  def buildTargets(targets: Tensor[T]) = {
    val iouThres = param.iouTargetAnchorTrainingThreshold
    val labelNum = targets.dim()
    val txy = ArrayBuffer[Tensor[T]]()
    val twh = ArrayBuffer[Tensor[T]]()
    val tcls = ArrayBuffer[Tensor[T]]()
    val indices = ArrayBuffer[Array[Array[T]]]()

    for (i <- 0 to 2) {
      val stride = param.strides(i)
      val ngInt: Int = param.imgSize / stride
      val ng = ev.fromType(ngInt)
      var gwh: Tensor[T] = targets.narrow(2, 5, 2) * ng
      var t = targets
      var a = Array[T]()
      val mask = param.masks(i)
      val anchors = for (i <- 0 until mask.length) yield param.anchors(mask(i))
      val anchorVector = for (elem <- anchors) yield Array(elem(0) / stride, elem(1) / stride)
      if (labelNum > 0) {
        val iou = for (elem <- anchorVector) yield whIou(elem, gwh)
        val res = bestIou(iou.toVector)
        val iouMax = res._1
        a = res._2

        // reject below threshold ious (OPTIONAL, increases P, lowers R)
        val reject = true
        if (reject){
          val iouTensor: Tensor[T] = Tensor(Storage(Array(
            ev.fromType(iouThres))))
            .expand(Array(iouMax.size(1)))

          iouMax.map(iouTensor, (x: T, y: T) =>
            if (ev.toType[Double](x) > ev.toType[Double](y)) ev.fromType(1) else ev.fromType(0))
          t = maskTensor(iouMax, targets)
          gwh = maskTensor(iouMax, gwh)
          a = maskArray(iouMax, a)
        }
      }

      val (b, c, gxy, gi, gj) = if (t.size(1) > 0) {
        val bc = t.narrow(2, 1, 2).t()
        val gxyTmp = t.narrow(2, 3, 2) * ng
        (bc.narrow(1, 1, 1).contiguous().storage().array().map(x => ev.fromType(ev.toType[Long](x) + 1)),
          bc.narrow(1, 2, 1).contiguous().apply1(x => ev.fromType(ev.toType[Long](x))),
        gxyTmp, gxyTmp.select(2, 1).contiguous().storage().array().map(x => ev.fromType(ev.toType[Long](x) + 1)),
          gxyTmp.select(2, 2).contiguous().storage().array().map(x => ev.fromType(ev.toType[Long](x) + 1)))
      } else {
        (Array[T](), Tensor[T](Array(0, 2)), Tensor[T](Array(0, 2)), Array[T](), Array[T]())
      }
//      val gxy = t.narrow(2, 3, 2) * ng
//      val gi = gxy.select(2, 1).contiguous().storage().array().map(x => ev.toType[Long](x))
//      val gj = gxy.select(2, 2).contiguous().storage().array().map(x => ev.toType[Long](x))
      indices += Array(b, a.map(x => ev.fromType(ev.toType[Long](x) + 1)), gj, gi)
      txy += gxy.apply1(x => ev.fromType(ev.toType[Double](x) - floor(ev.toType[Double](x))))
      val anchorArray = ArrayBuffer[T]()
      for (elem <- anchorVector) {
        anchorArray ++= elem.map(x => ev.fromType(x))
      }
      val anchorTensor = Tensor(Storage(anchorArray.toArray)).reshape(Array(anchorVector.length, 2))
      val indexedAnchorTensor = Tensor[T]()
      indexedAnchorTensor.index(1, Tensor(Storage(a)) + ev.fromType(1), anchorTensor)
      twh += (gwh / indexedAnchorTensor).apply1(x => ev.fromType(log(ev.toType[Double](x))))
      tcls += c

      if (c.size(2) > 0) {
        require(ev.toType[Int](c.max()) <= param.classNum, s"Target classes exceed model classes")
      }
    }

    (txy, twh, tcls, indices)
  }

  def maskTensor(mask: Tensor[T], origin: Tensor[T]): Tensor[T] = {
    val length = mask.sum()
    val res = Tensor[T]()
    res.resize(ev.toType[Double](length).toInt, origin.size(2))
    val idx = mask.storage().array()
    var currPosition = 1
    for (i <- 0 until idx.length) {
      if (ev.toType[Int](idx(i)) == 1) {
        res.select(1, currPosition).copy(origin.select(1, i + 1))
        currPosition += 1
      }
    }

    res
  }

  def maskArray(mask: Tensor[T], origin: Array[T]): Array[T] = {
    val length = mask.sum()
    val res = new Array[T](ev.toType[Int](length))
    val idx = mask.storage().array()
    var currPosition = 0
    for (i <- 0 until idx.length) {
      if (ev.toType[Int](idx(i)) == 1) {
        res(currPosition) = origin(i)
        currPosition += 1
      }
    }
    res
  }

  def bestIou(ious: Vector[Tensor[T]]) = {
    val result = Tensor[T]()
    val rowLen = ious.length
    if (rowLen == 0) {
      val res = (result, Array[T]())
      res
    } else {
      val res1 = ious(0)
      result.resizeAs(res1).copy(res1)
      val maxIdx = Tensor[T]()
      val mask = Tensor[T]()
      val maskResult = Tensor[T]()
      maxIdx.resizeAs(res1).fill(ev.fromType(1))

      var i = 2
      while (i <= ious.length) {
        mask.resize(res1.size())
        mask.gt(ious(i - 1), result)
        maxIdx.maskedFill(mask, ev.fromType(i))

        if (ev.isGreater(mask.sum(), ev.fromType(0))) {
          result.maskedCopy(mask, ious(i - 1).maskedSelect(mask, maskResult))
        }
        i += 1
      }

      val idxArr: Array[T] = (maxIdx - ev.fromType(1)).storage().array()

      (result, idxArr)
    }
  }

  def whIou(box1: Array[Float], box2: Tensor[T]): Tensor[T] = {
    val box2T = box2.t()
    val w1 = Tensor(Storage(Array(box1(0))))
    val h1 = Tensor(Storage(Array(box1(1))))
    val w2 = box2T.select(1, 1).contiguous()
    val h2 = box2T.select(1, 2).contiguous()

    val w1Expand = w1.expand(Array(w2.size(1)))
    val h1Expand = h1.expand(Array(h2.size(1)))

    val wMinLayer = CMinTable()
    val wMin = wMinLayer.forward(T(w1Expand, w2))
    val hMinLayer = CMinTable()
    val hMin = hMinLayer.forward(T(h1Expand, h2))

    // Calculate intersection area
    val interMulLayer = CMulTable()
    val interArea: Tensor[T] = interMulLayer.forward(T(wMin, hMin))
    val mulLayer1 = CMulTable()
    val whMul1: Tensor[T] = mulLayer1.forward(T(w1Expand, h1Expand))
    val mulLayer2 = CMulTable()
    val whMul2: Tensor[T] = mulLayer2.forward(T(w2, h2))
    val unionArea = (whMul1 + ev.fromType(1e-16)) + whMul2 - interArea

    val result: Tensor[T] = interArea / unionArea
    result
  }
}

//class YoloCriterion[T: ClassTag](param: YOLOLossParam)(implicit ev: TensorNumeric[T]) extends KerasZooModel[Tensor[T], Tensor[T], T]{
//
//  /**
//   * Override this method to define a model.
//   */
//
//
//
//  override protected def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {
//
//
////    val input: ModuleNode[T] = Input(inputShape = Shape(10647, 85))
////    val targetInput: ModuleNode[T] = Input(inputShape = Shape(6))
////
////    val model = Sequential[T]()
////    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
//  }
//
//  private def buildTargets(targets: ModuleNode[T]): ModuleNode[T] = {
//    val labelNum = 1
//    var txy = new util.ArrayList[T]()
//    var twh = new util.ArrayList[T]()
//    var tcls = new util.ArrayList[T]()
//    var indices = new util.ArrayList[T]()
//    val nodes = List()
//
//    for (i <- 0 to 2) {
//      val stride = param.strides(i)
//      val ngInt: Int = param.imgSize / stride
////      val gwh = targets.narrow(2, 4, 2) * ng
//      val wh = Narrow[T](1, 3, 2).inputs(targets)
//      val gwh = MulConstant[T](ngInt.toDouble).inputs(wh)
//      if (labelNum > 0) {
//        val mask = param.masks(i)
//        val anchors = for (i <- 0 until mask.length) yield param.anchors(mask(i))
//        val anchorVector = for (elem <- anchors) yield Array(elem(0) / stride, elem(1) / stride)
//        val iou = for (elem <- anchorVector) yield whIou(elem, gwh)
//      }
//    }
//    targets // TODO: change return value
//  }
//
//  def whIou(box1: Array[Float], box2: ModuleNode[T]): Float = {
//    val box2T = Permute(Array(2, 1)).inputs(box2)
//    val box1Tensor = Tensor(Storage(box1))
//    val w1 = box1(0)
//    val h1 = box1(1)
////    val w2 = box2T.select(1, 1)
////    val h2 = box2T.select(1, 2)
//    val w2 = Select[T](0, 0).inputs(box2T)
//    val h2 = Select[T](0, 1).inputs(box2T)
//
//
//    // Calculate intersection area
//    val interArea = 1
//    val result: Float = 0.5f
//    result
//  }
//}
