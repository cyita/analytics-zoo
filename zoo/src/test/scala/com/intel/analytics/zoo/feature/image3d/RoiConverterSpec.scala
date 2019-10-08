package com.intel.analytics.zoo.feature.image3d

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.python
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, LocalImageFrame}
import java.util.{ArrayList => JArrayList, HashMap => JHashMap, List => JList, Map => JMap}

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.FeatureSet
import com.intel.analytics.zoo.feature.image.ImageMatToFloats
import com.intel.analytics.zoo.feature.image.roi.RoiFeatureLabelConverter
import com.intel.analytics.zoo.models.image.objectdetection.common.loss.{YOLOLoss, YOLOLossParam}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext

class RoiConverterSpec  extends ZooSpecHelper with Matchers {
  var sqlContext: SQLContext = _
  var sc: SparkContext = _

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("NCFTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
  }

  "YOLOLoss" should "work properly" in {
    val output: Tensor[Float] = Tensor(Array(1, 10647, 85))
    val target: Tensor[Float] = Tensor(T(T(0.0000f, 60.0000f,  0.4988f,  0.4969f, 0.9950,  0.5629),
      T(0.0000, 41.0000,  0.5121,  0.4345,  0.3277,  0.3328),
      T(0.0000, 43.0000,  0.5036,  0.6295,  0.9203,  0.1533)))
    val params = new YOLOLossParam()
    val lossFunc = new YOLOLoss[Float](params)
    lossFunc.forward(output, target)
  }

  "RoiFeatureLabelConverter" should "work properly" in{
    val feature = new ImageFeature()
    var images = new JArrayList[Array[Float]]()
    var labels = new JArrayList[Tensor[Float]]()

    val tensor: Tensor[Float] = Tensor(T(
      T(0f, 1f, 2f, 3f, 4f, 5f),
      T(0f, 4f, 5f, 6f, 7f, 8f)))
    val s = tensor.dim()

    for (i <- 1 to 3) {
      images.add(Array.fill(519168){scala.util.Random.nextFloat()})
      labels.add(Tensor(T(
        T(0f, 1f, 2f, 3f, 4f, 5f),
        T(0f, 4f, 5f, 6f, 7f, 8f))))
    }

    val features = (0 until images.size()).map(i => {
      val img = images.get(i)
      val label = labels.get(i)
      val mat = OpenCVMat.fromFloats(img, 416, 416, 3)
      val feature = new ImageFeature()
      feature(ImageFeature.bytes) = OpenCVMat.imencode(mat)
      feature(ImageFeature.mat) = mat
      feature(ImageFeature.originalSize) = mat.shape()
      feature(ImageFeature.label) = label
      feature
    })

    val imageFrame = new LocalImageFrame(features.toArray)
    // val transformed = imageFrame.transform(new RoiFeatureLabelConverter(5))
//    val transformer = new RoiFeatureLabelConverter(5) -> ImageMatToFloats(validHeight = 416,
//      validWidth = 416) -> RoiImageToSSDBatch(2)

    val transformed = imageFrame.transform(new RoiFeatureLabelConverter(5) -> ImageMatToFloats
    (416, 416))
    //val minibatch = imageFrame -> transformer
//    val imageFrame = new DistributedImageFrame()

    val feature_set = FeatureSet.rdd(imageFrame.toDistributed(sc).rdd)
    //val train_set = feature_set -> transformer
    print("a")
  }
}
