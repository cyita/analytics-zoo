package com.intel.analytics.zoo.feature.image.roi

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature}
import com.intel.analytics.zoo.feature.image.{ImageProcessing}

import scala.reflect.ClassTag

class RoiFeatureLabelConverter(dim: Int=6) extends ImageProcessing {

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform(_))
  }

  override def transform(feature: ImageFeature): ImageFeature = {
    val target = feature.getLabel[Tensor[Float]]
    val gtClasses: Tensor[Float] = target.narrow(2, 1, dim - 4).transpose(1, 2)
    val gtBoxes: Tensor[Float] = target.narrow(2, dim - 3, 4)
    val roiLabel = new RoiLabel(gtClasses, gtBoxes)
    feature(ImageFeature.label) = roiLabel
    feature
  }
}

object RoiFeatureLabelConverter {

  def apply[T: ClassTag](): RoiFeatureLabelConverter =
    new RoiFeatureLabelConverter

}

