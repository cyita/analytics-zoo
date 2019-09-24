/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.models.python

import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.objectdetection.common.dataset.{ByteRecord, Imdb}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

class PythonImdb {
  def loadRoiSeqFiles(seqFloder: String, sc: SparkContext, nPartition: Option[Int] = None):
  RDD[ByteRecord] = {
    Imdb.loadRoiSeqFiles(seqFloder, sc, nPartition)
  }

  def roiSeqFilesToImageSet(url: String, sc: SparkContext, partitionNum: Option[Int] = None):
  ImageSet = {
    Imdb.roiSeqFilesToImageSet(url, sc, partitionNum)
  }
}
