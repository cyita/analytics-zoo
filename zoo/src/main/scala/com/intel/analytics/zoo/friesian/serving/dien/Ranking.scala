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
package com.intel.analytics.zoo.friesian.serving.dien

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.NNContext
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.{col, max, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OptionParser

import scala.collection.mutable

object Ranking {
  case class RankingParams(modelPath: String = "/home/yina/Documents/model/dien",
                           dataDir: String = "/home/yina/Documents/data/dien")
  val logger: Logger = Logger.getLogger(getClass)
  logger.setLevel(Level.WARN)
  var argv: RankingParams = _
  var userFeatures: DataFrame = _
  var itemFeatures: DataFrame = _
  val userFeatureColumns = List("user", "item_history", "category_history", "item_history_mask",
    "length")
  val itemFeatureColumns = List("item", "category")
  val parser: OptionParser[RankingParams] = new OptionParser[RankingParams]("Text Classification Example") {
    opt[String]('m', "modelPath")
      .text("model path of DIEN")
      .action((x, params) => params.copy(modelPath = x))
    opt[String]("dataDir")
      .text("DIEN preprocessed data folder")
      .action((x, params) => params.copy(dataDir = x))
  }

  def loadUserItemFeatures(spark: SparkSession): Unit = {
    var df = spark.read.parquet(argv.dataDir + "/all_data.parquet")
    val getLabelUDF = udf((label: mutable.WrappedArray[Float]) => {
      label(0)
    })
    df = df.withColumn("label_value", getLabelUDF(col("label")))
    itemFeatures = df.select("item", "category").distinct()
    df = df.filter("label_value > 0")
    userFeatures =df.select("user", "item_history", "category_history",
      "item_history_mask", "length")
    val user_length_df = userFeatures.groupBy("user").agg(max("length").alias("max_length"))
    userFeatures = userFeatures.join(user_length_df, "user")
    userFeatures = userFeatures.filter("length == max_length").drop("max_length")
    itemFeatures.cache()
    userFeatures.cache()
  }

  def getFeatures(df: DataFrame, key: String, ids: List[Int], featureColumns: List[String])
  : Array[Seq[Tensor[Int]]] = {
    val conditions = ids.map(id => s"$key = $id").mkString(" or ")
    val features = df.filter(conditions).select(featureColumns.map(col):_*)
    val featuresList = features.rdd.map(row => {
      val rowd = row.toSeq
      val tensors = row.toSeq.map {
        case d: Int  => Tensor[Int](1).fill(d).addSingletonDimension()
        case d: Long => Tensor[Int](1).fill(d.toInt).addSingletonDimension()
        case d: mutable.WrappedArray[Int] =>
          Tensor[Int](d.toArray, Array(1, d.size))
        case data => throw new IllegalArgumentException(
          s"Unsupported value type ${data.getClass.getName}.")
      }
      tensors
    }).collect()
    featuresList
  }

  def main(args: Array[String]): Unit = {
    argv = parser.parse(args, RankingParams()).head
    val sc = NNContext.initNNContext("Ranking service")
    sc.setLogLevel("WARN")
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()
    loadUserItemFeatures(spark)
    itemFeatures.show()
    userFeatures.show()
    val userIdList = List(6674, 9243)
    val itemIdList = List(1060, 1684)
    val userF = getFeatures(userFeatures, "user", userIdList, userFeatureColumns)
    val itemF = getFeatures(itemFeatures, "item", itemIdList, itemFeatureColumns)
  }
}
