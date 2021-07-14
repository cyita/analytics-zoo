package com.intel.analytics.zoo.friesian.serving.wnd

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.friesian.serving.dien.Ranking.{RankingParams, argv, parser}
import com.intel.analytics.zoo.pipeline.inference.InferenceModel
import com.intel.analytics.zoo.serving.serialization.StreamSerializer
import org.apache.spark.sql.SparkSession


object WND {
  def main(args: Array[String]): Unit = {
    argv = parser.parse(args, RankingParams()).head
    val sc = NNContext.initNNContext("Ranking service")
    sc.setLogLevel("WARN")
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()
    val dataDir = "/home/yina/Documents/data/wnd/test_parquet"
    val df = spark.read.parquet(dataDir).limit(50).drop("c0")

    val data = df.rdd.map(row => {
      val r = row.toSeq
      r.slice(39, 41) ++ r.slice(13, 39) ++ r.slice(0, 13)
    }).collect()
    val model = new InferenceModel(1)
    model.doLoadTensorflow("/home/yina/Documents/model/wnd/frozen", "frozenModel", 1, 1, true)

    val itemNumberArr = Array(2, 5, 10, 15, 20, 25, 30, 40, 50)
    itemNumberArr.foreach(itemNumber => {
      val cutData = data.slice(0, itemNumber)
      val dataTensorList = (0 until 41).map(idx => {
        Tensor[Float](T.seq(cutData.map(d => {
          d(idx)
        })))
      }).toArray
      val input = T.array(dataTensorList)
      val bytes = StreamSerializer.objToBytes(input)
      val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
      println(b64)
      for (i <- 0 until 100) {
        val result = model.doPredict(input)
        result
      }
      for (t <- 0 until 3) {
        val begin = System.nanoTime()
        val num = 1000
        for (i <- 0 until num) {
          model.doPredict(input)
          //        val resultArr = (0 until 3).toParArray.map(i => {
          //          val d = inputArr(i)
          //          val r = model.doPredict(d)
          //          r
          //        })
          //        resultArr
        }
        val end = System.nanoTime()
        val time = (end - begin) / num
        println(s"itemNumber: ${itemNumber}, t: ${t}, time: $time")
      }
    })
  }
}
