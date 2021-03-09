package com.intel.analytics.zoo.examples.dlrm

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions.broadcast
import com.intel.analytics.zoo.common.NNContext

import scala.collection.mutable.ArrayBuffer
import scala.math.log

object JoinPreprocess {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

//    val dataPath = "/var/backups/dlrm/terabyte/"
    //    val dataPath = "/home/kai/Downloads/dac_sample/"
    //    val dataPath = "hdfs://172.16.0.165:9000/dlrm/"
//    val modelPath = dataPath + "models/"
//    val parquetPath = dataPath + "parquet/"
    val dataPath = "hdfs://172.168.0.108:9000/"
    val modelPath = dataPath + "dlrm/na/models/"
    val parquetPath = dataPath + "parquet/"
    val sc = NNContext.initNNContext("DLRM Preprocess")
    println("Spark default parallelism: " + sc.defaultParallelism) // total cores
    val sqlContext = SQLContext.getOrCreate(sc)

    val files = new ArrayBuffer[String]()
    for( i <- 0 to 22) {
      files.append(parquetPath + "day_" + i + ".parquet")
    }
    val start = System.nanoTime
    var df = sqlContext.read.parquet(files.toList:_*)
    for( i <- 14 to 39) {
      val colName = "_c" + i
      val model = sqlContext.read.parquet(modelPath + i + ".parquet").drop("model_count")
        .withColumnRenamed("data", colName)
      // missing check would_broadcast
      val broadcastModel = broadcast(model)
      df = df.join(broadcastModel, df(colName) === broadcastModel(colName), joinType="left")
        .drop(colName)
        .withColumnRenamed("id", colName)
    }

    val preprocessed = df.rdd.map(row => {
      val intFeatures = new ArrayBuffer[Float]()
      for( i <- 1 to 13) {
        if (row.isNullAt(i)) {
          intFeatures.append(0)
        }
        else {
          val intFeature = row.getInt(i)
          if (intFeature < 0) intFeatures.append(0) else intFeatures.append(log(intFeature + 1).toFloat)
        }
      }
      val catFeatures = new ArrayBuffer[Int]()
      for( i <- 14 to 39) {
        if (row.isNullAt(i)) {
          catFeatures.append(0)
        }
        else {
          catFeatures.append(row.getInt(i))
        }
      }
      // (y, X_int, X_cat)
      (row.getInt(0), intFeatures.toList, catFeatures.toList)
    })
    //    val resRDD = preprocessed.repartition(sc.defaultParallelism)
    val count = preprocessed.count()
    val end = System.nanoTime
    println("Train data loading and preprocessing time: " + (end - start) / 1e9d)
    println("Preprocessed train count: " + count)
    println("Preprocessed train partitions: " + preprocessed.getNumPartitions)
    preprocessed.take(5).foreach(record => {
      println(record._1 + " " + record._2 + " " + record._3)
    })
    sc.stop()
  }
}