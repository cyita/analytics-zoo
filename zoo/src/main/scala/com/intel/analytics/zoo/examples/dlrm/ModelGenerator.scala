package com.intel.analytics.zoo.examples.dlrm

import com.intel.analytics.zoo.common.NNContext

import org.apache.hadoop.fs._
import org.apache.spark.TaskContext
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{row_number, _}
import scopt.OptionParser

case class Params(dayStart: Int = 0,
                  dayEnd: Int = 0,
                  inputFolder: String = "",
                  frequencyLimit: Option[String] = None,
                  outputFolder: Option[String] = None,
                  modelSizeFile: Option[String] = None)

object ModelGenerator {
  val CAT_COLS: List[Int] = (14 until 40).toList
  //  val CAT_COLS: List[Int] = (14 until 15).toList

  def read_parquet(spark: SQLContext, params: Params): Dataset[Row] ={
    val dayRange = (params.dayStart to params.dayEnd).toList
    val paths= dayRange.map(x => s"${params.inputFolder}/day_${x}.parquet")
    spark.read.parquet(paths:_*)
  }

  def get_column_counts_with_frequency_limit(df: Dataset[Row], params: Params): List[DataFrame] = {
    val cols = CAT_COLS.map(x => s"_c${x}")
    val freq_list = params.frequencyLimit.getOrElse("").split(",")
    var default_limit: Option[Int] = None
    val freq_map = scala.collection.mutable.Map[String, Int]()
    for(fl <- freq_list){
      val frequency_pair = fl.split(":")
      if(frequency_pair.length == 1){
        default_limit = Some(frequency_pair(0).toInt)
      } else if (frequency_pair.length == 2){
        freq_map += (frequency_pair(0) -> frequency_pair(1).toInt)
      }
    }
    val df_count_filtered_list: List[DataFrame] = cols.map(col_n => {
      val df_col = df
        .select(col_n)
        .filter(s"${col_n} is not null")
        .groupBy(col_n)
        .count()
      val df_col_filtered = if(freq_map.contains(col_n)) {
        df_col.filter(s"count >= ${freq_map(col_n)}")
      } else if (default_limit.isDefined) {
        df_col.filter(s"count >= ${default_limit.get}")
      } else {
        df_col
      }

      df_col_filtered
    })

    df_count_filtered_list
  }

  def get_count(rows: Iterator[Row]): Iterator[(Int, Int)] ={
    if(rows.isEmpty){
      Array().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }

  def add_id_and_write_data(df_count_list: List[DataFrame], params: Params): Unit ={
    val tmp = CAT_COLS.map(x => {
      val col_n = s"_c${x}"
      val df = df_count_list(x - 14).withColumn("part_id", spark_partition_id())
      df.cache()
      val count_list: Array[(Int, Int)] = df.rdd.mapPartitions(get_count).collect()
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df.rdd.sparkContext.broadcast(base_dict)
      val windowSpec  = Window.partitionBy("part_id").orderBy("count")
      val df_row_number = df.withColumn("row_number", row_number.over(windowSpec))
      val get_label = udf((part_id: Int, row_number: Int)=> {
        row_number + base_dict_bc.value.getOrElse(part_id, 0)
      })
      val df_label = df_row_number.withColumn("label",
        get_label(col("part_id"), col("row_number")))
        .drop("part_id")
        .withColumnRenamed("count", "model_count")
        .withColumnRenamed(col_n, "data")
      save_column_model(df_label, x, params.outputFolder.getOrElse("/tmp/gen_models/"))
    })
  }

  def save_column_model(model: DataFrame, cat_no: Int, model_folder: String): Unit ={
    val path = s"${model_folder}/${cat_no}.parquet"
    model.write.mode("overwrite").parquet(path)
  }

  def load_column_models(spark: SQLContext, model_folder: String, count_required: Boolean)
  : List[(Int, DataFrame, Row, Boolean)] ={
    CAT_COLS.map(x => {
      val path = s"${model_folder}/${x}.parquet"
      val df = spark.read.parquet(path)
      val discnt = df.select("label").distinct().count()
      val maxmin = df.agg(min("label"), max("label")).head()
      val values = if (count_required) {
        df.agg(count("model_count").alias("cnt")).collect()(0)
      } else{
        null
      }
      (x, df, values, would_broadcast(spark, path))
    })
  }

  def would_broadcast(spark: SQLContext, str_path: String): Boolean = {
    val sc = spark.sparkContext
    val config = sc.hadoopConfiguration
    val path = new Path(str_path)
    val fs = FileSystem.get(config)
    val stat = fs.listFiles(path, true)
    var sum: Long = 0
    while (stat.hasNext) {
      sum += stat.next().getLen
    }
    val sql_conf = internal.SQLConf.get
    val cutoff =  sql_conf.autoBroadcastJoinThreshold * sql_conf.fileCompressionFactor
    sum <= cutoff
  }

  def main(args: Array[String]): Unit = {
    val time1 = System.nanoTime()
    val defaultParams = Params()
    val parser = new OptionParser[Params]("Generate models") {
      opt[Int]("dayStart")
        .text(s"day start")
        .action((x, c) => c.copy(dayStart = x))
      opt[Int]("dayEnd")
        .text(s"dayEnd")
        .action((x, c) => c.copy(dayEnd = x))
      opt[String]("inputFolder")
        .text(s"inputFolder")
        .required()
        .action((x, c) => c.copy(inputFolder = x))
      opt[String]("frequencyLimit")
        .text(s"frequencyLimit")
        .action((x, c) => c.copy(frequencyLimit = Some(x)))
      opt[String]("outputFolder")
        .text(s"outputFolder")
        .action((x, c) => c.copy(outputFolder = Some(x)))
      opt[String]("modelSizeFile")
        .text(s"modelSizeFile")
        .action((x, c) => c.copy(modelSizeFile = Some(x)))
    }
    val params = parser.parse(args, defaultParams).getOrElse(defaultParams)
    print(params)
    val sc = NNContext.initNNContext("generate model")
    val spark = SQLContext.getOrCreate(sc)
    val time_start = System.nanoTime()
    val df = read_parquet(spark, params)
    val df_count_list = get_column_counts_with_frequency_limit(df, params)
    add_id_and_write_data(df_count_list, params)
    val time_end = System.nanoTime()
    println(s"Generate models consuming time: ${(time_end - time_start)/1e9d}")
//    val models = load_column_models(spark,
//      params.outputFolder.getOrElse("/tmp/gen_models/"),
//      params.modelSizeFile.isDefined)
//    val time2 = System.nanoTime()
//    println(s"Total consuming time: ${(time2 - time1)/1e9d}")
  }
}

