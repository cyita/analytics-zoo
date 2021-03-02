package com.intel.analytics.zoo.examples.dlrm

import com.intel.analytics.zoo.common.NNContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext}
import scopt.OptionParser

object FilterGroupBy {
  val CAT_COLS: List[Int] = (14 until 40).toList

  def read_parquet(spark: SQLContext, params: Params): Dataset[Row] ={
    val dayRange = (params.dayStart to params.dayEnd).toList
    val paths= dayRange.map(x => s"${params.inputFolder}/day_${x}.parquet")
    spark.read.parquet(paths:_*)
  }

  def get_column_counts_with_frequency_limit(df: Dataset[Row], params: Params): List[DataFrame] = {
    val cols = CAT_COLS.map(x => s"_c${x}")
    var default_limit: Option[Int] = None
    val freq_map = scala.collection.mutable.Map[String, Int]()
    if(params.frequencyLimit.isDefined) {
      val freq_list = params.frequencyLimit.orNull.split(",")
      for(fl <- freq_list){
        val frequency_pair = fl.split(":")
        if(frequency_pair.length == 1){
          default_limit = Some(frequency_pair(0).toInt)
        } else if (frequency_pair.length == 2){
          freq_map += (frequency_pair(0) -> frequency_pair(1).toInt)
        }
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

      df_col_filtered.rdd.count()

      df_col_filtered
    })

    df_count_filtered_list
  }

  def main(args: Array[String]): Unit = {
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
    }
    val params = parser.parse(args, defaultParams).getOrElse(defaultParams)
    print(params)
    val sc = NNContext.initNNContext("Generate models filter and groupby")
    val spark = SQLContext.getOrCreate(sc)
    spark.sparkContext.setLogLevel("WARN")
    val time_start = System.nanoTime()
    val df = read_parquet(spark, params)
    val df_count_list = get_column_counts_with_frequency_limit(df, params)
    val time_end = System.nanoTime()
    println("Filter group by time: " + (time_end - time_start) / 1e9d)
    df.show(5)
    sc.stop()
  }
}
