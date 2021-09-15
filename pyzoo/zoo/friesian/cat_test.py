from zoo.friesian.feature import FeatureTable
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
input = "/home/yina/Documents/data/cat_str"
item_df = spark.read.parquet(input)
item_df.show()
print(item_df.count())
item_tbl = FeatureTable(item_df)
category_index = item_tbl.gen_string_idx(["category"])
encoded = item_tbl.encode_string(["category"], category_index)
print(item_tbl.df.select("category").distinct().count())
print(category_index[0].df.select("category").count())
print(encoded.df.select("category").count())
