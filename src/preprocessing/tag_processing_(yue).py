from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
spark = SparkSession.builder.master("local").appName("tagProcessing").getOrCreate()

# Enter file path
inputFilePath = "/content/drive/MyDrive/BigData/ml-latest-small/tags.csv"
df = (spark.read.option("header", "true").option("inferSchema", "true").csv(inputFilePath) )
df.printSchema()
df.show()

# Select only 'movieId' and 'tag' columns
df = df.select("movieId", "tag").orderBy("movieId")

# Group by 'movieId' and aggregate tags into a list, then join them into a single string
grouped_df = df.groupBy("movieId") \
    .agg(collect_list("tag").alias("tags")) \
    .withColumn("tags_str", concat_ws(" ", "tags"))
df.show()
grouped_df.show()

import pandas as pd
# Convert the Spark DataFrame to a Pandas DataFrame
grouped_pd_df = grouped_df.toPandas()

# Check for missing values (nulls) in each column
print("Missing values in each column:")
print(grouped_pd_df.isnull().sum())

# Check for duplicate rows
duplicate_rows = grouped_pd_df[["movieId", "tags_str"]].duplicated()
print(f"\nNumber of duplicate rows: {duplicate_rows.sum()}")

# save the processed data to a CSV file
outputFilePath = "/content/drive/MyDrive/BigData/smallTag"
grouped_df.select("movieId", "tags_str").coalesce(1).write.option("header", "true").csv(outputFilePath)