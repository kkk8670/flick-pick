from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
spark = SparkSession.builder.master("local").appName("joinRatingMovie").getOrCreate()

# file path
ratings_path = "/content/drive/MyDrive/BigData/smallRating/ratings.csv"
movies_path = "/content/drive/MyDrive/BigData/ml-latest-small/movies.csv"

# read ratings.csv
ratings_df = spark.read.option("header", "true").option("inferSchema", "true").csv(ratings_path)

# read movies.csv
movies_df = spark.read.option("header", "true").option("inferSchema", "true").csv(movies_path)

ratings_df.show()
movies_df.show()

# merge two DataFrame (using movieId)
merged_df = ratings_df.join(movies_df, on="movieId", how="inner")
merged_df.show()

# Reorder columns to put 'userId' as the first column and sort by 'userId'
merged_df = merged_df.select("userId", *[col for col in merged_df.columns if col != "userId"]) \
    .orderBy("userId")

merged_df.show(truncate=False)

# add a new column "movieLink" which shows the link of the corresponding movie
merged_df = merged_df.withColumn('movieLink',concat(lit('https://movielens.org/movies/'), col('movieId').cast('string')))
merged_df.show()

# divide the "title" column into two columns: "titleClean" and "year"
# Extract movie title (remove last space and year in parentheses)
merged_df = merged_df.withColumn('titleClean', regexp_extract(col('title'), r'^(.*)\s\(\d{4}\)$', 1))

# Extract year (4-digit number in parentheses)
merged_df = merged_df.withColumn('year', regexp_extract(col('title'), r'\((\d{4})\)', 1))

# results
merged_df.show()
merged_df.printSchema()

import pandas as pd
# Convert the Spark DataFrame to a Pandas DataFrame
merged_pd_df = merged_df.toPandas()

# Check for missing values (nulls) in each column
print("Missing values in each column:")
print(merged_pd_df.isnull().sum())

# Check for duplicate rows
duplicate_rows = merged_pd_df.duplicated()
print(f"\nNumber of duplicate rows: {duplicate_rows.sum()}")

# save the processed data to a CSV file
outputFilePath = "/content/drive/MyDrive/BigData/smallRatingMovie2"
merged_df.coalesce(1).write.option("header", "true").csv(outputFilePath)