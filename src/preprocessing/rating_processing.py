from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
spark = SparkSession.builder.master("local").appName("ratingProcessing").getOrCreate()

# Enter file path
inputFilePath = "/content/drive/MyDrive/BigData/ml-latest-small/ratings.csv"
df = (spark.read.option("header", "true").option("inferSchema", "true").csv(inputFilePath) )
df.printSchema()
df.show()

# remove the "timestamp" column
df = df.drop("timestamp")
df.show()

# Convert the data in the "rating" column from double to float type
df = df.withColumn("rating", col("rating").cast(FloatType()))
df.printSchema()

# Check for missing values
import pandas as pd
dfP = pd.read_csv(inputFilePath)
print("Check for missing values：")
print(dfP.isnull().sum())

# Ensure each column's data format is correct
df = df.withColumn("userId", col("userId").cast("int")) \
       .withColumn("movieId", col("movieId").cast("int")) \
       .withColumn("rating", col("rating").cast("float"))
df.printSchema()

# Check for duplicate rows
if df.count() == df.dropDuplicates().count():
    print("No duplicate row")
else:
    print("Duplicate rows exist")

# Cache the data and save it to a CSV file
df.cache()
outputFilePath = "/content/drive/MyDrive/BigData/smallRating"
df.coalesce(1).write.option("header", "true").csv(outputFilePath)