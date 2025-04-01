from pyspark.sql import SparkSession
import pandas as pd

# Create a SparkSession
spark = SparkSession.builder.master("local").appName("Processing").getOrCreate()

# Read CSV files, preserving headers and automatically inferring data types
inputFilePath = "/content/tags.csv"
df = spark.read.option("header", "true").option("inferSchema", "true").csv(inputFilePath)

df.printSchema()
df.show()

# Check for missing values
print("Check for missing values:")
dfP = pd.read_csv(inputFilePath)
print(dfP.isnull().sum())

# Get the de-duplicated DataFrame
distinct_df = df.dropDuplicates()

# Compare the number of rows before and after de-duplication to determine if there are any duplicates
if df.count() == distinct_df.count():
    print("No duplicate rows")
else:
    print("Rows with duplicates")

# Check the type of each column
df.printSchema()

# Cache the DataFrame into memory
distinct_df.cache()

# Display first 5 lines (for checking data)
distinct_df.show(5)

# Save the DataFrame as a single CSV file, preserving the table header
distinct_df.coalesce(1).write.option("header", "true").mode("overwrite").csv("/content/csv_tags")

# Show CSV files
import os
print(os.listdir("/content/csv_tags"))
