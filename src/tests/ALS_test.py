#!/usr/bin/env python
# @Auther liukun
# @Time 2025/03/26

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.sql import functions as F

import os
os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17'

# file_path = "dbfs:/FileStore/tables/test/ratings.csv"
file_path = "./data/test/ratings.csv"


# init
spark = SparkSession 
    .builder 
    .appName("flick-pick") 
    .config("spark.driver.bindAddress", "localhost") 
    .getOrCreate() 

ratings = spark.read.csv(file_path, header=True, inferSchema=True)

ratings = ratings.select(col("userId"), col("movieId"), col("rating"))

# training
als = ALS(
    rank=5,        # default  10
    maxIter=5,      # default   10   
    regParam=0.1,     # default 0.1   
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"   
)

# 拟合
model = als.fit(ratings)

# recommend for all users
recommendations = model.recommendForAllUsers(10) 

def recommend_to(user_id):
    # user_df = spark.createDataFrame([(user_id,)], ["userId"])
    # recommendations = model.recommendForUserSubset(user_df, numItems=10)

    recommendations = recommendations.filter(col("userId") == user_id) 
    recommendations.show(5, truncate=False)  # show 5



def flat_data(recommendations):
    flat_recommendations = recommendations.withColumn("rec", F.explode("recommendations"))
    flat_recommendations = flat_recommendations.select(
        "userId", 
        F.col("rec.movieId").alias("movieId"), 
        F.col("rec.rating").alias("rating")
    )

    # save
    pd = flat_recommendations.toPandas()
    pd.to_csv('./data/for_visual_sample.csv', index=True, header=True)

# recommend_to()
flat_data(recommendations)

spark.stop()