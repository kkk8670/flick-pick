#!/usr/bin/env python
# @Auther liukun
# @Time 2025/04/13

import os, json, time, random, glob
import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALSModel


load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
model_path =    f"{ROOT_DIR}/models/als_recommendation"  
movie_path = f"{ROOT_DIR}/data/raw/ml-latest-small/movies.csv" 
input_path = f"{ROOT_DIR}/data/streaming_input/" 
output_path = f"{ROOT_DIR}/data/output/realtime_streaming_recommend.csv"

def init_spark():
    spark = SparkSession.builder \
        .appName("flick-pick") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("INFO")
    return spark


def load_model_and_data(spark):
    model = ALSModel.load(model_path)
    movies = spark.read.csv(movie_path, header=True, inferSchema=True)
    return model, movies

 
def process_batch(batch_df, batch_id, model, movies, spark):
    if batch_df.count() == 0:
        print(f"[batch {batch_id}] empty, skipping.")
        return

    user_ids = [row["userId"] for row in batch_df.select("userId").distinct().collect()]
    user_df = spark.createDataFrame([(uid,) for uid in user_ids], ["userId"])

    recs = model.recommendForUserSubset(user_df, 10)

    recs = recs.selectExpr("userId", "explode(recommendations) as rec") \
               .select("userId", col("rec.movieId"), col("rec.rating"))

    pandas_df = recs.join(movies, on="movieId", how="left").toPandas()

    # save  
    header = not os.path.exists(output_path)  
    pandas_df.to_csv(output_path, mode='a', header=header, index=False)
    # recs.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
    print(f"[batch {batch_id}] recommendations written.")


def start_streaming(spark, model, movies):
    schema = "userId INT, movieId INT, rating FLOAT, timestamp LONG"
    rating_stream = (spark.readStream  
            .schema(schema)  
            .json(input_path)
        )
    def process_batch_wrapper(batch_df, batch_id):
        process_batch(batch_df, batch_id, model, movies, spark)
    
    query = (
            rating_stream.writeStream  
                .trigger(processingTime="10 seconds")  
                .foreachBatch(process_batch_wrapper) 
                .start()
        )
    return query


def get_realtime_recommendations(user_id, top_n=10, spark=None, model=None, movies=None):
    if spark is None:
        spark = init_spark()
    if model is None or movies is None:
        model, movies = load_model_and_data(spark)
    
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    recs = model.recommendForUserSubset(user_df, top_n)
    recs = recs.selectExpr("userId", "explode(recommendations) as rec") \
               .select("userId", col("rec.movieId"), col("rec.rating"))
    joined = recs.join(movies, on="movieId", how="left").toPandas()
    df = joined.sort_values(by="rating", ascending=False).head(10)
    return df.to_dict(orient="records")


def get_latest_recommendations_from_csv(user_id=None):
    if not os.path.exists(output_path):
        return []
    
    df = pd.read_csv(output_path)
    if user_id:
        df = df[df['userId'] == user_id]
    df = df.sort_values(by="rating", ascending=False).head(10)  
    return df.to_dict(orient="records")
     

def main():
    spark = init_spark()
    model, movies = load_model_and_data(spark)
    query = start_streaming(spark, model, movies)
    query.awaitTermination()


if __name__ == "__main__":
    main()

