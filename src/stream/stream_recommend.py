#!/usr/bin/env python
# @Auther liukun
# @Time 2025/04/13

import os, json, time, random, glob
from pathlib import Path
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALSModel


load_dotenv()
ROOT_DIR = Path(os.getenv('ROOT_DIR'))
model_path = str(Path(os.getenv('ROOT_DIR')) / "models/als_model" ) 
movie_path = str(Path(os.getenv('ROOT_DIR')) / "data/test/movies.csv")
inpput_path = str(Path(os.getenv('ROOT_DIR')) / "data/streaming_input/")
output_path = str(Path(os.getenv('ROOT_DIR')) / "data/output/streaming_recommend")

spark = SparkSession.builder \
    .appName("flick-pick") \
    .getOrCreate()

spark.sparkContext.setLogLevel("INFO")

schema = "userId INT, movieId INT, rating FLOAT, timestamp LONG"

model = ALSModel.load(model_path)

movies = spark.read.csv(movie_path, header=True, inferSchema=True)

rating_stream = (spark.readStream  
        .schema(schema)  
        .json(inpput_path)
    )
 
def process_batch(batch_df, batch_id):
    if batch_df.count() == 0:
        print(f"[batch {batch_id}] empty, skipping.")
        return

    user_ids = [row["userId"] for row in batch_df.select("userId").distinct().collect()]
    user_df = spark.createDataFrame([(uid,) for uid in user_ids], ["userId"])

    recs = model.recommendForUserSubset(user_df, 10)

    recs = recs.selectExpr("userId", "explode(recommendations) as rec") \
               .select("userId", col("rec.movieId"), col("rec.rating"))

    recs = recs.join(movies, on="movieId", how="left")

    # save as parquet  
    recs.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
    print(f"[batch {batch_id}] recommendations written.")


def main():
    query = rating_stream.writeStream \
        .foreachBatch(process_batch) \
        .start()

    query.awaitTermination()


if __name__ == "__main__":
    main()

