#!/usr/bin/env python
# @Auther liukun
# @Time 2025/03/26

from pyspark.sql import SparkSession, Row
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, split, concat_ws, concat, lit
from pyspark.sql import functions as F
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import HashingTF, IDF
from scipy.spatial.distance import cosine
import numpy as np

import os
from pathlib import Path
from dotenv import load_dotenv

# env
load_dotenv()
ROOT_DIR = Path(os.getenv('ROOT_DIR'))
rating_path = str(ROOT_DIR / "data/test/ratings.csv")
movies_path = str(ROOT_DIR / "data/test/movies.csv")
tags_path = str(ROOT_DIR / "data/test/tags.csv")

spark = (
    SparkSession
    .builder 
    .appName("flick-pick") 
    .config("spark.driver.host", "localhost") 
    .getOrCreate() 
)

def ALS_init():
    # init

    ratings = spark.read.csv(rating_path, header=True, inferSchema=True)

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
    
    # recommend_to()
    flat_data(recommendations)

    spark.stop()

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
    pd.to_csv(ROOT_DIR / 'data/processed/for_visual_sample.csv', index=True, header=True)

def TF_IDF(movies_with_tags):
    #  TF-IDF for similarity
    tfidf = HashingTF(inputCol="features", outputCol="tf_features")
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
    pipeline = Pipeline(stages=[tfidf, idf])
    tfidf_model = pipeline.fit(movies_with_tags)
    movies_tfidf = tfidf_model.transform(movies_with_tags)

    movies_pd = movies_tfidf.select("movieId", "tfidf_features").toPandas()
    return movies_pd


def calc_similarity(movie_ids, tfidf_vectors):
    sim_matrix = np.zeros((len(movie_ids), len(movie_ids)))
    for i in range(len(movie_ids)):
        for j in range(i+1, len(movie_ids)):
            sim = 1 - cosine(tfidf_vectors[i], tfidf_vectors[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim
    return sim_matrix


def similar_movie_topN(top_n, movie_ids, sim_matrix):
 
    similar_movies = []
    for i in range(len(movie_ids)):
        sim_scores = list(zip(movie_ids, sim_matrix[i]))
        sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]  
        similar_movies.append({
            "movieId": movie_ids[i],
            "similar_movies": [x[0] for x in sim_scores_sorted],
            "similarity_scores": [x[1] for x in sim_scores_sorted]
        })
    return similar_movies


def similarity_init():
    movies = spark.read.csv(movies_path, header=True, inferSchema=True)
    tags = spark.read.csv(tags_path, header=True, inferSchema=True)
    movie_tags = tags.groupBy("movieId").agg(
        F.concat_ws(" ", F.collect_list("tag")).alias("all_tags")  #  
    )

    movies_with_features = movies.join(movie_tags, on="movieId", how="left").fillna({"all_tags": ""})

    movies_with_features = movies_with_features.withColumn(
        "genres_array",
        split(col("genres"), "\\|")
    )

    movies_with_features = movies_with_features.withColumn(
    "tags_array",
    split(col("all_tags"), " ")
    )

    movies_with_features = movies_with_features.withColumn(
        "features",
        F.concat(col("genres_array"), col("tags_array"))
    )

    movies_pd = TF_IDF(movies_with_features.select("movieId", "features"))

    movie_ids = movies_pd["movieId"].values
    tfidf_vectors = [v.toArray() for v in movies_pd["tfidf_features"]]
    sim_matrix = calc_similarity(movie_ids, tfidf_vectors)

    similar_movies = similar_movie_topN(top_n, movie_ids, sim_matrix)

    similar_movies_df = spark.createDataFrame(similar_movies)
    similar_movies_df.write.csv(
        "movie_similarity",
        header=True,
        mode="overwrite"
    )

if __name__ == "__main__":
    # ALS_init()
    similarity_init()