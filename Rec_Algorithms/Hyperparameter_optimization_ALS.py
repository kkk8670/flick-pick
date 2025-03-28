import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import explode, col, desc
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import collect_list, concat_ws

# Set environment variables
os.environ["HADOOP_HOME"] = "D:/download_app/hadoop"
os.environ["SPARK_HOME"] = (
    "C:\\Users\\Zelda\\AppData\\Roaming\\Python\\Python311\\site-packages\\pyspark"
)
os.environ["PYSPARK_PYTHON"] = "D:\\download_app\\anaconda\\envs\\start2\\python.exe"

# Initialize SparkSession
spark = (
    SparkSession.builder.appName("MovieRecommendation")
    .config("spark.hadoop.validateOutputSpecs", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


def load_data():
    """Load rating, movie, and tag datasets."""
    ratings_df = (
        spark.read.csv("../data/test/ratings.csv", header=True, inferSchema=True)
        .select("userId", "movieId", "rating")
        .withColumn("userId", col("userId").cast("int"))
        .withColumn("movieId", col("movieId").cast("int"))
        .withColumn("rating", col("rating").cast("float"))
    )

    movie_df = spark.read.csv("../data/test/movies.csv", header=True, inferSchema=True)

    tags_df = (
        spark.read.csv("../data/test/tags.csv", header=True, inferSchema=True)
        .select("movieId", "tag")
        .groupby("movieId")
        .agg(collect_list("tag").alias("tags"))
    )
    tags_df = tags_df.withColumn("tags_str", concat_ws(" ", "tags"))

    links_df = spark.read.csv("../data/test/links.csv", header=True, inferSchema=True)

    return ratings_df, movie_df, tags_df, links_df


def build_base_als():
    """Define the base ALS model."""
    return ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
    )


def hyperparameter_optimization(ratings_df):
    """Perform hyperparameter tuning for the ALS model."""
    als = build_base_als()
    param_grid = (
        ParamGridBuilder()
        .addGrid(als.rank, [10, 20, 50])
        .addGrid(als.maxIter, [10, 20])
        .addGrid(als.regParam, [0.01, 0.1, 1.0])
        .build()
    )
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    crossval = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
    cv_model = crossval.fit(ratings_df)
    return cv_model.bestModel


def weighted_recommendations(user_id, best_als_model, movie_df, tags_df):
    """Generate weighted recommendations combining ALS and content-based filtering."""
    user_recommendations = best_als_model.recommendForUserSubset(
        spark.createDataFrame([(user_id,)], ["userId"]), 10
    )
    user_rec_movies = user_recommendations.withColumn("rec", explode("recommendations")) \
        .select("userId", "rec.movieId", "rec.rating")

    tokenizer = Tokenizer(inputCol="tags_str", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
    tags_df_transformed = pipeline.fit(tags_df).transform(tags_df)

    user_tags_df = tags_df_transformed.join(movie_df, "movieId").select("movieId", "features")
    content_recommendations = (
        user_tags_df.join(user_rec_movies, "movieId", "inner")
        .select("movieId", "features", user_rec_movies["rating"].alias("als_rating"))
    )

    weighted_recommendations = content_recommendations.withColumn("final_score",
                                                                  0.7 * col("als_rating") + 0.3 * col("features"))
    weighted_recommendations = weighted_recommendations.orderBy(desc("final_score"))

    print("Weighted Recommendations:")
    weighted_recommendations.show(truncate=False)


def main():
    """Main function to load data, train models, and save the best model."""
    ratings_df, movie_df, tags_df, links_df = load_data()
    train_data, _ = ratings_df.randomSplit([0.8, 0.2], seed=42)
    user_id = 1  # Example user ID
    best_model = hyperparameter_optimization(train_data)
    weighted_recommendations(user_id, best_model, movie_df, tags_df)

    save_path = "E:/prepare/se_rec/model"
    os.makedirs(save_path, exist_ok=True)
    best_model.write().overwrite().save(save_path)

    spark.stop()


if __name__ == "__main__":
    main()
