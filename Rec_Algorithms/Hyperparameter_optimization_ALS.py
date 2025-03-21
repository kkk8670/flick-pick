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

    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    crossval = CrossValidator(
        estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3
    )

    cv_model = crossval.fit(ratings_df)
    return cv_model.bestModel


def model_evaluation(model, test_data, evaluator):
    """Evaluate the model using RMSE metric."""
    predictions = model.transform(test_data)
    return evaluator.evaluate(predictions)


def train_and_evaluate_models(
    train_data, test_data, movie_df, links_df, user_id, tags_df, best_als_model
):
    """Train and evaluate ALS models, then recommend movies for a given user."""
    evaluation_results = []

    base_als_model = build_base_als().fit(train_data)
    evaluator = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    )
    base_als_rmse = model_evaluation(base_als_model, test_data, evaluator)
    evaluation_results.append(("Base ALS", base_als_rmse))

    best_als_rmse = model_evaluation(best_als_model, test_data, evaluator)
    evaluation_results.append(("Optimized ALS", best_als_rmse))

    for model_name, rmse in evaluation_results:
        print(f"{model_name} RMSE: {rmse}")

    user_recommendations_als = best_als_model.recommendForUserSubset(
        spark.createDataFrame([(user_id,)], ["userId"]), 10
    )
    user_rec_movies_als = user_recommendations_als.withColumn("rec", explode("recommendations")) \
        .select("userId", "rec.movieId", "rec.rating")

    user_rec_with_titles_als = (
        user_rec_movies_als.join(movie_df, "movieId", "inner")
        .join(links_df, "movieId", "left")
        .select("userId", "movieId", movie_df["title"].alias("movie_title"), "rating", "tmdbId")
    )

    tokenizer = Tokenizer(inputCol="tags_str", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
    tags_df_transformed = pipeline.fit(tags_df).transform(tags_df)

    user_tags_df = tags_df_transformed.join(movie_df, "movieId").select("movieId", "features")
    content_recommendations = (
        user_tags_df.join(user_rec_with_titles_als, "movieId", "inner")
        .select(
            "movieId",
            "movie_title",
            user_rec_with_titles_als["rating"].alias("user_rating"),
            "tmdbId",
            "features",
        )
        .orderBy(desc("user_rating"))
    )

    print("Content-based recommendations:")
    content_recommendations.show(truncate=False)

    user_rec_with_titles_als = user_rec_with_titles_als.alias("als")
    content_recommendations = content_recommendations.alias("content")
    final_recommendations = user_rec_with_titles_als.join(
        content_recommendations, user_rec_with_titles_als["movieId"] == content_recommendations["movieId"], "outer"
    ).select(
        col("als.userId"),
        col("als.movieId"),
        col("als.movie_title").alias("als_movie_title"),
        col("content.movie_title").alias("content_movie_title"),
        col("als.tmdbId").alias("als_tmdbId"),
        col("content.tmdbId").alias("content_tmdbId"),
        col("als.rating"),
    )

    print("Final merged recommendations:")
    final_recommendations.show(truncate=False)


def main():
    """Main function to load data, train models, and save the best model."""
    ratings_df, movie_df, tags_df, links_df = load_data()
    train_data, test_data = ratings_df.randomSplit([0.8, 0.2], seed=42)

    user_id = 1  # Example user ID
    best_model = hyperparameter_optimization(train_data)
    train_and_evaluate_models(train_data, test_data, movie_df, links_df, user_id, tags_df, best_model)

    save_path = "E:/prepare/se_rec/model"
    os.makedirs(save_path, exist_ok=True)
    best_model.write().overwrite().save(save_path)

    spark.stop()


if __name__ == "__main__":
    main()
