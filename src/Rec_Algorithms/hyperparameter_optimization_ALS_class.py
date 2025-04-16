import os
from pathlib import Path
from pyspark.sql.functions import lit
import numpy as np
from dotenv import load_dotenv
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, desc, explode, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.recommendation import ALSModel


load_dotenv()
ROOT_DIR = Path(os.getenv('ROOT_DIR'))

def cosine_similarity_udf(v1, v2):
    """
    Calculate cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity score (float)
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    return float(
        np.dot(v1, v2) / (norm_v1 * norm_v2)
        if norm_v1 != 0 and norm_v2 != 0
        else 0.0
    )


class Recommendation:
    def __init__(self):

        self.data_path = ROOT_DIR / "data"
        self.processed_path = str(self.data_path / "processed")
        self.result_path = str(self.data_path / "result")

        self.spark = (SparkSession.builder
                        .appName("MovieRecommendation")
                        # .config("spark.executorEnv.PYSPARK_PYTHON", os.getenv("PYSPARK_PYTHON"))
                        # .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON",
                        #         os.getenv("PYSPARK_PYTHON"))
                        # .config("spark.python.profile", "false")
                        # .config("spark.hadoop.validateOutputSpecs", "false")
                        .config("spark.driver.host", "localhost") 
                        .getOrCreate()
                    )
        self.spark.sparkContext.setLogLevel("ERROR")


    def setup_environment(self):
        """Configure and validate environment variables."""
        env_path = Path(__file__).parent.parent.parent / '.env'
        print(f"Loading environment variables from: {env_path}")

        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_path}")

        load_dotenv(dotenv_path=env_path, override=True)

        required_vars = ["PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"]
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"Required environment variable {var} not set")

        print("Environment variables loaded successfully:")
        print(f"PYSPARK_PYTHON={os.getenv('PYSPARK_PYTHON')}")
        print(f"PYSPARK_DRIVER_PYTHON={os.getenv('PYSPARK_DRIVER_PYTHON')}")

        os.environ.update({
            "PYSPARK_PYTHON": os.getenv("PYSPARK_PYTHON"),
            "PYSPARK_DRIVER_PYTHON": os.getenv("PYSPARK_DRIVER_PYTHON"),
            "PATH": f"{os.path.dirname(os.getenv('PYSPARK_PYTHON'))};"
                    f"{os.environ['PATH']}"
        })



    def load_data(self):
        """Load ratings, movie, and tag datasets."""
        # Note: Preserving timestamp field if needed
        ratings = "clean_ratings.csv"
        tags = "extra_tags.csv"
        rating_movie = "rating_movie.csv"

        self.ratings_df = (
            self.spark.read.csv(f"{self.processed_path}/{ratings}", header=True, inferSchema=True)
            .select("userId", "movieId", "rating","timestamp")
            .withColumn("userId", col("userId").cast("int"))
            .withColumn("movieId", col("movieId").cast("int"))
            .withColumn("rating", col("rating").cast("float"))
            .withColumn("timestamp", col("timestamp").cast("int"))
        )

        # Load movie data with standardized column prefixes
        self.movie_df = (
            self.spark.read.csv(f"{self.processed_path}/{rating_movie}",
                           header=True,
                           inferSchema=True)
            .select(
                col("movieId").alias("m_movieId"),
                col("title").alias("m_title"),
                col("genres").alias("m_genres"),
                col("movieLink").alias("m_movieLink"),
                col("titleClean").alias("m_titleClean"),
                col("year").alias("m_year")
            )
            .withColumn("m_movieId", col("m_movieId").cast("int"))
        )

        # Load tag data with standardized column prefixes
        self.tags_df = (
            self.spark.read.csv(f"{self.processed_path}/{tags}",
                           header=True,
                           inferSchema=True)
            .select(
                col("movieId").alias("t_movieId"),
                col("tags_str").alias("t_tags_str")
            )
            .withColumn("t_movieId", col("t_movieId").cast("int"))
        )



    def build_base_als(self):
        """Create base ALS model for parameter optimization."""
        return ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
            rank=10,  
            maxIter=10,   
            regParam=0.1   
        )


    def hyperparameter_optimization(self, train_ratings_df):
        """
        Perform hyperparameter tuning for the ALS model.

        Args:
            ratings_df: DataFrame containing ratings data

        Returns:
            Best trained ALS model based on cross-validation
        """
        # Cache data for faster repeated access
        train_ratings_df.cache()
        print(train_ratings_df.head())

        als = self.build_base_als()
        param_grid = (
            ParamGridBuilder()
            .addGrid(als.rank, [10, 20, 50])
            .addGrid(als.maxIter, [10, 20])
            .addGrid(als.regParam, [0.01, 0.1, 1.0])
            .build()
        )

        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )

        crossval = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=4  # Adjust based on cluster configuration
        )

        cv_model = crossval.fit(train_ratings_df)

        # Release cache when done
        train_ratings_df.unpersist()

        return cv_model.bestModel


    def get_model(self, force_train=False):

        model_path = str(ROOT_DIR / "models/als_recommendation")

        if os.path.exists(model_path) and not force_train:
            print("Loading existing model")
            best_model = ALSModel.load(model_path)
        else:
            print("training new model")
            train_ratings_df, _ = self.ratings_df.randomSplit([0.8, 0.2], seed=42)
            best_model = self.hyperparameter_optimization(train_ratings_df)
            best_model.write().overwrite().save(model_path)

        return best_model


    def get_ALS_recommendations(self, best_als_model, user_id):
        # Step 1: Get ALS reçcommendations
        user_recs = best_als_model.recommendForUserSubset(
                self.spark.createDataFrame([(user_id,)], ["userId"]),
                100
            )
        user_rec_movies = user_recs.withColumn(
            "rec", explode("recommendations")
        ).select(
            "userId",
            col("rec.movieId").alias("movie_rec_id"),  # Renamed to avoid conflicts
            col("rec.rating").alias("als_rating")
        )

        return user_rec_movies
        

    def get_content_feature_matrix(self):
        # Step 2: Build content feature matrix
        print("Building movie content feature matrix...")

        movie_tags_enriched = self.tags_df.alias("tags").join(
            self.movie_df.alias("movies"),
            col("tags.t_movieId") == col("movies.m_movieId"),
            "left"
        ).withColumn(
            "combined_tags",
            F.concat_ws(" ",
                F.coalesce("tags.t_tags_str", lit("")),
                "movies.m_genres",
                "movies.m_titleClean",
                F.col("movies.m_year").cast("string"))
        ).filter(col("combined_tags") != "")
        print("movie_tags_enriched", movie_tags_enriched.show(n=5))

        # TF-IDF pipeline
        tokenizer = Tokenizer(inputCol="combined_tags", outputCol="words")
        hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(),
                               outputCol="rawFeatures",
                               numFeatures=100)
        idf = IDF(inputCol=hashing_tf.getOutputCol(),
                  outputCol="features")

        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
        tags_df_transformed = pipeline.fit(movie_tags_enriched).transform(movie_tags_enriched)
        return tags_df_transformed, hashing_tf


    def get_user_profile(self, user_id, tags_df_transformed, hashing_tf):
        # Step 3: Build user profile
        print("👤 Building user preference profile...")
        user_history = self.ratings_df.alias("ratings").filter(
            col("ratings.userId") == user_id
        )

        user_tag_features = user_history.join(
            tags_df_transformed.alias("features"),
            col("ratings.movieId") == col("features.t_movieId"),
            "inner"
        ).select("features.features").cache()

        # Handle cold start
        if user_tag_features.count() == 0:
            print(f"No valid ratings found for user {user_id}, using default vector")
            avg_vector = np.zeros(hashing_tf.getNumFeatures())
        else:
            avg_vector = user_tag_features.rdd.map(
                lambda row: row["features"].toArray()
            ).mean()

        avg_vector_broadcast = self.spark.sparkContext.broadcast(avg_vector)

        return avg_vector_broadcast, user_tag_features


    def get_content_similarity(self, user_rec_movies, tags_df_transformed, avg_vector_broadcast):
        # Step 4: Calculate content similarity
        print("Calculating content similarity scores...")
        rec_with_features = user_rec_movies.alias("recs").join(
            tags_df_transformed.alias("features"),
            col("recs.movie_rec_id") == col("features.t_movieId"),
            "left"
        )

        avg_vector = avg_vector_broadcast.value

        similarity_udf = udf(
            lambda v: cosine_similarity_udf(avg_vector, v) if v else 0.0,
            DoubleType()
        )

        rec_with_similarity = rec_with_features.withColumn(
            "content_similarity",
            similarity_udf(col("features.features"))
        )
        return rec_with_similarity
    

    def get_final_combined(self, rec_with_similarity, user_id):
        # Step 5: Hybrid scoring
        print("Performing hybrid ranking...")
        max_rating = rec_with_similarity.agg({"als_rating": "max"}).collect()[0][0]
        rec_normalized = rec_with_similarity.withColumn(
            "norm_rating",
            col("als_rating") / (max_rating if max_rating != 0 else 1.0)
        )

        rec_with_score = rec_normalized.withColumn(
            "final_score",
            0.7 * col("norm_rating") + 0.3 * col("content_similarity")
        ).cache()
            
        try:
            final_recs = (
                rec_with_score.alias("scores")
                .join(self.movie_df.alias("movies"), col("scores.movie_rec_id") == col("movies.m_movieId"), "left")
                .join(self.ratings_df.alias("ratings"), col("scores.userId") == col("ratings.userId"), "left")
                .select(
                    col("scores.userId"),
                    col("scores.movie_rec_id").alias("movieId"),
                    col("movies.m_title").alias("title"),
                    col("movies.m_genres").alias("genres"),
                    col("ratings.timestamp").cast("int").alias("timestamp"),  
                    col("scores.als_rating").alias("rating"),
                    lit(None).cast("double").alias("similarity"), 
                    lit(None).cast("double").alias("final_score"), 
                    lit("recommendation").alias("data_type")
                )
            )
            # print("final_recs:", final_recs.columns)    
            user_history_out = (
                self.ratings_df.filter(col("userId") == user_id)
                .join(self.movie_df, col("movieId") == col("m_movieId"), "left")
                .select(
                    col("userId"),
                    col("movieId"),
                    col("m_title").alias("title"),
                    col("m_genres").alias("genres"),
                    col("timestamp").cast("int"), 
                    col("rating"),
                    lit(None).cast("double").alias("similarity"),
                    lit(None).cast("double").alias("final_score"),
                    lit("history").alias("data_type")
                )
            )
            # print("user_history_out:", user_history_out.count())    
            final_combined = user_history_out.unionByName(
                    final_recs.select(*user_history_out.columns)
                )
            # print("final_combined:", final_combined.count())  
            return final_combined, rec_with_score
        
        except Exception as e:
            print("combine data error:", e)
    

    def export_result(self, user_id, final_combined):
        # Step 7: Export results
        output_path = str(self.result_path / f"user_{user_id}_recommendation_and_history")
        final_combined.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
        print(f"Results saved to: {output_path}")

    
    def weighted_recommendations(self, user_id, best_als_model):
        """
        Hybrid recommendation system combining ALS and content-based filtering.

        Args:
            user_id: Target user ID
            best_als_model: Trained ALS model
            movie_df: DataFrame containing movie metadata
            tags_df: DataFrame containing movie tags
            ratings_df: DataFrame containing user ratings
            spark: Spark session

        Returns:
            Transformed tags DataFrame with features
        """
        try:
            print(f"Generating hybrid recommendations for user {user_id}...")

            user_rec_movies = self.get_ALS_recommendations(best_als_model, user_id)
            # print("user_rec_movies:", user_rec_movies.show(n=5))
            # |userId|movie_rec_id|als_rating|

            tags_df_transformed, hashing_tf = self.get_content_feature_matrix()
            print("tags_df_transformed:::::", tags_df_transformed.show(n=5))

            # avg_vector_broadcast, user_tag_features = self.get_user_profile(user_id, tags_df_transformed, hashing_tf)

            # rec_with_similarity = self.get_content_similarity(user_rec_movies, tags_df_transformed, avg_vector_broadcast)

            # final_combined, rec_with_score = self.get_final_combined(rec_with_similarity, user_id)
            # # print("final_combined:",final_combined)
            # self.export_result(user_id, final_combined)

            # # Clean cached data
            # rec_with_score.unpersist()
            # if user_tag_features:
            #     user_tag_features.unpersist()

            # return tags_df_transformed 

        except Exception as e:
            print(f"Recommendation generation failed for user {user_id}: {str(e)}")
            raise  # Re-raise exception for outer handling


    def export_movie_similarity_network(self, tags_df_transformed):
        """
        Calculate content similarity between movies for network visualization.

        Args:
            tags_df_transformed: Transformed tags DataFrame with features
            movie_df: DataFrame containing movie metadata
            spark: Spark session
        """
        from pyspark.ml.linalg import Vectors
        from itertools import combinations
        from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

        # Extract movieId and feature vectors
        movie_vectors = tags_df_transformed.select(
            col("t_movieId"),
            col("features")
        ).rdd.map(
            lambda row: (row["t_movieId"], row["features"].toArray())
        )
        movie_vector_list = movie_vectors.collect()

        # Calculate pairwise cosine similarity (limit to first 500 movies)
        result = []
        for (id1, vec1), (id2, vec2) in combinations(movie_vector_list[:500], 2):
            sim = cosine_similarity_udf(vec1, vec2)
            if sim > 0.6:
                result.append((id1, id2, sim))

        schema = StructType([
            StructField("movieId1", IntegerType()),
            StructField("movieId2", IntegerType()),
            StructField("similarity", DoubleType())
        ])
        sim_df = self.spark.createDataFrame(result, schema=schema)

        # Add movie title information
        sim_df = (
            sim_df
            .join(self.movie_df.select("m_movieId", "m_title")
                  .withColumnRenamed("m_movieId", "movieId1")
                  .withColumnRenamed("m_title", "title1"),
                  sim_df["movieId1"] == col("movieId1"))
            .join(self.movie_df.select("m_movieId", "m_title")
                  .withColumnRenamed("m_movieId", "movieId2")
                  .withColumnRenamed("m_title", "title2"),
                  sim_df["movieId2"] == col("movieId2"))
        )

        sim_df.coalesce(1).write.mode("overwrite").option("header", True).csv(
            str(self.result_path / "movie_similarity_network.csv")
        )
        print("Movie similarity network data exported to movie_similarity_network.csv")


    def export_wordcloud_fields(self):
        """
        Export text fields for word cloud visualization.

        Args:
            tags_df: DataFrame containing movie tags
            movie_df: DataFrame containing movie metadata
        """
        # Correct column references
        tags_enriched = self.tags_df.join(
            self.movie_df,
            self.tags_df["t_movieId"] == self.movie_df["m_movieId"],
            "left"
        ).select(
            col("t_movieId").alias("movieId"),
            "t_tags_str",
            "m_genres"
        )

        tags_enriched = tags_enriched.withColumn(
            "wordcloud_text",
            F.concat_ws(" ", "t_tags_str", "m_genres")
        )
     
        tags_enriched.coalesce(1).write.mode("overwrite").option("header", True).csv(
             str(self.result_path / "wordcloud_text.csv")
        )
        print("Word cloud data exported to  wordcloud_text.csv")


    def generate_recommendations(self, best_model):
        user_ids = self.ratings_df.select("userId").distinct().rdd.map(
            lambda row: row["userId"]
        ).collect()
        print(f"Generating recommendations for {len(user_ids)} users...")

        tags_df_transformed = None
        for user_id in user_ids:
            try:
                tags_df_transformed = self.weighted_recommendations(user_id, best_model)
                break
            except Exception as e:
                print(f"Recommendation failed for user {user_id}, skipping: {str(e)}")
                continue

        if tags_df_transformed:
            self.export_movie_similarity_network(tags_df_transformed)
            self.export_wordcloud_fields()
        else:
            print("No successful recommendations generated, skipping exports")


def main():
    """Entry point for the recommendation system."""
    rc = Recommendation()
    # rc.setup_environment()
    rc.load_data()
    
    best_model = rc.get_model(force_train=False)

    rc.generate_recommendations(best_model)

    rc.spark.stop()


if __name__ == "__main__":
    main()

