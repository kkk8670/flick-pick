import os, sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import lit, col, desc, explode, udf, count, coalesce, concat_ws, max
from pyspark.sql.types import DoubleType
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.recommendation import ALSModel

import inspect
from functools import wraps

load_dotenv()
ROOT_DIR = os.getenv('ROOT_DIR')
project_path = os.getenv("PYTHONPATH")
if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)
from utils import save_to_csv 


def log_caller(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        stack = inspect.stack()
        caller_function = stack[1].function
        print(f" called with no parameters from: {caller_function}")
        return func(*args, **kwargs)
    return wrapper

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

        self.data_path = f"{ROOT_DIR}/data"
        self.processed_path = f"{self.data_path }/processed"
        self.result_path = f"{self.data_path }/output"  

        self.spark = (SparkSession.builder
                        .appName("MovieRecommendation")
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
        ratings = "clean_ratings.csv"
        tags = "movie_train_tags.csv"

        self.ratings_df = self.spark.read.csv(f"{self.processed_path}/{ratings}", header=True, inferSchema=True).dropDuplicates()

        self.tags_df = self.spark.read.csv(f"{self.processed_path}/{tags}",
                           header=True,
                           inferSchema=True).dropDuplicates()

    def build_base_als(self):
        """Create base ALS model for parameter optimization."""
        return ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            # ratingCol="adjusted_rating",
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
        model_path = f"{ROOT_DIR}/models/als_recommendation"

        from pyspark.sql.functions import row_number
        from pyspark.sql.window import Window

        # 为每个 user 的评分记录添加时间顺序编号
        window_spec = Window.partitionBy("userId").orderBy("timestamp")

        df_with_row = self.ratings_df.withColumn("row_num", row_number().over(window_spec))

        # 计算每个 user 的评分数量
        user_counts = df_with_row.groupBy("userId").count().withColumnRenamed("count", "total_ratings")

        # 加入每条记录在用户下的总评分数
        df_with_total = df_with_row.join(user_counts, on="userId")

        # 设置训练阈值：75% 留在训练集
        from pyspark.sql.functions import floor

        df_labeled = df_with_total.withColumn(
            "is_train",
            (col("row_num") <= floor(col("total_ratings") * 0.8))
        )

        # 拆分
        train_ratings_df = df_labeled.filter("is_train = true").drop("row_num", "total_ratings", "is_train")
        test_ratings_df = df_labeled.filter("is_train = false").drop("row_num", "total_ratings", "is_train")

        self.test_df = test_ratings_df

        if os.path.exists(model_path) and not force_train:
            print("Loading existing model")
            best_model = ALSModel.load(model_path)
            return best_model

        print("Training new model")
        
        best_model = self.hyperparameter_optimization(train_ratings_df)
        best_model.write().overwrite().save(model_path)

        # ✅ 新增验证
        # self.evaluate_top_k(best_model, test_ratings_df, top_k=10)

        return best_model



    def get_content_feature_matrix(self):
        print("Step 0: Build content feature matrix...")
        # self.check_dup(self.tags_df)
 
        movie_features_df = (
                            self.tags_df
                                .withColumn("title", coalesce(self.tags_df["title"], lit(""))) 
                                .withColumn("genres", coalesce(self.tags_df["genres"], lit(""))) 
                                .withColumn("tag", coalesce(self.tags_df["tag"], lit(""))) 
                                .withColumn("year", coalesce(self.tags_df["year"], lit("")))
                            )
        # self.check_dup(movie_features_df)

        movie_features_df = (
                                movie_features_df  
                                .withColumn("combined_features", concat_ws(" ", "title", "genres", "tag", "year"))
                                .where(col("combined_features") != lit(""))  
                                .select("movieId", "combined_features")
                            )

        # self.check_dup(movie_features_df)
        # print("1st", movie_features_df.first().asDict())

        # TF-IDF pipeline
        tokenizer = Tokenizer(inputCol="combined_features", outputCol="words")
        hashing_tf = HashingTF(inputCol=tokenizer.getOutputCol(),
                               outputCol="rawFeatures",
                               numFeatures=100)
        idf = IDF(inputCol=hashing_tf.getOutputCol(),
                  outputCol="features")

        pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
        # tags_df_transformed = pipeline.fit(movie_tags_enriched).transform(movie_tags_enriched)  
        tags_df_transformed = pipeline.fit(movie_features_df).transform(movie_features_df) 

        return tags_df_transformed, hashing_tf


    def get_ALS_recommendations(self, best_als_model, user_id):
        print("Step 1: Get ALS reçcommendations...")
        user_recs = best_als_model.recommendForUserSubset(
                self.spark.createDataFrame([(user_id,)], ["userId"]),
                100
            )
        # print("1st", user_recs.first().asDict())
        user_rec_movies = user_recs.withColumn(
            "rec", explode("recommendations")
        ).select( "userId", "rec.movieId", "rec.rating")
        # print("1st", user_rec_movies.first().asDict())
        return user_rec_movies
        

    def get_user_profile(self, user_id, tags_df_transformed, hashing_tf):
        print("Step 2: Building user preference profile...")
        user_history = self.ratings_df.alias("ratings").filter(
            col("ratings.userId") == user_id
        )

 
        user_tag_features = user_history.join(
            tags_df_transformed,
            "movieId"   
        ).select(
            "movieId",
            "features"
        ).cache()

        # print("user_tag_features 1st", user_history.first().asDict())

        # Handle cold start
        if user_tag_features.count() == 0:
            print(f"No valid ratings found for user {user_id}, using default vector")
            avg_vector = np.zeros(hashing_tf.getNumFeatures())
        else:
            avg_vector = user_tag_features.rdd.map(
                lambda row: row["features"].toArray()
            ).mean()

        avg_vector_broadcast = self.spark.sparkContext.broadcast(avg_vector)

        return user_tag_features, avg_vector_broadcast


    def get_content_similarity(self, user_rec_movies, tags_df_transformed, avg_vector_broadcast):
        print("Step 3: Calculating content similarity scores...")

 
        rec_with_features = user_rec_movies.join(
            tags_df_transformed,
            "movieId"   
        )
        # print("rec_with_features 1st ", rec_with_features.first().asDict())

        avg_vector = avg_vector_broadcast.value

        similarity_udf = udf(
            lambda v: cosine_similarity_udf(avg_vector, v) if v else 0.0,
            DoubleType()
        )

        rec_with_similarity = rec_with_features.withColumn(
            "content_similarity",
        F.when(
                col("features").isNull(), 0.0  
            ).otherwise(
                similarity_udf(col("features"))
            )
        )
        # print("rec_with_similarity 1st ", rec_with_similarity.first().asDict())
        return rec_with_similarity
    

    def get_final_combined(self, rec_with_similarity, user_id):
        print("Step 4: Performing hybrid ranking...")
        max_rating = rec_with_similarity.agg({"rating": "max"}).collect()[0][0]
        rec_normalized = rec_with_similarity.withColumn(
            "norm_rating",
            col("rating") / (max_rating if max_rating != 0 else 1.0)
        )

        rec_with_score = rec_normalized.withColumn(
            "final_score",
            0.7 * col("norm_rating") + 0.3 * col("content_similarity")
        ).cache()
            
        # print("rec_with_score 1st", rec_with_score.first().asDict())
        # print("self.tags_df 1st", self.tags_df.first().asDict())
        try:
            # # 排序 + 去重，确保保留每个 user/movie/rating 组合的最佳推荐项
            rec_with_score = rec_with_score.orderBy(col("final_score").desc())
            rec_with_score = rec_with_score.dropDuplicates(["userId", "movieId", "rating"])

            final_recs = rec_with_score.select(
                col("userId"),
                col("movieId"),
                col("rating"),
                col("content_similarity").alias("similarity"),   
                col("final_score"),
            )
            # print("final_recs:", final_combined.count())  
            return final_recs, rec_with_score
        
        except Exception as e:
            print("combine data error:", e)
    

    def export_result(self, output_path, final_recs, overwrite=True):
        
        if not overwrite and (
                os.path.exists(output_path) or 
                os.path.exists(f"{output_path}.csv")
            ):
            print(f"already existed：{output_path}")
            return

        final_recs.coalesce(1).write.option("header", True).mode("overwrite").csv(output_path)
        save_to_csv(output_path)
        print(f"save succeed：{output_path}")

    
    def weighted_recommendations(self, user_id, best_als_model, tags_df_transformed, hashing_tf):
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

            # Step 1: Get ALS reçcommendations
            user_rec_movies = self.get_ALS_recommendations(best_als_model, user_id)
            # self.check_dup(user_rec_movies)


            # Step 2: Build user preference profile
            user_tag_features, avg_vector_broadcast = self.get_user_profile(user_id, tags_df_transformed, hashing_tf)
            # self.check_dup(user_tag_features)

            # Step 3: Calculate content similarity scores
            rec_with_similarity = self.get_content_similarity(user_rec_movies, tags_df_transformed, avg_vector_broadcast)
            

            # Step 4: Hybrid scoring
            final_recs, rec_with_score = self.get_final_combined(rec_with_similarity, user_id)
            # self.check_dup(rec_with_similarity)

            # save
            output_path = f"{self.result_path}/user_{user_id}_recommendation"
            self.export_result(output_path, final_recs)

            # 提取 Top-K 推荐的 movieId（推荐列表已经去重和排序）
            predicted_movie_ids = final_recs.orderBy(col("final_score").desc()) \
                                            .select("movieId") \
                                            .limit(10) \
                                            .rdd.flatMap(lambda x: x).collect()

            # 计算 Precision@10（验证用 self.test_df）
            precision = self.precision_at_k_for_user(user_id, predicted_movie_ids, self.test_df, k=10)

            # # Clean cached data
            rec_with_score.unpersist()
            if user_tag_features:
                user_tag_features.unpersist()

            return tags_df_transformed, precision

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
            col("movieId"),
            col("features")
        ).rdd.map(
            lambda row: (row["movieId"], row["features"].toArray())
        )
        movie_vector_list = movie_vectors.collect()

        # Calculate pairwise cosine similarity (limit to first 500 movies)
        result = []
        for (id1, vec1), (id2, vec2) in combinations(movie_vector_list[:500], 2):
            sim = cosine_similarity_udf(vec1, vec2)
            if sim > 0.6:
                result.append((id1, id2, sim))

        schema = StructType([
            StructField("sourceMovie", IntegerType()),
            StructField("targetMovie", IntegerType()),
            StructField("similarity", DoubleType())
        ])
        sim_df = self.spark.createDataFrame(result, schema=schema)

        output_path = f"{self.result_path}/movie_similarity_network"
        self.export_result(output_path, sim_df)


    def export_wordcloud_fields(self):
        """
        Export text fields for word cloud visualization.

        Args:
            tags_df: DataFrame containing movie tags
            movie_df: DataFrame containing movie metadata
        """
        # Correct column references
        tags_enriched = self.tags_df.withColumn(
            "wordcloud_text",
            concat_ws(" ", "tag", "genres")
        )
     
        output_path = f"{self.result_path}/wordcloud_text"
        self.export_result(output_path, tags_enriched)


    def generate_recommendations(self, best_model):
        user_ids = self.test_df.select("userId").distinct().rdd.map(
            lambda row: row["userId"]
        ).collect()
        print(f"Generating recommendations for {len(user_ids)} users...")

        tags_df_transformed = None
  
        # step 0
        tags_df_transformed, hashing_tf = self.get_content_feature_matrix()
        # self.check_dup(tags_df_transformed)

        total_precision = 0.0
        valid_user_count = 0
        k = 10

        for user_id in user_ids[100:120]:
            try:
                tags_df_transformed, precision = self.weighted_recommendations(user_id, best_model, tags_df_transformed, hashing_tf)
                total_precision += precision
                valid_user_count += 1
            except Exception as e:
                print(f"Recommendation failed for user {user_id}, skipping: {str(e)}")
                continue
        
        if valid_user_count > 0:
            avg_precision = total_precision / valid_user_count
            print(f"\n==== Average Precision@{k} over {valid_user_count} users: {avg_precision:.4f} ====")

        if tags_df_transformed:
            self.export_movie_similarity_network(tags_df_transformed)
            self.export_wordcloud_fields()
        else:
            print("No successful recommendations generated, skipping exports")


    @log_caller
    def check_dup(self, df):
        print("=== Duplicate rows (if any) ===")
        df_count = df.count()
        dup_df = df.dropDuplicates()
        dup_count = dup_df.count()
  
        print(f"Total rows: {df.count()}, After dropDuplicates(): {dup_count}")

        if df_count > dup_count:
            print(f"Found {df_count - dup_count} duplicated rows. Showing top duplicates:")
            # dup_df.orderBy("count", ascending=False).show(truncate=False)
            dup_df.show(n=5)
        else:
            print(f"No duplicated rows found.")

    def evaluate_top_k(self, model, test_df, top_k=10):
        print(f"Evaluating Precision@{top_k}...")

        # 为所有用户生成 Top-K 推荐
        user_recs = model.recommendForAllUsers(top_k)

        # 展开推荐列表：每个用户一行 → 多行，每行一个推荐的 movieId
        exploded = user_recs.selectExpr("userId", "explode(recommendations) as rec") \
                            .selectExpr("userId", "rec.movieId as movieId", "rec.rating as prediction")

        # 认为 test 集中评分 >= 3 的是喜欢的项目（可改阈值）
        relevant = test_df.filter(test_df["rating"] >= 0.0)

        # 命中的推荐（推荐中也在 test 中并被喜欢）
        joined = exploded.join(relevant, on=["userId", "movieId"], how="inner")

        # 每个用户命中的数量
        hit_counts = joined.groupBy("userId").agg(F.count("movieId").alias("num_hits"))

        # 每个用户被推荐的总数（一般是 top_k）
        rec_counts = exploded.groupBy("userId").agg(F.count("movieId").alias("num_recs"))

        # 计算 precision = 命中 / 推荐
        metrics_df = hit_counts.join(rec_counts, on="userId", how="inner") \
                            .withColumn("precision_at_k", F.col("num_hits") / F.col("num_recs"))

        # 求平均 precision
        avg_precision = metrics_df.select(F.mean("precision_at_k")).first()[0]

        print(f"Average Precision@{top_k}: {avg_precision:.4f}")
        return avg_precision

    def precision_at_k_for_user(self, user_id, predicted_movie_ids, test_df, k=10):
        from pyspark.sql import Row
        from pyspark.sql import DataFrame

        print(f"\n==== User {user_id} ====")

        # 推荐的 Top-K
        print(f"[Top {k} predicted movieIds]: {predicted_movie_ids}")

        # 用户 test data 中真实评分记录（只取前几行打印）
        user_test_df = test_df.filter((col("userId") == user_id))
        print("\n[Test data for this user (first few rows)]:")
        user_test_df.show(5)

        # 获取用户 test 中喜欢的电影（评分 ≥ 3.0）
        relevant_df = user_test_df.filter(col("rating") >= 3.0)
        relevant_movie_ids = relevant_df.select("movieId").rdd.flatMap(lambda x: x).collect()

        # 命中的 movieId（推荐的 ∩ 喜欢的）
        hit_movie_ids = list(set(predicted_movie_ids) & set(relevant_movie_ids))

        # 打印命中的并评分合格的电影
        print(f"\n[Hit relevant movies among Top {k}]: {hit_movie_ids}")
        if hit_movie_ids:
            print("\n[Matching entries in test data]:")
            relevant_df.filter(col("movieId").isin(hit_movie_ids)).show()

        # Precision = 命中数 / k
        precision = len(hit_movie_ids) / k
        print(f"[Precision@{k} for user {user_id}]: {precision:.4f}")
        return precision


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

