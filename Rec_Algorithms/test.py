import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.sql.functions import explode, col, desc
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
import numpy as np




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
    # 加载 ratings 数据（rating.csv）
    ratings_df = (
        spark.read.csv("../../data-smallest/ratings.csv", header=True, inferSchema=True)
        .select("userId", "movieId", "rating")
        .withColumn("userId", col("userId").cast("int"))
        .withColumn("movieId", col("movieId").cast("int"))
        .withColumn("rating", col("rating").cast("float"))
    )

    # 加载 ratingMovie 数据（ratingMovie.csv），这包含了电影的 title 和 genres 信息
    movie_df = (
        spark.read.csv("../../data-smallest/ratingMovie.csv", header=True, inferSchema=True)
        .select("movieId", "title", "genres")
    )

    # 加载 tags 数据（tags.csv）
    tags_df = (
        spark.read.csv("../../data-smallest/tags.csv", header=True, inferSchema=True)
        .select("movieId", "tags_str")
    )

    return ratings_df, movie_df, tags_df



def build_base_als():
    """Define the base ALS model."""
    return ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
    )


# def hyperparameter_optimization(ratings_df):
#     """Perform hyperparameter tuning for the ALS model."""
#     als = build_base_als()
#     param_grid = (
#         ParamGridBuilder()
#         .addGrid(als.rank, [10, 20, 50])
#         .addGrid(als.maxIter, [10, 20])
#         .addGrid(als.regParam, [0.01, 0.1, 1.0])
#         .build()
#     )
#     evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
#     crossval = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
#     cv_model = crossval.fit(ratings_df)
#     return cv_model.bestModel

# test version
def hyperparameter_optimization(ratings_df):
    """Use default ALS model for quick testing."""
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        rank=10,          # 默认值
        maxIter=10,       # 默认值
        regParam=0.1      # 默认值
    )
    model = als.fit(ratings_df)
    return model

def cosine_similarity_udf(v1, v2):
    v1 = np.array(v1)  # 将 v1 转换为 numpy 数组
    v2 = np.array(v2)  # 将 v2 转换为 numpy 数组
    norm_v1 = np.linalg.norm(v1)  # 计算 v1 的范数
    norm_v2 = np.linalg.norm(v2)  # 计算 v2 的范数
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2) if norm_v1 != 0 and norm_v2 != 0 else 0.0)

def weighted_recommendations(user_id, best_als_model, movie_df, tags_df, ratings_df):
    """ALS + 内容相似度融合排序推荐"""
    # 🎯 Step 1: ALS 推荐前 100 个（多取一点）
    user_recommendations = best_als_model.recommendForUserSubset(
        spark.createDataFrame([(user_id,)], ["userId"]), 100
    )
    user_rec_movies = user_recommendations.withColumn("rec", explode("recommendations")) \
        .select("userId", "rec.movieId", "rec.rating")

    # 🧠 Step 2: 构建 tags 的 TF-IDF 特征
    tokenizer = Tokenizer(inputCol="tags_str", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
    tag_model = pipeline.fit(tags_df)
    tags_df_transformed = tag_model.transform(tags_df)

    # 🎯 Step 3: 找出用户评分过的电影，取其内容特征的平均向量作为"用户兴趣画像"
    user_history = ratings_df.filter(col("userId") == user_id)
    user_tag_features = user_history.join(tags_df_transformed, "movieId").select("features")

    # 缓存 user_tag_features 数据集
    user_tag_features.cache()

    avg_vector = user_tag_features.rdd.map(lambda row: row["features"].toArray()).mean()
    avg_vector_broadcast = spark.sparkContext.broadcast(avg_vector)

    # 🧠 Step 4: 对 ALS 推荐列表的每部电影内容与"用户兴趣画像"做余弦相似度计算
    rec_with_features = user_rec_movies.join(tags_df_transformed, "movieId")

    # 添加余弦相似度分数 - Moved this after avg_vector_broadcast is defined
    similarity_udf = udf(lambda v: cosine_similarity_udf(avg_vector_broadcast.value, v), DoubleType())
    rec_with_similarity = rec_with_features.withColumn("similarity", similarity_udf(col("features")))

    # 🎯 Step 5: 加权评分
    rec_with_score = rec_with_similarity.withColumn(
        "final_score", 0.7 * col("rating") + 0.3 * col("similarity")
    )

    # 🎬 Step 6: 拼接电影标题信息
    final_result = rec_with_score \
        .join(movie_df, "movieId", "left") \
        .select("movieId", "title", "rating", "similarity", "final_score") \
        .orderBy(desc("final_score"))

    print(f"\n🎯 Top Recommendations for User {user_id}:\n")
    final_result.show(truncate=False)


def main():
    """Main function to load data, train models, and save the best model."""
    # 加载数据
    ratings_df, movie_df, tags_df = load_data()  # 删除了 links_df，因为不再需要它

    # 划分数据集（80% 训练，20% 测试）
    train_data, _ = ratings_df.randomSplit([0.8, 0.2], seed=42)

    # 设置用户 ID 进行推荐测试
    user_id = 1  # 例如，使用用户 ID = 1

    # 获取最佳模型（ALS模型）
    best_model = hyperparameter_optimization(train_data)

    # 获取推荐结果并打印
    weighted_recommendations(user_id, best_model, movie_df, tags_df, ratings_df)

    # Windows 下暂不保存模型，避免 Hadoop 路径/权限问题
    # save_path = "E:/prepare/se_rec/model"
    # os.makedirs(save_path, exist_ok=True)
    # best_model.write().overwrite().save(save_path)

    # 结束 Spark 会话
    spark.stop()


if __name__ == "__main__":
    main()
