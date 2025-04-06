import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import col, desc, explode, udf
from pyspark.sql.types import DoubleType


# 环境变量配置
def setup_environment():
    """配置并验证环境变量"""
    env_path = Path(__file__).parent.parent.parent / '.env'
    print(f"正在加载环境变量文件: {env_path}")

    if not env_path.exists():
        raise FileNotFoundError(f"未找到.env文件: {env_path}")

    load_dotenv(dotenv_path=env_path, override=True)

    required_vars = ["PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"必需的环境变量 {var} 未设置")

    print("环境变量加载成功:")
    print(f"PYSPARK_PYTHON={os.getenv('PYSPARK_PYTHON')}")
    print(f"PYSPARK_DRIVER_PYTHON={os.getenv('PYSPARK_DRIVER_PYTHON')}")

    os.environ.update({
        "PYSPARK_PYTHON": os.getenv("PYSPARK_PYTHON"),
        "PYSPARK_DRIVER_PYTHON": os.getenv("PYSPARK_DRIVER_PYTHON"),
        "PATH": f"{os.path.dirname(os.getenv('PYSPARK_PYTHON'))};"
                f"{os.environ['PATH']}"
    })


def init_spark():
    """初始化Spark会话"""
    return (
        SparkSession.builder
        .appName("MovieRecommendation")
        .config("spark.executorEnv.PYSPARK_PYTHON", os.getenv("PYSPARK_PYTHON"))
        .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON",
                os.getenv("PYSPARK_PYTHON"))
        .config("spark.python.profile", "false")
        .config("spark.hadoop.validateOutputSpecs", "false")
        .getOrCreate()
    )


def load_data(spark):
    """加载评分、电影和标签数据集"""
    ratings_df = (
        spark.read.csv(
            "../../data-smallest/ratings.csv",
            header=True,
            inferSchema=True
        )
        .select("userId", "movieId", "rating")
        .withColumn("userId", col("userId").cast("int"))
        .withColumn("movieId", col("movieId").cast("int"))
        .withColumn("rating", col("rating").cast("float"))
    )

    movie_df = (
        spark.read.csv(
            "../../data-smallest/ratingMovie.csv",
            header=True,
            inferSchema=True
        )
        .select("movieId", "title", "genres")
    )

    tags_df = (
        spark.read.csv(
            "../../data-smallest/tags.csv",
            header=True,
            inferSchema=True
        )
        .select("movieId", "tags_str")
    )

    return ratings_df, movie_df, tags_df


def build_base_als():
    """定义基础ALS模型"""
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
        rank=10,
        maxIter=10,
        regParam=0.1
    )
    model = als.fit(ratings_df)
    return model
def cosine_similarity_udf(v1, v2):
    """计算余弦相似度"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return float(
        np.dot(v1, v2) / (norm_v1 * norm_v2)
        if norm_v1 != 0 and norm_v2 != 0
        else 0.0
    )


def weighted_recommendations(user_id, best_als_model, movie_df, tags_df,
                            ratings_df, spark):
    """ALS + 内容相似度融合排序推荐"""
    # Step 1: ALS推荐前100个
    user_recs = best_als_model.recommendForUserSubset(
        spark.createDataFrame([(user_id,)], ["userId"]), 100
    )
    user_rec_movies = user_recs.withColumn(
        "rec", explode("recommendations")
    ).select("userId", "rec.movieId", "rec.rating")

    # Step 2: 构建TF-IDF特征
    tokenizer = Tokenizer(inputCol="tags_str", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures",
                          numFeatures=100)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
    tags_df_transformed = pipeline.fit(tags_df).transform(tags_df)

    # Step 3: 创建用户兴趣画像
    user_history = ratings_df.filter(col("userId") == user_id)
    user_tag_features = user_history.join(tags_df_transformed, "movieId")
    user_tag_features = user_tag_features.select("features").cache()

    avg_vector = user_tag_features.rdd.map(
        lambda row: row["features"].toArray()
    ).mean()
    avg_vector_broadcast = spark.sparkContext.broadcast(avg_vector)

    # Step 4: 计算余弦相似度
    rec_with_features = user_rec_movies.join(tags_df_transformed, "movieId")
    similarity_udf = udf(
        lambda v: cosine_similarity_udf(avg_vector_broadcast.value, v),
        DoubleType()
    )
    rec_with_similarity = rec_with_features.withColumn(
        "similarity", similarity_udf(col("features"))
    )

    # Step 5: 加权评分
    rec_with_score = rec_with_similarity.withColumn(
        "final_score", 0.7 * col("rating") + 0.3 * col("similarity")
    )

    # Step 6: 最终结果
    final_result = (
        rec_with_score
        .join(movie_df, "movieId", "left")
        .select("movieId", "title", "rating", "similarity", "final_score")
        .orderBy(desc("final_score"))
    )

    print(f"\n🎯 Top Recommendations for User {user_id}:\n")
    final_result.show(truncate=False)


def main():
    """主函数"""
    setup_environment()
    spark = init_spark()
    spark.sparkContext.setLogLevel("ERROR")

    ratings_df, movie_df, tags_df = load_data(spark)
    train_data, _ = ratings_df.randomSplit([0.8, 0.2], seed=42)
    best_model = hyperparameter_optimization(train_data)
    weighted_recommendations(1, best_model, movie_df, tags_df, ratings_df, spark)

    # Windows 下暂不保存模型，避免 Hadoop 路径/权限问题
    # save_path = "E:/prepare/se_rec/model"
    # os.makedirs(save_path, exist_ok=True)
    # best_model.write().overwrite().save(save_path)

    spark.stop()


if __name__ == "__main__":
    main()
