from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import os

# 设置环境变量
os.environ["SPARK_HOME"] = "C:\\Users\\Zelda\\AppData\\Roaming\\Python\\Python311\\site-packages\\pyspark"
os.environ["PYSPARK_PYTHON"] = "D:\\download_app\\anaconda\\envs\\start2\\python.exe"

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("MovieRecommendation") \
    .getOrCreate()
# 设置日志级别为 ERROR
spark.sparkContext.setLogLevel("ERROR")

# 加载 ratings.csv
ratings_df = spark.read.csv("rating.csv", header=False, inferSchema=True)
# 为列重命名
ratings_df = ratings_df.withColumnRenamed("_c0", "userId") \
                       .withColumnRenamed("_c1", "movieId") \
                       .withColumnRenamed("_c2", "rating")

# 加载 movie.csv
movie_df = spark.read.csv("movie.csv", header=True, inferSchema=True)

# 划分训练集和测试集
(train_data, test_data) = ratings_df.randomSplit([0.8, 0.2], seed=42)

# 初始化 ALS 模型
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=10,               # 隐语义模型的维度
    maxIter=10,            # 最大迭代次数
    regParam=0.1,          # 正则化参数
    coldStartStrategy="drop"  # 处理冷启动问题
)

# 训练模型
model = als.fit(train_data)

# 预测测试集
predictions = model.transform(test_data)
predictions.show(10)

# 评估模型
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 为用户生成推荐
user_id = 1  # 示例用户 ID
user_recommendations = model.recommendForUserSubset(
    spark.createDataFrame([(user_id,)], ["userId"]), 10
)

# 将推荐结果的电影 ID 替换为电影名称
# 需要将 user_recommendations 中的 recommendations 列展开
from pyspark.sql.functions import explode

# 将 recommendations 列展开
user_rec_movies = user_recommendations.withColumn("rec", explode("recommendations")) \
    .select("userId", "rec.movieId", "rec.rating")

# 关联 movie_df，获取电影名称
user_rec_with_titles = user_rec_movies.join(movie_df, "movieId", "inner") \
    .select("userId", "movieId", "title", "rating")

# 显示推荐结果
print(f"为用户 {user_id} 推荐的电影：")
user_rec_with_titles.show(truncate=False)

# 关闭 SparkSession
spark.stop()
