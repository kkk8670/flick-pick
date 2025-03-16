from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

# file_path = "dbfs:/FileStore/tables/test/ratings.csv"
file_path = "./data/ml-latest-small/ratings.csv"


# init
spark = SparkSession.builder.appName("flick-pick").getOrCreate()

 
ratings = spark.read.csv(file_path, header=True, inferSchema=True)

 
ratings = ratings.select(col("userId"), col("movieId"), col("rating"))

# training
als = ALS(
    rank=5,        # default rank=10
    maxIter=5,      # default 迭代次数=10   
    regParam=0.1,     # default 正则化参数=0.1   
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"  # 解决新用户/电影数据稀疏问题
)

# 拟合
model = als.fit(ratings)

# recommend for all users
recommendations = model.recommendForAllUsers(10) 

# recommend for userid=5
user_id = 6
# user_df = spark.createDataFrame([(user_id,)], ["userId"])
# recommendations = model.recommendForUserSubset(user_df, numItems=10)

recommendations = recommendations.filter(col("userId") == user_id) 
recommendations.show(5, truncate=False)  # show 前五条

spark.stop()