import os, sys
from dotenv import load_dotenv
import pandas as pd 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat, lit, regexp_extract, regexp_replace

load_dotenv()
project_path = os.getenv("PYTHONPATH")
if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)
from utils import save_to_csv 




class DataPreprocess:
    def __init__(self):
        root_dir = os.getenv('ROOT_DIR')
        self.data_path = f"{root_dir}/data"
        self.raw_set = "raw/ml-latest-small"
        self.spark = (
                    SparkSession.builder  
                        .appName("flick-pick")  
                        .getOrCreate()
                )
        self.df = {}

    def load_df(self, name):
        if name in self.df:
            return self.df[name]

        file_path = f"{self.data_path}/{self.raw_set}/{name}.csv" 
        self.df[name] = (
            self.spark.read.option("header", "true").option("inferSchema", "true")
                .csv(file_path)
            )
        return self.df[name]

 
    def save_df(self, df, save_file, overwrite):
        if not overwrite and (
                os.path.exists(save_file) or 
                os.path.exists(f"{save_file}.csv")
            ):
            print(f"already existed：{save_file}")
            return

        df.coalesce(1).write.option("header", True).mode("overwrite").csv(save_file)
        save_to_csv(save_file)
        print(f"save succeed：{save_file}")


    def check_joined_null_timestamp(self, joined_df):
        # check 统计为空的 
        labeled_df = joined_df.withColumn(
            "null_status",
            when(col("rating").isNull() & col("tag").isNull(), "both_null")
            .when(col("rating").isNull(), "rating_null")
            .when(col("tag").isNull(), "tag_null")
            .otherwise("none_null")
        )

        labeled_df.groupBy("null_status").count().show()


    def merge_other_data(self, overwrite=False):
        """
        合并 ratings 和 tags 的 timestamp，补全缺失并生成可读日期列：
        - rating_date：基于 ratingTimestamp
        - tag_date：基于 tagTimestamp
        """
        ratings_df, tags_df = self.load_df("ratings"), self.load_df("tags")

        save_file = f"{self.data_path}/processed/user_movie_timestamp"

        # 统一重命名时间戳列，做 full outer join
        joined_df = (
            ratings_df.select("userId", "movieId", "timestamp").withColumnRenamed("timestamp", "ratingTimestamp")
            .join(
                tags_df.select("userId", "movieId", "timestamp").withColumnRenamed("timestamp", "tagTimestamp"),
                on=["userId", "movieId"],
                how="full"
            )
        )

        # 缺值互补：任一为空则用另一个值补上，全部为空设为 0
        joined_df = (
            joined_df
            .withColumn(
                "ratingTimestamp",
                when(col("ratingTimestamp").isNull(), col("tagTimestamp")).otherwise(col("ratingTimestamp"))
            )
            .withColumn(
                "tagTimestamp",
                when(col("tagTimestamp").isNull(), col("ratingTimestamp")).otherwise(col("tagTimestamp"))
            )
            .fillna({"ratingTimestamp": 0, "tagTimestamp": 0})
        )

        # 新增两个可读格式的日期列
        joined_df = joined_df \
            .withColumn("rating_date", from_unixtime(col("ratingTimestamp"), "yyyy-MM-dd")) \
            .withColumn("tag_date", from_unixtime(col("tagTimestamp"), "yyyy-MM-dd"))

        # 保存
        self.save_df(joined_df, save_file, overwrite)

 
    def get_rating_data(self, overwrite):
        """
        这个数据集用于 需要user和movie一起 & train 中 als评分列，只有movieid，userid和rating
        """
        
        save_file = f"{self.data_path}/processed/user_movie_rating"
     
        ratings_df = self.df["ratings"].select("userId", "movieId", "rating")
       
        # save
        self.save_df(ratings_df, save_file, overwrite)


    def check_joined_null_tag(self, joined_df):
        null_tag_count = joined_df.filter(col("tag").isNull()).count()
        print(f"Number of null tags: {null_tag_count}")
        joined_df.filter(col("tag").isNull()).show()


    def get_movie_train_tags(self, overwrite=False):
        """
        构建 movie + tag 的 tf-idf 特征列，格式统一，一行一电影，输出列为：
        movieId, title, genres, tag, year
        """
        movies_df, tags_df = self.load_df("movies"), self.load_df("tags")
        save_file = f"{self.data_path}/processed/movie_train_tags"

        tags_df = tags_df.fillna({"tag": ""})

        tags_df_grouped = (
            tags_df
            .groupBy("movieId")
            .agg(collect_list("tag").alias("tag_list"))
        )

        def clean_tags(tag_list):
            tags = set()
            for t in tag_list:
                if t:
                    for item in t.lower().split("|"):
                        item = item.strip().replace(" ", "_")
                        if item:
                            tags.add(f"tag_{item}")
            return " ".join(sorted(tags))

        clean_tags_udf = udf(clean_tags, StringType())
        tags_df_grouped = tags_df_grouped.withColumn("tag_clean", clean_tags_udf("tag_list"))

        def clean_genres(genres_str):
            if not genres_str:
                return ""
            genres = genres_str.lower().split("|")
            genres = set(g.strip().replace(" ", "_") for g in genres if g.strip())
            return " ".join(sorted(f"genre_{g}" for g in genres))

        clean_genres_udf = udf(clean_genres, StringType())
        movies_df = (
            movies_df
            .withColumn("genres_clean", clean_genres_udf("genres"))
            .withColumn("year", regexp_extract("title", r"\((\d{4})\)", 1))
            .withColumn("title", regexp_replace("title", r"\s*\(\d{4}\)", ""))
        )

        joined_df = (
            movies_df
            .join(tags_df_grouped.select("movieId", "tag_clean"), on="movieId", how="left")
            .fillna({"tag_clean": ""})
        )

        # 对列重命名
        joined_df = joined_df.select(
            "movieId",
            "title",
            col("genres_clean").alias("genres"),
            col("tag_clean").alias("tag"),
            "year"
        )

        self.save_df(joined_df, save_file, overwrite)


    def get_movie_link(self, overwrite=False):
        """
        此函数用于 movie信息 & 非训练其他信息，目前只有movielink列：
        """
        save_file = f"{self.data_path}/processed/movie_links"

        movies_df = self.load_df("movies").select("movieId")
        movies_df = movies_df.withColumn('movieLink',concat(lit('https://movielens.org/movies/'), col('movieId').cast('string')))

        # save
        self.save_df(movies_df, save_file, overwrite)




def main():
    data_pre = DataPreprocess()
    # data_pre.merge_other_data(overwrite=False)
    # data_pre.get_rating_data(overwrite=False)
    # data_pre.get_movie_train_tags(overwrite=False)
    data_pre.get_movie_link(overwrite=False)

if __name__ == "__main__":
    main()
