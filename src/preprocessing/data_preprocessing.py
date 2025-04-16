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


    def merge_other_data(overwrite=False):
        """
        这个数据集用于 非训练&需要user和movie一起定义 的列，目前只有timestamp
        """
        ratings_df, tags_df = self.load_df("ratings"), self.load_df("tags")

        save_file = f"{self.data_path}/processed/user_movie_timestamp"

        # ts的策略是Full Outer Join，就地（不产生新列）只缺失一列的值填对方，都缺失的填充0
        joined_df = (
                        ratings_df.select("userId", "movieId", "ratingTimestamp")
                            .join(tags_df.select("userId", "movieId", "tagTimestamp"), 
                                on=["userId", "movieId"], 
                                how="full"
                            )
                    )

        # check_joined_null_timestamp(joined_df)

        joined_df = (
                        joined_df 
                        .withColumn(
                            "ratingTimestamp",
                            when(col("ratingTimestamp").isNull(), col("tagTimestamp")) 
                            .otherwise(col("ratingTimestamp"))
                        ) 
                        .withColumn(
                            "tagTimestamp",
                            when(col("tagTimestamp").isNull(), col("ratingTimestamp")) 
                            .otherwise(col("tagTimestamp"))
                        ) 
                        .fillna({"ratingTimestamp": 0, "tagTimestamp": 0})
                    )

        # df_filled.show(n=5)

        # save
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
        此函数用于 movie信息 & train 中 tf-id内容特征列，tags有：
        """
        movies_df, tags_df = self.load_df("movies"), self.load_df("tags")
        save_file = f"{self.data_path}/processed/movie_train_tags"

        # tags的策略是依据movieid拼接，缺失tags的填""
        joined_df = (
                        movies_df
                            .join(tags_df.select("movieId", "tag"), 
                                on=["movieId"], 
                                how="left"
                            )
                    )

        # check_joined_null_tag(joined_df)

        joined_df = joined_df.fillna({"tag": ""})

        joined_df = (   
                        joined_df
                            .withColumn("year", regexp_extract("title", r"\((\d{4})\)", 1))
                            .withColumn("title", regexp_replace("title", r"\s*\(\d{4}\)", ""))
                    )
     
        # joined_df.show(n=5)

        # save
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
