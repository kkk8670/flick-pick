#!/usr/bin/env python
# @Auther liukun, liyue, jinyujian

import os, sys
from dotenv import load_dotenv
import pandas as pd 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat, concat_ws, lit, regexp_extract, regexp_replace, from_unixtime, collect_list, explode, split, lower, trim, sort_array, collect_set, array, transform, array_distinct, length

load_dotenv()
project_path = os.getenv("PYTHONPATH")
if project_path and project_path not in sys.path:
    sys.path.insert(0, project_path)
from utils import save_to_csv 


class DataPreprocess:
    def __init__(self):
        ROOT_DIR = os.getenv('ROOT_DIR')
        self.data_path = f"{ROOT_DIR}/data"
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
        # check if cell is null
        labeled_df = joined_df.withColumn(
            "null_status",
            when(col("rating").isNull() & col("tag").isNull(), "both_null")
            .when(col("rating").isNull(), "rating_null")
            .when(col("tag").isNull(), "tag_null")
            .otherwise("none_null")
        )

        labeled_df.groupBy("null_status").count().show()


    def get_user_movie_info(self, overwrite=False):
        """
        merge ratings & tags timestamp
        """
        ratings_df, tags_df = self.load_df("ratings"), self.load_df("tags")

        save_file = f"{self.data_path}/output/user_movie_info"

        # join timestamp with full outer join
        joined_df = (
            ratings_df.select("userId", "movieId", "timestamp").withColumnRenamed("timestamp", "ratingTimestamp")
            .join(
                tags_df.select("userId", "movieId", "timestamp").withColumnRenamed("timestamp", "tagTimestamp"),
                on=["userId", "movieId"],
                how="full"
            )
        )

        # Missing value complementary strategy: if any of the values is null, use another value to fill in, set 0 for all nulls.
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

        # stamp -> date
        joined_df = joined_df \
            .withColumn("rating_date", from_unixtime(col("ratingTimestamp"), "yyyy-MM-dd")) \
            .withColumn("tag_date", from_unixtime(col("tagTimestamp"), "yyyy-MM-dd"))

        # save
        self.save_df(joined_df, save_file, overwrite)

 
    def get_rating_data(self, overwrite):
        """
        for primany key = userid+movieid   
        als with cols : movieid，userid and rating
        """
        
        save_file = f"{self.data_path}/processed/user_movie_rating"
     
        ratings_df = self.load_df("ratings").select("userId", "movieId", "rating")
       
        # save
        self.save_df(ratings_df, save_file, overwrite)


    def check_joined_null_tag(self, joined_df):
        null_tag_count = joined_df.filter(col("tag").isNull()).count()
        print(f"Number of null tags: {null_tag_count}")
        joined_df.filter(col("tag").isNull()).show()


    def get_movie_features(self, overwrite=False):
        """
        for tf-idf feature of movie，with cols：
        movieId, title, genres, tag, year
        """
        movies_df, tags_df = self.load_df("movies"), self.load_df("tags")
        save_tags_file = f"{self.data_path}/processed/movie_train_tags"
        save_info_file = f"{self.data_path}/output/movie_infos"

        movies_df = (
            movies_df
            .withColumn("year", regexp_extract("title", r"\((\d{4})\)", 1))
            .withColumn("title", regexp_replace("title", r"\s*\(\d{4}\)", ""))
        )
        # Normalize movie titles：Move (The, An, A) from the end to the beginning
        movies_df = movies_df.withColumn(
            "title",
            when(
                col("title").rlike(".*,\\s(The|An|A)$"),
                concat(
                    regexp_extract("title", r"^(.*),\s(The|An|A)$", 2),  # The/An/A
                    lit(" "),
                    regexp_extract("title", r"^(.*),\s(The|An|A)$", 1)  # Main title
                )
            ).otherwise(col("title"))
        )

        movies_info = movies_df.withColumn('link',concat(lit('https://movielens.org/movies/'), col('movieId').cast('string')))

        movies_info = movies_info.select(
            "movieId",
            "title",
            "year",
            'link'
        )

        
        self.save_df(movies_info, save_info_file, overwrite)

        movies_df = (movies_df
            .withColumn("genres_array", 
                        when(col("genres").isNull() | (col("genres") == ""), array())
                        .otherwise(split(lower("genres"), "\\|")))
            .withColumn("genres_trimmed", 
                        transform("genres_array", lambda x: trim(x)))
            # .withColumn("genres_replaced", 
            #             transform("genres_trimmed", lambda x: regexp_replace(x, " ", "_")))
            # .withColumn("genres_filtered", 
            #             array_distinct(transform("genres_replaced", lambda x: when(length(x) > 0, concat(lit("genre_"), x)))))
            .withColumn("genres_clean", 
                        concat_ws(" ", sort_array("genres_trimmed")))
            .drop("genres_array", "genres_trimmed", "genres_replaced", "genres_filtered")
        )


        tags_df_grouped = tags_df.fillna({"tag": ""}).groupBy("movieId").agg(collect_list("tag").alias("tag_list"))

        tags_processed = (tags_df_grouped
            .select("movieId", explode("tag_list").alias("tag"))
            .filter("tag IS NOT NULL AND tag != ''")
            .select("movieId", explode(split(lower("tag"), "\\|")).alias("tag_item"))
            .select("movieId", trim("tag_item").alias("tag_item"))
            # .select("movieId", regexp_replace("tag_item", " ", "_").alias("tag_item"))
            .filter("tag_item != ''")
            # .select("movieId", concat(lit("tag_"), col("tag_item")).alias("tag_item"))
            .groupBy("movieId")
            .agg(sort_array(collect_set("tag_item")).alias("sorted_tags"))
            .select("movieId", concat_ws(" ", "sorted_tags").alias("tag_clean"))
        )

        tags_df_final = tags_df_grouped.join(tags_processed, "movieId", "left")



        joined_df = (
            movies_df
            .join(tags_df_final.select("movieId", "tag_clean"), on="movieId", how="left")
            .fillna({"tag_clean": ""})
        )
 
        joined_df = joined_df.select(
            "movieId",
            "title",
            col("genres_clean").alias("genres"),
            col("tag_clean").alias("tag"),
            "year"
        )

        self.save_df(joined_df, save_tags_file, overwrite)


def main():
    data_pre = DataPreprocess()
    # data_pre.get_user_movie_info(overwrite=1)
    # data_pre.get_rating_data(overwrite=1)
    data_pre.get_movie_features(overwrite=1)

if __name__ == "__main__":
    main()
