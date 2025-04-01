#!/usr/bin/env python
# @Auther liukun
# @Time 2025/03/26

import os
from pathlib import Path
from dotenv import load_dotenv


# for local
def conn_local():
    # env
    load_dotenv()
    ROOT_DIR = os.getenv('ROOT_DIR')
    data_path = Path(ROOT_DIR) / "data/test/movies.csv"
    print(ROOT_DIR, data_path)
    # from pyspark.sql import SparkSession
    # spark = SparkSession.builder.appName("Test").getOrCreate()
    # print(spark.version)

# for google colab
def conn_colab():
    from google.colab import drive
    drive.mount('/content/drive')
    # !ls -1 /content/drive/MyDrive/5003-BigData/data

# for databrick
def conn_databrick():
    file_path = "/FileStore/tables/test"
    scrpit_path = "/Workspace/Shared/flick-pick"
    dbutils.fs.ls(file_path)
    df = spark.read.csv(file_path, header=True)
    df.show()
    # %run /Workspace/Shared/flick-pick/ALS_test


if __name__ == "__main__":

    conn_local()