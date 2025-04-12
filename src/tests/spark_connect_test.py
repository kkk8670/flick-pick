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


# for cloudflare
def conn_R2(folder=""):
    import boto3
    load_dotenv()

    ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
    SECRET_KEY = os.getenv("R2_SECRET_KEY")
    R2_ENDPOINT = os.getenv("R2_ENDPOINT")
    BUCKET_NAME="movie-dataset"

    session = boto3.session.Session()
    s3 = session.client(
        service_name="s3",
        region_name="auto",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder)
    if "Contents" in response:
        for obj in response["Contents"]:
            print(obj["Key"])
    else:
        print("No files found.")


if __name__ == "__main__":

    # conn_local()
    conn_R2("test/")