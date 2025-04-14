#!/usr/bin/env python
# @Auther liukun
# @Time 2025/04/12

import os
import boto3
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO
import pandas as pd

class R2Client:
    def __init__(self):
        load_dotenv()
        ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
        SECRET_KEY = os.getenv("R2_SECRET_KEY")
        R2_ENDPOINT = os.getenv("R2_ENDPOINT")
        self.bucket_name = "movie-dataset"     

        session = boto3.session.Session()
        self.client = session.client(
            service_name="s3",
            region_name="auto",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
        )

    def file_exists(self, file):
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=file)
            return True
        except self.client.exceptions.ClientError:
            return False


    def list_folder(self, prefix="", recursive=False):
        """list all data structure"""
        print(f"Listing datasets under: '{prefix}'")

        paginator = self.client.get_paginator("list_objects_v2")
        operation_parameters = {
            "Bucket": self.bucket_name,
            "Prefix": prefix,
        }
        if not recursive:
            operation_parameters["Delimiter"] = "/"

        result = paginator.paginate(**operation_parameters)

        for page in result:
            if not recursive and "CommonPrefixes" in page:
                print("Subfolders:")
                for sub in page["CommonPrefixes"]:
                    print(" -", sub["Prefix"])

            if "Contents" in page:
                print("Files:")
                for obj in page["Contents"]:
                    if obj["Key"] != prefix:
                        print(" -", obj["Key"])

    def get_file(self, file_path):
        """get .csv from R2 """
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=file_path)
            df = pd.read_csv(response["Body"])
            return df
        except Exception as e:
            print("Error downloading file:", e)
            return None


    def upload_local_file(self, local_path, remote_path, overwrite=False):
        """update local file to R2"""
        try:
            if self.file_exists(remote_path) and not overwrite:
                print(f"'{remote_path}' already exists. Skipping upload.\n")
                return

            with open(local_path, "rb") as f:
                self.client.upload_fileobj(f, self.bucket_name, remote_path)
            print(f"Uploaded '{local_path}' to '{remote_path}'")

        except Exception as e:
            print("Upload failed:", e) 


    def upload_df_file(self, df, remote_path, overwrite=True):
        """update DataFrame file to R2"""
        try:
            if self.file_exists(remote_path):
                if not overwrite:
                    print(f"Remote file '{remote_path}' already exists and SKIPPING upload.\n")
                    return
                else:
                    print(f"Remote file '{remote_path}' already exists. Will OVERWRITE old version.\n")
            csv_buffer = BytesIO()
            df.to_csv(csv_buffer)
            csv_buffer.seek(0)
            self.client.upload_fileobj(csv_buffer, self.bucket_name, remote_path)
            print(f"Uploaded DataFrame as CSV to '{remote_path}'\n")

        except Exception as e:
            print("Upload DataFrame failed:", e)


    def upload_folder(self, local_folder, remote_path=""):
        """upload local folder to R2"""
        local_folder = os.path.abspath(local_folder)
        print(f"Uploading folder '{local_folder}' to remote prefix '{remote_path}'...\n")

        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                s3_key = os.path.join(remote_path, relative_path).replace("\\", "/")  # Windows compatibility

                try:
                    with open(local_path, "rb") as f:
                        self.client.upload_fileobj(f, self.bucket_name, s3_key)
                    print(f"Uploaded {relative_path} -> {s3_key}")
                except Exception as e:
                    print(f"Failed to upload {relative_path}:", e)

    def delete_file(self, remote_path):
        if not self.file_exists(remote_path):
            print(f"Remote file '{remote_path}' does not exist. Cannot delete.")
            return

        self.client.delete_object(Bucket=self.bucket_name, Key=remote_path)
        print(f"🗑️ Deleted remote file '{remote_path}' successfully.\n")

    def delete_folder(self, prefix):
        keys = self.list_folder(prefix)
        if not keys:
            print(f"📁 Folder '{prefix}' is empty or doesn't exist.")
            return

        for key in keys:
            delete_file(key)

        print(f"✅ Deleted all files under folder '{prefix}'")

if __name__ == "__main__":
    r2 = R2Client()
    ROOT_DIR = Path(os.getenv('ROOT_DIR'))

    # upload folder test
    # local_folder = str(ROOT_DIR / 'data/raw')
    # remote_folder = "raw"
    # r2.upload_folder(local_folder, remote_folder)

    # upload local file test
    # file_path = 'processed/for_visual_sample.csv'
    # local_file = '/'.join([str(ROOT_DIR), "data", file_path])
    # r2.upload_local_file(local_file, file_path)

    # upload pd test
    # df = pd.read_csv(local_file, index_col=0)
    # print(df.head())
    # r2.upload_df_file(df, file_path)

    # delete test
    # file_path = 'result/user_471_recommendation_and_history.csv'
    # r2.delete_file(file_path)

    # list structure test
    r2_path =  "" # "test/"
    r2.list_folder(r2_path, 1)   

    # get file test 
    # r2_file = '/'.join([r2_path, "ratings.csv"])
    # print(r2_file)
    # df = r2.get_file(r2_file)
    # print(df.head())