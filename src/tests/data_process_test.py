#!/usr/bin/env python
# @Auther liukun
# @Time 2025/04/05

import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# env
load_dotenv()
file_path = Path(os.getenv('ROOT_DIR')) / "data/processed/for_visual_sample.csv"
moive_path = Path(os.getenv('ROOT_DIR')) / "data/test/movies.csv"

pd.set_option('display.max_columns', None)


def extra_movies(df):
    df['year'] = df['title'].str.extract(r'\((\d{4})\)')
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    # print(df.head(10))
    return df.reset_index(drop=True)


def user_cf_recommend(df):
    # observe rating range
    # rating = sorted(df['rating'].dropna().unique().tolist())
    # print(rating)
    # print(df['rating'].describe())

    # rating bin
    bins = [2.0, 4.0, 5.0, 6.0, 7.0, 8.2]    
    labels = ['2-4', '4-5', '5-6', '6-7', '7+']
    df['rating_bin'] = pd.cut(df['rating'], bins=bins, labels=labels, include_lowest=True)

    # merge movie info
    movies = extra_movies(pd.read_csv(moive_path))
    df = df.merge(movies, on='movieId', how='left')

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(file_path, index_col=0)
    df = user_cf_recommend(df)
    print(df.head(10))

