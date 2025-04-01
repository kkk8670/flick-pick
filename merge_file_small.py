import pandas as pd

# 1. Read four CSV files
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
links = pd.read_csv('links.csv')

# 2. Rename the timestamp field
ratings.rename(columns={'timestamp': 'timestamp_rating'}, inplace=True)
tags.rename(columns={'timestamp': 'timestamp_tags'}, inplace=True)

# 3. Merge ratings and tags (based on userId and movieId)
ratings_tags = pd.merge(ratings, tags, on=['userId', 'movieId'], how='left')

# 4. Add movies (based on movieId)
ratings_tags_movies = pd.merge(ratings_tags, movies, on='movieId', how='left')

# 5. Finally add links (also based on movieId)
full_data = pd.merge(ratings_tags_movies, links, on='movieId', how='left')

# 6. View the first few rows of the merged result
print(full_data.head())

# Save it.
full_data.to_csv('full_merged_data1.csv', index=False)
