import pandas as pd
from google.colab import files

# Read CSV files
tags = pd.read_csv('tags.csv')
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
links = pd.read_csv('links.csv')
genome_tags = pd.read_csv('genome-tags.csv')
genome_scores = pd.read_csv('genome-scores.csv')

# Rename columns with duplicates (to prevent confusion)
tags = tags.rename(columns={'timestamp': 'timestamp_tags'})
ratings = ratings.rename(columns={'timestamp': 'timestamp_ratings'})

# Merge ratings and tags (by userId and movieId)
user_movie = pd.merge(ratings, tags, on=['userId', 'movieId'], how='left')

# Merge movies info
user_movie = pd.merge(user_movie, movies, on='movieId', how='left')

# Merge links information
user_movie = pd.merge(user_movie, links, on='movieId', how='left')

# Merge genome_scores and genome_tags (add tag names)
genome_scores = pd.merge(genome_scores, genome_tags, on='tagId', how='left')

# Get the top 5 most relevant genome tags for each movieId (to prevent merge explosion)
top_genome = genome_scores.sort_values(['movieId', 'relevance'], ascending=[True, False])
top_genome = top_genome.groupby('movieId').head(5)

# Splicing the first 5 tags into a string
top_genome = top_genome.groupby('movieId').agg({
    'tag': lambda x: ', '.join(x),
    'relevance': lambda x: ', '.join(map(str, x))
}).reset_index().rename(columns={
    'tag': 'top_genome_tags',
    'relevance': 'top_genome_relevance'
})

# Merge top genome information
user_movie = pd.merge(user_movie, top_genome, on='movieId', how='left')

# Save as a new CSV file
user_movie.to_csv('combined_movie_data.csv', index=False)
print("文件保存成功：combined_movie_data.csv")
