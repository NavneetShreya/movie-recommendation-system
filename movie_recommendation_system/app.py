import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load your movie data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Basic content-based filtering using genres
def content_based_filtering(movie_title):
    # Using CountVectorizer to convert genres to a count matrix
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(movies['genres'])
    
    # Compute the cosine similarity matrix based on the count_matrix
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Get index of the movie that matches the title
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]

# Example usage:
recommended_movies = content_based_filtering("Toy Story")
print(recommended_movies)
