import torch
import pandas as pd
import joblib
from train import RecSysModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderAPI:
   
    def __init__(self, model_path, le_movie_path, movie_data_path, device):
        # Load model and set it to evaluation mode
        self.device = device
        self.model = RecSysModel(num_users=610, num_movies=9724)  # Adjust dimensions as necessary
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
       
        # Load the label encoder for movies and movie data
        self.le_movie = joblib.load(le_movie_path)
        self.df_movies = pd.read_csv(movie_data_path)

    def recommend_movies_for_new_user(self, watched_movie_titles, k=10):
        """Generate movie recommendations based on similar movies for a new user."""

        # Get movie IDs and genres for watched movies
        watched_movies = self.df_movies[self.df_movies['title'].isin(watched_movie_titles)]
        watched_movie_ids = watched_movies['movieId'].tolist()
        watched_genres = set("|".join(watched_movies['genres'].tolist()).split("|"))

        # Encode the movie IDs
        watched_movie_indices = self.le_movie.transform(watched_movie_ids)
        
        # Extract embeddings for each watched movie
        with torch.no_grad():
            watched_movie_embeddings = self.model.movie_embedding(torch.tensor(watched_movie_indices).to(self.device))
        
        # Calculate similarity for each watched movie embedding individually
        all_movie_indices = torch.arange(self.model.movie_embedding.num_embeddings).to(self.device)
        all_movie_embeddings = self.model.movie_embedding(all_movie_indices).detach().cpu().numpy()
        
        aggregated_scores = None
        for watched_embedding in watched_movie_embeddings:
            # Compute similarity scores for each watched movie
            similarity_scores = cosine_similarity(watched_embedding.detach().cpu().numpy().reshape(1, -1), all_movie_embeddings).flatten()
            if aggregated_scores is None:
                aggregated_scores = similarity_scores
            else:
                aggregated_scores += similarity_scores  # Aggregate scores from each watched movie

        # Get the indices of the top K recommended movies
        similar_movie_indices = aggregated_scores.argsort()[::-1]
        
        # Filter out already watched movies and match genres
        recommendations = []
        for idx in similar_movie_indices:
            if idx in watched_movie_indices:
                continue
            
            movie_id = self.le_movie.inverse_transform([idx])[0]
            movie_genres = set(self.df_movies[self.df_movies['movieId'] == movie_id]['genres'].values[0].split("|"))
            
            # Check if genres match any from watched movies
            if watched_genres & movie_genres:  # Intersection of genres
                recommendations.append(movie_id)
            
            if len(recommendations) >= k:
                break
        
        # Return titles and genres for recommendations
        return self.get_movies_with_genres(recommendations)

    def get_movies_with_genres(self, movie_ids):
        """Fetch movie titles along with genres for given movie IDs."""
        movies_with_genres = self.df_movies[self.df_movies['movieId'].isin(movie_ids)].copy()
        movies_with_genres['title_with_genres'] = movies_with_genres[['title', 'genres']].agg(' - '.join, axis=1)
        return movies_with_genres['title_with_genres'].tolist()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recommendation_model_path = "models/recommendation_model.pth"
    le_movie_path = "models/le_movie.pkl"
    movie_data_path = "data/movies.csv"
   
    recommender = RecommenderAPI(
        model_path=recommendation_model_path,
        le_movie_path=le_movie_path,
        movie_data_path=movie_data_path,
        device=device
    )
   
    # Example input for a new user
    watched_movies = ["Avengers: Age of Ultron (2015)", "Thor (2011)"]
    recommended_movies = recommender.recommend_movies_for_new_user(watched_movies)
   
    print(f"Recommended movies based on the watched list:\n{recommended_movies}, based on the movies:\n{watched_movies}")



