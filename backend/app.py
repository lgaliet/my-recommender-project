from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from recommender import RecommenderAPI

# initialize Flask app, and enable CORS
app = Flask(__name__)
CORS(app)

# path to model, label encoder, and data
MODEL_PATH = "models/recommendation_model.pth"
LE_MOVIE_PATH = "models/le_movie.pkl"
MOVIE_DATA_PATH = "data/movies.csv"

# initialize recommender system
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recommender = RecommenderAPI(
    model_path=MODEL_PATH,
    le_movie_path=LE_MOVIE_PATH, 
    movie_data_path=MOVIE_DATA_PATH,
    device=device
)

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    # API endpoint for getting movie recommendations based on watched movies
    try: 
        # parse input
        data = request.json
        watched_movies = data.get("movies", [])

        if not watched_movies or not isinstance(watched_movies, list):
            return jsonify({"error": "Invalid input. 'movies' must be a non-empty list."}), 400
        
        # get recommendations
        # call method from recommender.py
        recommendations = recommender.recommend_movies_for_new_user(watched_movies)

        return jsonify({"recommendations": recommendations})
    except Exception as e:
        # catches exceptions
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)