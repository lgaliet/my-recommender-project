import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn import preprocessing, model_selection
from sklearn.metrics import root_mean_squared_error
from collections import defaultdict
import sys
import pandas as pd
import joblib

# Uses collaborative filtering

# Define the Dataset
class MovieDataset(Dataset):
    # initialize the dataset object with user, movie, and rating data
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    # gets the total number of samples in the dataset
    def __len__(self):
        return len(self.users)
    
    # gets a sample form the dataset at the specified index
    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]
        
        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }

# recommendation system model
# neural network designed for making recommendations
# structure overview - embedding layers for users and movies to provide
# categorical data into continuous space, hidden layers/non-linearity 
# to allow model to learn complex relationships with ReLU, and a dropout 
# layer to prevent overfitting.

class RecSysModel(nn.Module):
    # use embeddingsfor dimensionality reduction, collaborative filterning, 
    # and adjustments to minimize the prediction error.
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.2,
    ):
        super(RecSysModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
            
        # embedding layers
        self.user_embedding = nn.Embedding (
            num_embeddings = self.num_users,
            embedding_dim=self.embedding_size
        )
        self.movie_embedding = nn.Embedding (
            num_embeddings=self.num_movies,
            embedding_dim=self.embedding_size
        )
        # hidden layers
        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        # dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        # activation function
        self.relu = nn.ReLU()
    
    def forward(self, users, movies):
        # embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)
        # concatenate user and movie embeddings
        combined = torch.cat([user_embedded, movie_embedded], dim=1)
        # pass through hiddle layers with ReLU activation and dropout
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)
        return output

def train():
    # preprocess and encode data befor feeding it into a model
    # convert userId and movieId into numerical format suitable for model
    df = pd.read_csv("data/ratings.csv")
    le_user = preprocessing.LabelEncoder()
    le_movie = preprocessing.LabelEncoder()
    df.userId = le_user.fit_transform(df.userId.values)
    df.movieId = le_movie.fit_transform(df.movieId.values)

    # splitting the dataset
    df_train, df_val = model_selection.train_test_split(
        df, test_size=0.1, random_state=3, stratify=df.rating.values
    )

    # data loaders
    train_dataset = MovieDataset(df_train.userId.values, df_train.movieId.values, df_train.rating.values)
    valid_dataset = MovieDataset(df_val.userId.values, df_val.movieId.values, df_val.rating.values)


    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training loop setup
    recommendation_model = RecSysModel(
        num_users=len(le_user.classes_),
        num_movies=len(le_movie.classes_),
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(recommendation_model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    EPOCHS = 5

    # Function to log progress
    def log_progress(epoch, step, total_loss, log_progress_step, data_size, losses):
        avg_loss = total_loss / log_progress_step
        sys.stderr.write(
            f"\r{epoch+1:02d}/{EPOCHS:02d} | Step: {step}/{data_size} | Avg Loss: {avg_loss:<6.9f}"
        )

        sys.stderr.flush()
        losses.append(avg_loss)
        
    total_loss = 0
    log_progress_step = 100
    losses = []
    train_dataset_size = len(train_dataset)
    print(f"Training on {train_dataset_size} samples...")

    
    for e in range(EPOCHS):
        recommendation_model.train()
        # reset step count at the beginning of each epoch
        step_count = 0  # Changed to step_count
        for i, train_data in enumerate(train_loader):
            print(f"Batch {i}: {train_data}")  # Debugging line to see the content of train_data
            output = recommendation_model(
                train_data["users"].to(device), train_data["movies"].to(device)
            )
            # reshape model output to match the target's shape
            output = output.squeeze()
            # removes singleton dimension
            ratings = (
                train_data["ratings"].to(torch.float32).to(device)
            )
            # assuming ratings is already ID
            loss = loss_func(output, ratings)
            total_loss += loss.sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # increment step count by the actual size of the batch
            step_count += len(train_data["users"])
            # check if it's time to log progress
            if step_count % log_progress_step == 0 or i == len(train_loader) - 1:
                # log at the end of each epoch
                log_progress(e, step_count, total_loss, log_progress_step, train_dataset_size, losses)
                total_loss = 0

    # RMSE Calculation
    y_pred = []
    y_true = []

    recommendation_model.eval()

    with torch.no_grad():
        for i, valid_data in enumerate(val_loader):
            output = recommendation_model(
                valid_data["users"].to(device), valid_data["movies"].to(device)
            )
            ratings = valid_data["ratings"].to(device)
            y_pred.extend(output.cpu().numpy())
            y_true.extend(ratings.cpu().numpy())

        # Calculate RMSE
        rms = root_mean_squared_error(y_true, y_pred)
        print(f"RMSE: {rms:.4f}")

    # Precision@k and Recall@k
    def calculate_precision_recall(user_ratings, k, threshold):
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k])
        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
        return precision, recall

    user_ratings_comparison = defaultdict(list)
    with torch.no_grad():
        for valid_data in val_loader:
            users = valid_data["users"].to(device)
            movies = valid_data["movies"].to(device)
            ratings = valid_data["ratings"].to(device)
            output = recommendation_model(users, movies)
            for user, pred, true in zip(users, output, ratings):
                user_ratings_comparison[user.item()].append((pred[0].item(), true.item()))

    k, threshold = 50, 3
    user_precisions, user_recalls = {}, {}

    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall = calculate_precision_recall(user_ratings, k, threshold)
        user_precisions[user_id] = precision
        user_recalls[user_id] = recall

    average_precision = sum(user_precisions.values()) / len(user_precisions)
    average_recall = sum(user_recalls.values()) / len(user_recalls)
    print(f"precision @ {k}: {average_precision:.4f}")
    print(f"recall @ {k}: {average_recall:.4f}")

    # save the trained model path
    model_save_path = "models/recommendation_model.pth"
    torch.save(recommendation_model.state_dict(), model_save_path)

    # save the label encoder for movies
    le_movie_save_path = "models/le_movie.pkl"
    joblib.dump(le_movie, le_movie_save_path)




if __name__ == '__main__':
    train()