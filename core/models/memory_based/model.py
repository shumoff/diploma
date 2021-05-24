import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models.base_model import Learner


class MemoryBased(Learner):
    default_params = {'sim_users_amount': 100, 'mean_centered': True, 'standardized': True, 'item_based': False}

    sim_users_amount = None
    mean_centered = None
    standardized = None
    item_based = None

    params = ['sim_users_amount', 'mean_centered', 'standardized', 'item_based']
    real_metrics = [RMSE, NDCGScore, MAE]
    class_metrics = [F1Score, PRScore, ROCScore]
    data_fields = ['train_ratings_matrix', 'test_ratings_matrix', 'predicted_ratings']

    def train(self):
        ratings_matrix = np.nan_to_num(self.train_ratings_matrix)
        predictions = np.zeros((self.n_users, self.n_movies))
        if self.item_based:
            ratings_matrix = ratings_matrix.T
            predictions = predictions.T

        if self.mean_centered:
            mean_ratings = np.ma.array(ratings_matrix, mask=ratings_matrix == 0).mean(axis=1)
            mean_centered_ratings_matrix = np.where(ratings_matrix == 0, 0, (ratings_matrix.T - mean_ratings).T)
            similarity_matrix = cosine_similarity(mean_centered_ratings_matrix)
        else:
            similarity_matrix = cosine_similarity(ratings_matrix)

        for i in range(ratings_matrix.shape[0]):
            users_similarity = similarity_matrix[i]
            similar_users_indices = users_similarity.argsort()[-1::-1][1:self.sim_users_amount + 1].astype(np.int)
            weights = users_similarity[similar_users_indices]
            sim_ratings_matrix = ratings_matrix[similar_users_indices]

            user_mean_rating = ratings_matrix[i][ratings_matrix[i] > 0].mean()
            sim_users_mean_vector = np.ma.array(sim_ratings_matrix, mask=sim_ratings_matrix == 0).mean(axis=1)

            normalized_sim_ratings_matrix = (sim_ratings_matrix.T - sim_users_mean_vector).T
            normalized_sim_ratings_matrix = np.where(sim_ratings_matrix == 0, 0, normalized_sim_ratings_matrix)

            if self.standardized:
                user_standard_deviation = np.std(ratings_matrix[i][ratings_matrix[i] > 0])
                sim_users_std_vector = np.ma.array(sim_ratings_matrix, mask=sim_ratings_matrix == 0).std(axis=1)
                normalized_sim_ratings_matrix = (normalized_sim_ratings_matrix.T / sim_users_std_vector).T

                numerators = weights.dot(normalized_sim_ratings_matrix)
                denominators = weights.dot(sim_ratings_matrix > 0)
                denominators[denominators == 0] = 1
                predictions[i] = user_mean_rating + user_standard_deviation * numerators / denominators
            else:
                numerators = weights.dot(normalized_sim_ratings_matrix)
                denominators = weights.dot(sim_ratings_matrix > 0)
                denominators[denominators == 0] = 1
                predictions[i] = user_mean_rating + numerators / denominators

        if self.item_based:
            predictions = predictions.T

        self.user_embeddings = predictions
        self.item_embeddings = predictions.T

        self.predicted_ratings = predictions
        self.evaluate()


if __name__ == '__main__':
    params = {
        'sim_users_amount': 100,
        'mean_centered': True,
        'standardized': True,
        'item_based': True,
    }
    model = MemoryBased(
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
        params=params,
        name='item_based',
    )

    model.fit()
    model.evaluate(final=True)
