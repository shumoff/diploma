import os

import numpy as np

from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models.base_model import Learner


class ALSFactorization(Learner):
    default_params = {'epochs': 40, 'n_factors': 100, 'reg': 6}

    epochs = None
    n_factors = None
    reg = None

    user_embeddings = None
    item_embeddings = None

    params = ['epochs', 'n_factors', 'reg']
    real_metrics = [RMSE, NDCGScore, MAE]
    class_metrics = [F1Score, PRScore, ROCScore]
    data_fields = [
        'train_ratings_matrix',
        'test_ratings_matrix',
        'user_embeddings',
        'item_embeddings',
        'predicted_ratings',
    ]

    def initialize_params(self, **kwargs):
        super().initialize_params(**kwargs)

        if not self.fitted:
            self.user_embeddings = np.random.normal(
                scale=1. / self.n_factors,
                size=(self.n_users, self.n_factors),
            )
            self.item_embeddings = np.random.normal(
                scale=1. / self.n_factors,
                size=(self.n_movies, self.n_factors),
            )

    def train(self):
        while self.current_epoch < self.epochs:
            self.predicted_ratings = self.user_embeddings.dot(self.item_embeddings.T)
            self.evaluate(train=True)

            self.als_step(type='user')
            self.als_step(type='item')

            self.current_epoch += 1

        self.predicted_ratings = self.user_embeddings.dot(self.item_embeddings.T)
        self.evaluate(train=True)

    def als_step(self, type='user',):
        if type == 'user':
            for user_idx in range(self.n_users):
                rated_items_indices = ~np.isnan(self.train_ratings_matrix[user_idx])
                rated_items_embeddings = self.item_embeddings[rated_items_indices]

                YTY = rated_items_embeddings.T.dot(rated_items_embeddings)
                lambdaI = np.eye(YTY.shape[0]) * self.reg

                self.user_embeddings[user_idx, :] = np.linalg.solve(
                    YTY + lambdaI,
                    self.train_ratings_matrix[user_idx, rated_items_indices].dot(rated_items_embeddings),
                )

            return self.user_embeddings

        elif type == 'item':
            for item_idx in range(self.n_movies):
                watchers_indices = ~np.isnan(self.train_ratings_matrix[:, item_idx])
                watchers_embeddings = self.user_embeddings[watchers_indices]

                XTX = watchers_embeddings.T.dot(watchers_embeddings)
                lambdaI = np.eye(XTX.shape[0]) * self.reg

                self.item_embeddings[item_idx, :] = np.linalg.solve(
                    XTX + lambdaI,
                    self.train_ratings_matrix[watchers_indices, item_idx].T.dot(watchers_embeddings),
                )

            return self.item_embeddings


if __name__ == '__main__':
    params = {
        'epochs': 40,
        'n_factors': 100,
        'reg': 6,
    }
    model = ALSFactorization(
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
        params=params,
        verbose=True,
    )
    model.fit()
    model.evaluate(final=True)
