import os

import numpy as np

from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models.base_model import Learner


class SGDFactorization(Learner):
    default_params = {'epochs': 100, 'n_factors': 80, 'learning_rate': 0.01, 'reg': 0.1}

    epochs = None
    n_factors = None
    learning_rate = None
    user_reg = None
    item_reg = None

    non_zero_elems_row_ids = None
    non_zero_elems_col_ids = None
    training_indices = None
    user_embeddings = None
    item_embeddings = None
    user_bias = None
    item_bias = None
    global_bias = None

    params = ['epochs', 'n_factors', 'learning_rate', 'user_reg', 'item_reg']
    real_metrics = [RMSE, NDCGScore, MAE]
    class_metrics = [F1Score, PRScore, ROCScore]
    data_fields = [
        'train_ratings_matrix',
        'test_ratings_matrix',
        'user_embeddings',
        'item_embeddings',
        'user_bias',
        'item_bias',
        'predicted_ratings',
    ]

    def initialize_params(self, **kwargs):
        super().initialize_params(**kwargs)

        self.non_zero_elems_row_ids, self.non_zero_elems_col_ids = (~np.isnan(self.train_ratings_matrix)).nonzero()
        self.training_indices = np.arange(len(self.non_zero_elems_row_ids))
        self.global_bias = np.nanmean(self.train_ratings_matrix)

        if not self.fitted:
            self.user_embeddings = np.random.normal(
                scale=1. / self.n_factors,
                size=(self.n_users, self.n_factors),
            )
            self.item_embeddings = np.random.normal(
                scale=1. / self.n_factors,
                size=(self.n_movies, self.n_factors),
            )
            self.user_bias = np.zeros(self.n_users)
            self.item_bias = np.zeros(self.n_movies)

    def train(self):
        while self.current_epoch < self.epochs:
            self.predicted_ratings = (
                    self.global_bias + self.user_bias.reshape(-1, 1) + self.item_bias.reshape(1, -1) +
                    self.user_embeddings.dot(self.item_embeddings.T)
            )
            self.evaluate(train=True)

            self.sgd()

            self.current_epoch += 1

        self.predicted_ratings = (
                self.global_bias + self.user_bias.reshape(-1, 1) + self.item_bias.reshape(1, -1) +
                self.user_embeddings.dot(self.item_embeddings.T)
        )
        self.evaluate(train=True)

    def sgd(self):
        np.random.shuffle(self.training_indices)
        for idx in self.training_indices:
            user_idx = self.non_zero_elems_row_ids[idx]
            item_idx = self.non_zero_elems_col_ids[idx]

            prediction = (
                    self.global_bias + self.user_bias[user_idx] + self.item_bias[item_idx] +
                    self.user_embeddings[user_idx].dot(self.item_embeddings[item_idx].T)
            )
            err = self.train_ratings_matrix[user_idx, item_idx] - prediction

            user_embedding = self.user_embeddings[user_idx]
            item_embedding = self.item_embeddings[item_idx]

            self.user_bias[user_idx] += self.learning_rate * (err - self.user_reg * self.user_bias[user_idx])
            self.item_bias[item_idx] += self.learning_rate * (err - self.item_reg * self.item_bias[item_idx])

            self.user_embeddings[user_idx] += self.learning_rate * (err * item_embedding - self.user_reg * user_embedding)
            self.item_embeddings[item_idx] += self.learning_rate * (err * user_embedding - self.item_reg * item_embedding)


if __name__ == '__main__':
    params = {
        'epochs': 70,  # после 70 переобучается
        'n_factors': 80,
        'learning_rate': 0.01,
        'user_reg': 0.1,
        'item_reg': 0.01,
    }
    model = SGDFactorization(
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
        params=params,
        verbose=True,
        name='1KK'
    )
    model.fit()
    model.evaluate(final=True)

    # model.find_best_params({
    #     'epochs': [50, 100],
    #     'n_factors': [5, 10, 20, 40, 80],
    #     'learning_rate': [0.001, 0.01, 0.1],
    #     'user_reg': [0.01, 0.05, 0.1, 0.5, 1],
    #     'item_reg': [0.01, 0.05, 0.1, 0.5, 1],
    # })
    # print(model.best_params)
    # model.save_model_data()
