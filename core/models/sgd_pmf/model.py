import os

import numpy as np

from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models.base_model import Learner


class SGDPMF(Learner):
    default_params = {'n_factors': 20, 'initial_std': 0.3, 'learning_rate': 0.001, 'epochs': 150}

    n_factors = None
    initial_std = None
    learning_rate = None
    epochs = None

    non_zero_elems_row_ids = None
    non_zero_elems_col_ids = None
    training_indices = None
    user_embeddings = None
    item_embeddings = None
    r_std = None
    u_lambda = None
    v_lambda = None

    params = ['n_factors', 'initial_std', 'learning_rate', 'epochs']
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

        self.non_zero_elems_row_ids, self.non_zero_elems_col_ids = (~np.isnan(self.train_ratings_matrix)).nonzero()
        self.training_indices = np.arange(len(self.non_zero_elems_row_ids))

        self.r_std = np.nanstd(self.train_ratings_matrix)
        self.u_lambda = (self.r_std ** 2) / (self.initial_std ** 2)
        self.v_lambda = (self.r_std ** 2) / (self.initial_std ** 2)

        if not self.fitted:
            self.user_embeddings = np.random.normal(0.0, self.initial_std, (self.n_factors, self.n_users))
            self.item_embeddings = np.random.normal(0.0, self.initial_std, (self.n_factors, self.n_movies))

    def train(self):
        while self.current_epoch < self.epochs:
            self.predicted_ratings = self.user_embeddings.T.dot(self.item_embeddings)
            self.evaluate(train=True)

            self.pmf_step()
            self.u_lambda = self.get_lambda_for_data(self.user_embeddings)
            self.v_lambda = self.get_lambda_for_data(self.item_embeddings)

            self.current_epoch += 1

        self.predicted_ratings = self.user_embeddings.T.dot(self.item_embeddings)
        self.evaluate(train=True)

    def pmf_step(self):
        for user_idx in np.unique(self.non_zero_elems_row_ids):
            self.u_lambda = self.get_lambda_for_data(self.user_embeddings)
            user_embedding = self.user_embeddings[:, user_idx]

            rated_items_indices = ~np.isnan(self.train_ratings_matrix[user_idx])
            rated_items_embeddings = self.item_embeddings[:, rated_items_indices]

            user_grad = - (
                    self.train_ratings_matrix[user_idx, rated_items_indices] - user_embedding.T.dot(rated_items_embeddings)
            ).dot(rated_items_embeddings.T) + self.u_lambda * user_embedding.T
            self.user_embeddings[:, user_idx] -= user_grad * self.learning_rate

        for item_idx in np.unique(self.non_zero_elems_col_ids):
            self.v_lambda = self.get_lambda_for_data(self.item_embeddings)
            item_embedding = self.item_embeddings[:, item_idx]

            watchers_indices = ~np.isnan(self.train_ratings_matrix[:, item_idx])
            watchers_embeddings = self.user_embeddings[:, watchers_indices]

            item_grad = - (
                self.train_ratings_matrix[watchers_indices, item_idx] - watchers_embeddings.T.dot(item_embedding)
            ).T.dot(watchers_embeddings.T) + self.v_lambda * item_embedding.T

            self.item_embeddings[:, item_idx] -= item_grad * self.learning_rate

    def get_lambda_for_data(self, data):
        return (self.r_std ** 2) / (np.std(data) ** 2)

    def log_a_posteriori(self):
        predicted_ratings = self.user_embeddings.T.dot(self.item_embeddings)
        diff_matrix = (
                self.train_ratings_matrix[~np.isnan(self.train_ratings_matrix)] - 
                predicted_ratings[~np.isnan(self.train_ratings_matrix)]
        )

        return -0.5 * (
                np.sum(diff_matrix.dot(diff_matrix.T))
                + self.u_lambda * np.sum(self.user_embeddings.dot(self.user_embeddings.T))
                + self.v_lambda * np.sum(self.item_embeddings.dot(self.item_embeddings.T))
        )


if __name__ == '__main__':
    params = {
        'n_factors': 20,
        'initial_std': 0.3,
        'learning_rate': 0.001,
        'epochs': 50,
    }
    model = SGDPMF(
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
        params=params,
        verbose=True,
    )
    model.fit()
    model.evaluate(final=True)

    # model.find_best_params({
    #     'epochs': [150, 200],
    #     'n_factors': [20, 40, 80],
    #     'learning_rate': [0.001],
    #     'initial_std': [0.3, 0.5, 1, 2, 3],
    # })
    # model.save_model_data()

    # print(model.best_params)
