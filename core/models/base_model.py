import os
import pickle
import sys

from collections import defaultdict
from tabulate import tabulate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid

from core.data_processing import get_data_matrix_from_dataset


MIN_RATING = 0.5
MAX_RATING = 5


class Learner:
    name = ''
    fitted = False
    best_params = None

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)
    evaluation_train = {}
    evaluation_test = {}

    epochs = 0
    current_epoch = 0
    print_every_n_epochs = 10

    user_id_to_index = None
    movie_id_to_index = None

    params = []
    real_metrics = []
    class_metrics = []
    data_fields = ['train_ratings_matrix', 'test_ratings_matrix']

    fields = [
        'user_id_to_index',
        'movie_id_to_index',
        'best_params',
        'params_grid',
        'params_tuning_history',
        'train_losses',
        'val_losses',
        'train_ratings_path',
        'test_ratings_path',
        'n_users',
        'n_movies',
        'current_epoch',
    ]
    params_grid = []
    params_tuning_history = []

    predicted_ratings = None
    user_embeddings = None
    item_embeddings = None

    def __init__(self, fitted=False, verbose=False, **kwargs):
        self.dir = os.path.dirname(os.path.abspath(sys.modules[self.__class__.__module__].__file__))
        self.name = kwargs.get('name', '')

        self.fitted = fitted
        self.verbose = verbose

        if self.fitted:
            self.fitted = True
            self.load_model_data()
            return

        self.train_ratings_path = kwargs['train_ratings']
        self.test_ratings_path = kwargs['test_ratings']

        self.train_ratings = pd.read_csv(self.train_ratings_path)
        self.test_ratings = pd.read_csv(self.test_ratings_path)
        self.train_ratings_matrix, self.test_ratings_matrix = self.get_ratings_as_matrices()
        self.n_users, self.n_movies = self.train_ratings_matrix.shape

        self.initialize_params(**kwargs)

    def initialize_params(self, **kwargs):
        for param, value in kwargs.get('params', {}).items():
            if param not in self.params:
                raise Exception(f'Unknown parameter {param}.')

            setattr(self, param, value)

    def get_ratings_as_matrices(self):
        train_ratings_matrix, user_id_to_index, movie_id_to_index = get_data_matrix_from_dataset(self.train_ratings)
        self.user_id_to_index, self.movie_id_to_index = user_id_to_index, movie_id_to_index
        predefined = {
            'n_users': train_ratings_matrix.shape[0],
            'n_movies': train_ratings_matrix.shape[1],
            'user_id_to_index': user_id_to_index,
            'movie_id_to_index': movie_id_to_index,
        }
        test_ratings_matrix, _, _ = get_data_matrix_from_dataset(self.test_ratings, predefined)

        return train_ratings_matrix, test_ratings_matrix

    def fit(self):
        self.train()
        self.fitted = True
        self.save_model_data()

    def train(self):
        pass

    def save_model_data(self):
        os.makedirs(os.path.join(self.dir, self.name, 'data', 'images'), exist_ok=True)
        model_data = {'params': {}, 'fields': {}}

        for param in self.params:
            model_data['params'][param] = getattr(self, param)

        for field in self.fields:
            model_data['fields'][field] = getattr(self, field)

        with open(os.path.join(self.dir, self.name, 'data', 'model_data.pickle'), 'wb') as f:
            pickle.dump(model_data, f)

        for field_name in self.data_fields:
            field_data = getattr(self, field_name)
            np.savetxt(os.path.join(self.dir, self.name, 'data', f'{field_name}.csv'), field_data, delimiter=',')

    def load_model_data(self):
        with open(os.path.join(self.dir, self.name, 'data', 'model_data.pickle'), 'rb') as f:
            model_data = pickle.load(f)

        for field in self.fields:
            setattr(self, field, model_data['fields'][field])

        self.train_ratings = pd.read_csv(self.train_ratings_path)
        self.test_ratings = pd.read_csv(self.test_ratings_path)

        for field_name in self.data_fields:
            field_data = np.genfromtxt(os.path.join(self.dir, self.name, 'data', f'{field_name}.csv'), delimiter=',')
            setattr(self, field_name, field_data)

        self.initialize_params(params=model_data['params'])

    def evaluate(self, train=False, final=False):
        predicted_train_ratings, true_train_ratings = self.bulk_predict_ratings(self.train_ratings_matrix)
        predicted_test_ratings, true_test_ratings = self.bulk_predict_ratings(self.test_ratings_matrix)

        predicted_train_likes, true_train_likes = self.bulk_predict_likes(self.train_ratings_matrix)
        predicted_test_likes, true_test_likes = self.bulk_predict_likes(self.test_ratings_matrix, test=True)

        if self.verbose and self.current_epoch % self.print_every_n_epochs == 0:
            print(f'Epoch {self.current_epoch}')

        self.calculate_metrics(
            self.real_metrics,
            predicted_train_ratings,
            true_train_ratings,
            predicted_test_ratings,
            true_test_ratings,
            train=train,
            final=final,
        )
        self.calculate_metrics(
            self.class_metrics,
            predicted_train_likes,
            true_train_likes,
            predicted_test_likes,
            true_test_likes,
            train=train,
            final=final,
        )

        if final:
            table = [[epoch] for epoch in range(0, self.current_epoch + 1, self.print_every_n_epochs)]
            headers = ['epoch']
            for metric in self.real_metrics + self.class_metrics:
                headers.append(f'Train {metric.name}')
                headers.append(f'Test {metric.name}')
                for epoch in range(0, self.current_epoch + 1, self.print_every_n_epochs):
                    table[epoch // self.print_every_n_epochs].append(self.train_losses[metric.name][epoch])
                    table[epoch // self.print_every_n_epochs].append(self.val_losses[metric.name][epoch])

            print(tabulate(table, headers=headers))

            self.plot_train_losses('Real')
            self.plot_train_losses('Class')

        return self.train_losses, self.val_losses

    def calculate_metrics(self, metrics, predicted_train, true_train, predicted_test, true_test, train, final):
        metrics_to_print = []
        for metric in metrics:
            train_metric_value = metric(predicted_train, true_train, verbose=final).eval()
            test_metric_value = metric(predicted_test, true_test, verbose=final).eval()
            if train:
                self.train_losses[metric.name].append(train_metric_value)
                self.val_losses[metric.name].append(test_metric_value)

            self.evaluation_train[metric.name] = train_metric_value
            self.evaluation_test[metric.name] = test_metric_value

            if self.verbose and self.current_epoch % self.print_every_n_epochs == 0:
                metrics_to_print += [
                    f'Train {metric.name}: {train_metric_value}',
                    f'Test {metric.name}: {test_metric_value}',
                ]

        if self.verbose and self.current_epoch % self.print_every_n_epochs == 0:
            print(', '.join(metrics_to_print))

    def plot_train_losses(self, metric_type='Real'):
        names_for_legend = []
        metrics = self.real_metrics if metric_type == 'Real' else self.class_metrics
        for metric in metrics:
            names_for_legend += [f'train {metric.name.lower()}', f'test {metric.name.lower()}']
            plt.plot(list(range(self.epochs + 1)), self.train_losses[metric.name], c=f'{metric.color}', linestyle='--')
            plt.plot(list(range(self.epochs + 1)), self.val_losses[metric.name], c=f'{metric.color}', linestyle='-')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{metric_type} metric losses over epochs')
        plt.legend(names_for_legend, loc='best')
        plt.savefig(os.path.join('data', 'images', f'{metric_type.lower()}_losses.png'))
        plt.clf()

    def bulk_predict_ratings(self, target_matrix):
        known_predicted_ratings = np.copy(self.predicted_ratings)
        known_predicted_ratings[np.isnan(target_matrix)] = np.nan

        return known_predicted_ratings, target_matrix

    def bulk_predict_likes(self, target_matrix, test=False):
        likes_probabilities, _ = self.bulk_predict_ratings(target_matrix)

        mean_ratings = np.mean(self.predicted_ratings, axis=1).reshape(-1, 1)
        min_rating_deviation = mean_ratings - np.min(self.predicted_ratings, axis=1).reshape(-1, 1)
        max_rating_deviation = np.max(self.predicted_ratings, axis=1).reshape(-1, 1) - mean_ratings

        likes_probabilities = likes_probabilities - mean_ratings
        likes_probabilities[likes_probabilities < 0] = 0.5 * (
                likes_probabilities / min_rating_deviation
        )[likes_probabilities < 0]

        likes_probabilities[likes_probabilities > 0] = 0.5 * (
                likes_probabilities / max_rating_deviation
        )[likes_probabilities > 0]

        likes_probabilities += 0.5

        if test:
            all_ratings = np.nan_to_num(self.train_ratings_matrix) + np.nan_to_num(target_matrix)
            all_ratings[all_ratings == 0] = np.nan
            true_mean_ratings = np.nanmean(all_ratings, axis=1).reshape(-1, 1)
            likes_truth = target_matrix - true_mean_ratings
        else:
            likes_truth = target_matrix - np.nanmean(self.train_ratings_matrix, axis=1).reshape(-1, 1)

        likes_truth[likes_truth >= 0] = 1
        likes_truth[likes_truth < 0] = 0

        return likes_probabilities, likes_truth

    def raise_if_not_fit(self):
        if not self.fitted:
            raise Exception('Модель надо сначала обучить')

    def predict(self, user_id, movie_id) -> int:
        self.raise_if_not_fit()

        user_index = self.user_id_to_index[user_id]
        movie_index = self.movie_id_to_index[movie_id]

        return self.predicted_ratings[user_index][movie_index]

    def get_recommended_items(self, user_id, n):
        user_index = self.user_id_to_index[user_id]
        recommended_items = self.predicted_ratings[user_index].argsort()[-1::-1][:n].astype(np.int)
        movie_index_to_id = {v: k for k, v in self.movie_id_to_index.items()}

        return [movie_index_to_id[ix] for ix in recommended_items]

    def get_top_similar_objects(self, object_id, top_n, object_type='movie'):
        self.raise_if_not_fit()

        if object_type == 'movie':
            id_to_index_mapping = self.movie_id_to_index
            objects_embeddings = self.item_embeddings
        else:
            id_to_index_mapping = self.user_id_to_index
            objects_embeddings = self.user_embeddings

        object_index = id_to_index_mapping[object_id]

        item_similarity = cosine_similarity(objects_embeddings[object_index].reshape(1, -1), objects_embeddings)[0]
        similar_indices = item_similarity.argsort()[-1::-1][:top_n + 1].astype(np.int)

        index_to_id_mapping = {v: k for k, v in id_to_index_mapping.items()}

        return [index_to_id_mapping[ix] for ix in similar_indices if ix != object_index]

    def get_recommended_similar_items(self, user_id, item_id, n):
        similar_items = self.get_top_similar_objects(item_id, n, 'movie')

        return sorted(similar_items, key=lambda similar_item_id: self.predict(user_id, similar_item_id), reverse=True)

    def get_params_grid(self, param_grid):
        self.params_grid = list(ParameterGrid(param_grid))

    def find_best_params(self, params_grid):
        self.get_params_grid(params_grid)

        for params_set in self.params_grid:
            self.fitted = False
            self.initialize_params(params=params_set)

            self.train()
            self.evaluate()
            self.params_tuning_history.append({
                'metrics': {'train': self.evaluation_train.copy(), 'test': self.evaluation_test.copy()},
                'params': params_set,
            })
            print(params_set, 'train:', self.evaluation_train, 'test:', self.evaluation_test)

        best_params = {}
        for metric in self.real_metrics + self.class_metrics:
            comparison = min if metric.decreasing else max
            best_element = comparison(
                self.params_tuning_history,
                key=lambda x: x['metrics']['test'][metric.name],
            )
            best_params[metric.name] = {
                'params': best_element['params'],
                'train': best_element['metrics']['train'][metric.name],
                'test': best_element['metrics']['test'][metric.name],
            }

        self.best_params = best_params
