import os

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models import Learner


tf.random.set_seed(42)


class NeuralNet(Learner):
    default_params = {
        'learning_rate': 0.002,
        'embedding_dim': 20,
        'intermediate_dim': 350,
        'latent_dim': 80,
        'epochs': 15,
    }

    learning_rate = None
    embedding_dim = None
    intermediate_dim = None
    latent_dim = None
    epochs = None

    train_pairs = None
    val_pairs = None
    all_pairs = None
    net = None

    params = ['learning_rate', 'embedding_dim', 'intermediate_dim', 'latent_dim', 'epochs']

    print_every_n_epochs = 5
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
        self.train_pairs = np.array(list(zip(*(~np.isnan(self.train_ratings_matrix)).nonzero())))
        self.val_pairs = np.array(list(zip(*(~np.isnan(self.test_ratings_matrix)).nonzero())))
        user_indices_list = list(self.user_id_to_index.values())
        movie_indices_list = list(self.movie_id_to_index.values())
        self.all_pairs = np.transpose([
            np.tile(user_indices_list, len(movie_indices_list)),
            np.repeat(movie_indices_list, len(user_indices_list)),
        ])
        self.net = create_model(self.n_users, self.n_movies, self.embedding_dim, self.intermediate_dim, self.latent_dim)

    def train(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            decay_steps=495,
            decay_rate=0.96,
            staircase=True)
        self.net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        for param in self.params:
            print(param, getattr(self, param))

        print(self.net.summary())

        train_users, train_movies = self.train_pairs[:, 0], self.train_pairs[:, 1]
        val_users, val_movies = self.val_pairs[:, 0], self.val_pairs[:, 1]
        train_ratings = self.train_ratings_matrix[(train_users, train_movies)]
        val_ratings = self.test_ratings_matrix[(val_users, val_movies)]
        test_users, test_movies = self.all_pairs[:, 0], self.all_pairs[:, 1]

        while self.current_epoch < self.epochs:
            self.predicted_ratings = self.net.predict(
                [test_users, test_movies],
            ).reshape((self.n_users, self.n_movies), order='F')
            self.evaluate(train=True)

            self.net_epoch(train_users, train_movies, train_ratings, val_users, val_movies, val_ratings)

            self.current_epoch += 1

        self.predicted_ratings = self.net.predict(
            [test_users, test_movies],
        ).reshape((self.n_users, self.n_movies), order='F')
        self.evaluate(train=True)

        self.user_embeddings = self.net.get_layer(name='user_emb').embeddings.numpy()
        self.item_embeddings = self.net.get_layer(name='movie_emb').embeddings.numpy()
        self.evaluate(final=True)
        print(self.evaluation_train)
        print(self.evaluation_test)

    def net_epoch(self, train_users, train_movies, train_ratings, val_users, val_movies, val_ratings):
        self.net.fit(
            [train_users, train_movies], train_ratings,
            epochs=self.current_epoch + 1,
            initial_epoch=self.current_epoch,
            shuffle=True,
            validation_data=([val_users, val_movies], val_ratings),
        )


def create_model(user_vocab_size, movie_vocab_size, embedding_dim, intermediate_dim, latent_dim):
    user_input = tf.keras.Input(shape=(), name="user_input")
    movie_input = tf.keras.Input(shape=(), name="movie_input")
    user_embedding_layer = tf.keras.layers.Embedding(user_vocab_size, embedding_dim, name='user_emb')(user_input)
    movie_embedding_layer = tf.keras.layers.Embedding(movie_vocab_size, embedding_dim, name='movie_emb')(movie_input)
    concat_embeddings = tf.keras.layers.concatenate([user_embedding_layer, movie_embedding_layer], name='concat_emb')
    zero_dropout = tf.keras.layers.Dropout(0.5, seed=42)(concat_embeddings)
    first_relu = tf.keras.layers.Dense(intermediate_dim, activation='relu', name='first_relu')(zero_dropout)
    first_dropout = tf.keras.layers.Dropout(0.4, seed=42)(first_relu)
    second_relu = tf.keras.layers.Dense(latent_dim, activation='relu', name='second_relu')(first_dropout)
    second_dropout = tf.keras.layers.Dropout(0.15, seed=42)(second_relu)
    rating_output = 0.5 + 4.5 * tf.keras.layers.Dense(1, activation='sigmoid', name='output')(second_dropout)

    return tf.keras.Model(
        inputs=[user_input, movie_input],
        outputs=[rating_output],
    )


if __name__ == '__main__':
    params = {
        'learning_rate': 0.002,
        'embedding_dim': 100,
        'intermediate_dim': 150,
        'latent_dim': 100,
        'epochs': 15,
    }
    model = NeuralNet(
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
        verbose=True,
        params=params,
        name='1KK',
        fitted=True,
    )
    model.fit()
    model.evaluate(final=True)
    # model.find_best_params({
    #     'learning_rate': [0.001, 0.0001],
    #     'embedding_dim': [20, 40, 60, 100],
    #     'intermediate_dim': [50, 150, 350],
    #     'latent_dim': [20, 50, 150],
    #     'epochs': [20],
    # })
    # print(model.best_params)
