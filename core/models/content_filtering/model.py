import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models.base_model import Learner


class ContentFiltering(Learner):
    default_params = {
        'intermediate_size': 5000,
        'encoded_size': 100,
        'epochs': 20,
    }

    intermediate_size = None
    encoded_size = None
    epochs = None

    embeddings_matrix = None

    params = ['intermediate_size', 'encoded_size', 'epochs']
    real_metrics = [RMSE, NDCGScore, MAE]
    class_metrics = [F1Score, PRScore, ROCScore]
    data_fields = [
        'train_ratings_matrix',
        'test_ratings_matrix',
        'embeddings_matrix',
        'user_embeddings',
        'item_embeddings',
        'predicted_ratings',
    ]

    def initialize_params(self, **kwargs):
        super().initialize_params(**kwargs)

        if not self.fitted:
            movie_content_data = pd.read_csv(kwargs['movie_content_data'])
            movie_content_data = movie_content_data[movie_content_data['movieId'].isin(self.movie_id_to_index)]
            movie_content_data.replace({'movieId': self.movie_id_to_index}, inplace=True)
            movie_content_data.sort_values(by='movieId', ignore_index=True, inplace=True)

            if kwargs['use_tfidf']:
                tfidf_vectorizer = TfidfVectorizer(
                    ngram_range=(1, 1),
                    min_df=0.0001,
                    stop_words='english',
                    token_pattern=r"(?u)\b\w[\w\-]+\b",
                )
                self.embeddings_matrix = tfidf_vectorizer.fit_transform(movie_content_data['document']).toarray()
            else:
                tag_id_to_index = {tag_id: tag_ix for tag_ix, tag_id in enumerate(movie_content_data['tagId'].unique())}
                self.embeddings_matrix = np.zeros((self.n_movies, len(tag_id_to_index)))
                for _, movie_id, tag_id, relevance in movie_content_data.itertuples():
                    self.embeddings_matrix[movie_id][tag_id_to_index[tag_id]] = relevance

            print(self.embeddings_matrix.shape)
            print(self.train_ratings_matrix.shape)
            print(len(self.movie_id_to_index))
            print(self.n_users, self.n_movies)

    def train(self):
        tf.random.set_seed(42)
        autoencoder = Autoencoder(self.embeddings_matrix.shape[1], self.intermediate_size, self.encoded_size)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        train_tfidf, test_tfidf = train_test_split(
            self.embeddings_matrix,
            test_size=0.2,
            random_state=42,
        )

        autoencoder_train_losses, autoencoder_val_losses = autoencoder.fit(
            train_tfidf, train_tfidf,
            epochs=self.epochs,
            shuffle=True,
            validation_data=(test_tfidf, test_tfidf)
        ).history.values()

        legend = [f'train rmse', f'test rmse']
        plt.plot(list(range(self.epochs)), autoencoder_train_losses, c='r', linestyle='--')
        plt.plot(list(range(self.epochs)), autoencoder_val_losses, c='r', linestyle='-')

        plt.xticks(list(range(1, self.epochs + 1, 2)), range(2, self.epochs + 2, 2))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Autoencoder losses over epochs')
        plt.legend(legend, loc='best')
        plt.savefig(os.path.join('data', 'images', f'autoencoder_losses.png'))
        plt.clf()

        self.item_embeddings = autoencoder.encoder(self.embeddings_matrix).numpy()
        self.user_embeddings = np.nan_to_num(self.train_ratings_matrix).dot(self.item_embeddings)
        self.predicted_ratings = 0.5 + 4.5 * (0.5 + cosine_similarity(self.user_embeddings, self.item_embeddings) / 2)
        print(self.evaluate())


class Autoencoder(Model):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dense(intermediate_dim, activation='relu'),
            layers.Dropout(0.2, seed=42),
            layers.BatchNormalization(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.Dense(intermediate_dim, activation='relu'),
            layers.Dropout(0.2, seed=42),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    params = {
        'intermediate_size': 5000,
        'encoded_size': 100,
        'epochs': 20,
    }
    model = ContentFiltering(
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
        movie_content_data=os.path.join(DATA_DIR, '1KK', 'movie_genome_scores.csv'),
        use_tfidf=False,
        params=params,
    )
    model.fit()

    model = ContentFiltering(fitted=True)
    model.evaluate(final=True)
