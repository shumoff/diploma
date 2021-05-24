import os

import numpy as np
import pandas as pd

from core.config import DATA_DIR

TEST_SAMPLE_PART = 0.2


def process_data(data_dir):
    dataset = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    n = 1000000
    dataset = dataset[:n]
    genome_scores = pd.read_csv(os.path.join(data_dir, 'genome-scores.csv'))
    dataset = dataset[dataset['movieId'].isin(genome_scores['movieId'].unique())]

    n_users = len(dataset['userId'].unique())
    n_movies = len(dataset['movieId'].unique())

    print(n_users, n_movies)

    while True:
        dataset = dataset.groupby("userId").filter(lambda x: len(x) >= 20)
        dataset = dataset.groupby("movieId").filter(lambda x: len(x) >= 20)
        new_n_users = len(dataset['userId'].unique())
        new_n_movies = len(dataset['movieId'].unique())
        if new_n_users == n_users and new_n_movies == n_movies:
            break
        n_users, n_movies = new_n_users, new_n_movies

    print(n_users, n_movies)
    dataset = dataset.drop(['timestamp'], axis=1)
    data_matrix, user_id_to_index, movie_id_to_index = get_data_matrix_from_dataset(dataset)

    data_matrix[np.isnan(data_matrix)] = 0
    train_matrix = data_matrix.copy()
    test_matrix = np.zeros((n_users, n_movies))
    for user_index in range(n_users):
        non_zero_ratings_indices = data_matrix[user_index].nonzero()[0]
        non_zero_ratings_count = len(non_zero_ratings_indices)

        test_ratings_indices = np.random.choice(
            non_zero_ratings_indices,
            size=int(non_zero_ratings_count * TEST_SAMPLE_PART),
            replace=False,
        )
        train_matrix[user_index, test_ratings_indices] = 0.
        test_matrix[user_index, test_ratings_indices] = data_matrix[user_index, test_ratings_indices]

    movie_index_to_id = {movie_index: movie_id for movie_id, movie_index in movie_id_to_index.items()}
    movies_count = len(movie_index_to_id)
    while True:
        for movie_index in np.where(~test_matrix.any(axis=0))[0]:
            movie_id = movie_index_to_id[movie_index]
            del movie_id_to_index[movie_id]
            for key in movie_id_to_index.keys():
                if movie_id_to_index[key] > movie_index:
                    movie_id_to_index[key] -= 1

        train_matrix = train_matrix[:, test_matrix.any(axis=0)]
        test_matrix = test_matrix[:, test_matrix.any(axis=0)]

        for movie_index in np.where(~train_matrix.any(axis=0))[0]:
            movie_id = movie_index_to_id[movie_index]
            del movie_id_to_index[movie_id]
            for key in movie_id_to_index.keys():
                if movie_id_to_index[key] > movie_index:
                    movie_id_to_index[key] -= 1

        test_matrix = test_matrix[:, train_matrix.any(axis=0)]
        train_matrix = train_matrix[:, train_matrix.any(axis=0)]
        if movies_count == len(movie_index_to_id):
            break
        movies_count = len(movie_index_to_id)

    train_matrix[train_matrix == 0] = np.nan
    test_matrix[test_matrix == 0] = np.nan
    train_df = get_dataset_from_data_matrix(train_matrix, user_id_to_index, movie_id_to_index)
    test_df = get_dataset_from_data_matrix(test_matrix, user_id_to_index, movie_id_to_index)

    movie_ids = list(movie_index_to_id.values())
    movies = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    movies = movies[movies['movieId'].isin(movie_ids)]
    tags = pd.read_csv(os.path.join(data_dir, 'tags.csv'))
    genome_scores = genome_scores[genome_scores['movieId'].isin(movie_ids)]

    tags = tags[tags['movieId'].isin(movie_ids)]
    tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(map(str, list(x)))).reset_index()
    tags = pd.merge(tags, movies, on="movieId", how="right").fillna('')

    tags['genres'] = tags['genres'].str.replace(pat='|', repl=' ')
    tags['document'] = tags[['tag', 'genres']].apply(' '.join, axis=1)
    tags.drop(['tag', 'genres'], axis=1, inplace=True)

    train_df.to_csv(os.path.join(data_dir, '1KK', 'train_ratings.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, '1KK', 'test_ratings.csv'), index=False)
    tags.to_csv(os.path.join(data_dir, '1KK', 'movie_tags.csv'), index=False)
    genome_scores.to_csv(os.path.join(data_dir, '1KK', 'movie_genome_scores.csv'), index=False)


def get_data_matrix_from_dataset(dataset, predefined=None):
    user_ids = list(dataset['userId'].unique())
    movie_ids = list(dataset['movieId'].unique())

    if predefined:
        n_users, n_movies = predefined['n_users'], predefined['n_movies']
        user_id_to_index, movie_id_to_index = predefined['user_id_to_index'], predefined['movie_id_to_index']
    else:
        n_users, n_movies = len(user_ids), len(movie_ids)
        user_id_to_index = {user_id: user_index for user_index, user_id in enumerate(user_ids)}
        movie_id_to_index = {movie_id: movie_index for movie_index, movie_id in enumerate(movie_ids)}

    data_matrix = np.full((n_users, n_movies), np.nan)
    for _, user_id, movie_id, rating in dataset.itertuples():
        user_index = user_id_to_index[user_id]
        movie_index = movie_id_to_index[movie_id]
        data_matrix[user_index][movie_index] = rating

    return data_matrix, user_id_to_index, movie_id_to_index


def get_dataset_from_data_matrix(data_matrix, user_id_to_index, movie_id_to_index):
    user_index_to_id = {user_index: user_id for user_id, user_index in user_id_to_index.items()}
    movie_index_to_id = {movie_index: movie_id for movie_id, movie_index in movie_id_to_index.items()}
    dataset = pd.DataFrame()
    data = []
    for user_index in user_index_to_id:
        for movie_index in movie_index_to_id:
            rating = data_matrix[user_index][movie_index]
            if np.isnan(rating):
                continue

            user_id, movie_id = user_index_to_id[user_index], movie_index_to_id[movie_index]
            data.append([user_id, movie_id, rating])

    df_row = pd.DataFrame(data, columns=['userId', 'movieId', 'rating'])
    dataset = dataset.append(df_row, ignore_index=True)

    return dataset


if __name__ == '__main__':
    process_data(os.path.join(DATA_DIR))
