import os

from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models import MemoryBased, Learner


class MixedMemoryBased(Learner):
    params = ['alpha']
    real_metrics = [RMSE, MAE, NDCGScore]
    class_metrics = [F1Score, PRScore, ROCScore]
    data_fields = ['train_ratings_matrix', 'test_ratings_matrix', 'predicted_ratings']

    def __init__(self, model_1, model_2, alpha, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.model_1 = MemoryBased(params=model_1['params'], name=model_1['name'], **kwargs)
        self.model_2 = MemoryBased(params=model_2['params'], name=model_2['name'], **kwargs)
        if not self.fitted:
            self.model_1.train()
            self.model_2.train()

    def train(self):
        self.predicted_ratings = (
                (1 - self.alpha) * self.model_1.predicted_ratings + self.alpha * self.model_2.predicted_ratings
        )


if __name__ == '__main__':
    model_1_data = {
        'params': {
            'sim_users_amount': 100,
            'mean_centered': True,
            'standardized': True,
            'item_based': False,
        },
        'name': 'user_based',
        'fitted': True,
    }
    model_2_data = {
        'params': {
            'sim_users_amount': 100,
            'mean_centered': True,
            'standardized': True,
            'item_based': True,
        },
        'name': 'item_based',
        'fitted': True,
    }

    mixed_model = MixedMemoryBased(
        model_1_data,
        model_2_data,
        alpha=0.1,
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
    )
    mixed_model.fit()
    mixed_model.evaluate()
    params_grid = {
        'alpha': [i / 10 for i in range(11)],
    }
    mixed_model.find_best_params(params_grid)
    print(mixed_model.best_params)
