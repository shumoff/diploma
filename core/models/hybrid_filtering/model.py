import os

from core.config import DATA_DIR
from core.metrics import F1Score, NDCGScore, PRScore, RMSE, ROCScore, MAE
from core.models import ContentFiltering, Learner, SGDFactorization


class HybridFiltering(Learner):
    params = ['alpha']
    real_metrics = [RMSE, MAE, NDCGScore]
    class_metrics = [F1Score, PRScore, ROCScore]
    data_fields = ['train_ratings_matrix', 'test_ratings_matrix', 'predicted_ratings']

    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.model_1 = SGDFactorization(fitted=True, **kwargs)
        self.model_2 = ContentFiltering(fitted=True, **kwargs)
        if not self.fitted:
            self.train()

    def train(self):
        self.predicted_ratings = (
                (1 - self.alpha) * self.model_1.predicted_ratings + self.alpha * self.model_2.predicted_ratings
        )


if __name__ == '__main__':
    hybrid_model = HybridFiltering(
        alpha=0.3,
        fitted=True,
        train_ratings=os.path.join(DATA_DIR, '1KK', 'train_ratings.csv'),
        test_ratings=os.path.join(DATA_DIR, '1KK', 'test_ratings.csv'),
    )
    params_grid = {
        'alpha': [i / 10 for i in range(11)],
    }
    hybrid_model.find_best_params(params_grid)
