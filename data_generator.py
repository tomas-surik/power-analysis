import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from dataclasses import dataclass
from scipy.stats import bernoulli

@dataclass
class DataGenerator:
    """This class generates single observation of data."""
    max_nobs: int
    num_simulations: int
    alpha: float = 0.05

    def generate(self, random_state = None):
        
        np.random.seed(random_state)
        nobs_0 = np.random.randint(2, self.max_nobs + 1)
        nobs_1 = np.random.randint(2, self.max_nobs + 1)

        p_0 = np.round(np.random.uniform(), 4)
        p_1 = np.round(np.random.uniform(), 4)

        is_significant = []
        
        for _ in np.arange(self.num_simulations):
            count_0 = np.sum(bernoulli.rvs(p=p_0, size=nobs_0))
            count_1 = np.sum(bernoulli.rvs(p=p_1, size=nobs_1))
            _, p_value = proportions_ztest((count_0, count_1), (nobs_0, nobs_1), alternative='two-sided')
            is_significant.append(p_value < self.alpha)

        return (nobs_0, nobs_1, p_0, p_1), np.round(np.mean(is_significant), 4)