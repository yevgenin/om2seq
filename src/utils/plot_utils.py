from matplotlib import pyplot as plt
from statsmodels.stats.proportion import proportion_confint


def plot_confidence_intervals(x_values, counts, totals, confidence_alpha=0.05, **kwargs):
    """
    Plot confidence intervals for a binomial distribution.

    :param x_values: x values
    :param counts: number of successes
    :param totals: number of trials
    :param confidence_alpha: confidence level
    :param kwargs: passed to plt.vlines
    """
    ci = proportion_confint(count=counts, nobs=totals, alpha=confidence_alpha, method='beta')
    plt.vlines(x_values, *ci, **kwargs)
