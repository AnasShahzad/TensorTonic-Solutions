import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    col = Counter(x)
    return np.nanmean(x), np.nanmedian(x), col.most_common(1)[0][0]