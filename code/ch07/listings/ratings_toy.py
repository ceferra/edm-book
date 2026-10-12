"""A small, made-up rating matrix (1-5 stars) and two similarity measures (Unit 7)."""
import numpy as np

nan = np.nan
FILMS = ["Space Saga", "Paris Love", "Heist Night", "Silent Forest", "Laugh Club",
         "Deep Ocean"]
USERS = ["Ana", "Ben", "Carla", "David", "Elena", "Fran", "Ivo"]
R = np.array([[4.5, 2.0, 4.0, 3.0, 2.5, 4.0],      # Ana
              [5.0, 1.5, 4.5, nan, 2.0, 3.5],      # Ben
              [2.0, 4.5, 2.5, 4.0, 4.5, nan],      # Carla
              [nan, 4.0, 2.0, 3.5, 5.0, 2.5],      # David
              [3.5, 3.0, nan, 4.5, 3.0, 4.0],      # Elena
              [4.0, nan, 3.5, 2.0, nan, 4.5],      # Fran
              [4.0, nan, nan, nan, 2.0, 4.5]])     # Ivo, who wants recommendations

def euclidean_similarity(a, b):
    """1 / (1 + Euclidean distance) over the items rated by both."""
    both = ~np.isnan(a) & ~np.isnan(b)
    return 1 / (1 + np.sqrt(np.sum((a[both] - b[both]) ** 2))) if both.any() else 0.0

def pearson(a, b):
    """Pearson correlation over the items rated by both (0 if undefined)."""
    both = ~np.isnan(a) & ~np.isnan(b)
    if both.sum() < 2 or a[both].std() == 0 or b[both].std() == 0:
        return 0.0
    return float(np.corrcoef(a[both], b[both])[0, 1])
